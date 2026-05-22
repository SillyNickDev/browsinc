"""
BrowSync — Training pipeline.

Data flow:
  Raw session .jsonl files (each line = one BrowFrame JSON)
    → BrowSequenceDataset (sliding window, normalisation)
      → BrowSyncGRU trained with combined MSE + temporal smoothness loss
        → ONNX export

Session files come from two sources:
  1. Quest Pro labelled sessions: has_labels=True, targets = real brow AUs
  2. Unlabelled sessions: has_labels=False, targets = rule-based estimates
     (used for self-supervised pre-training only)

The training loop prioritises labelled data but can use unlabelled data
with a reduced loss weight for regularisation.
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from data.schema import (
    BrowFrame, BrowSequence, NormStats,
    NUM_INPUTS, NUM_BROW_OUTPUTS, SEQUENCE_LENGTH
)
from models.gru_model import BrowSyncGRU, export_to_onnx
from inference.rules import RuleBasedEstimator


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BrowSequenceDataset(Dataset):
    """
    Loads .jsonl session files and serves sliding-window sequences.

    Each .jsonl file is one recording session (one line per frame).
    Labelled sessions (Quest Pro) get sample_weight=1.0.
    Unlabelled sessions get sample_weight=unlabelled_weight.
    """

    def __init__(
        self,
        session_paths: List[Path],
        norm_stats: NormStats,
        sequence_length: int = SEQUENCE_LENGTH,
        stride: int = 3,                    # step between windows (at 90fps, stride=3 → 30fps effective sampling)
        unlabelled_weight: float = 0.25,    # loss weight for rule-estimated targets
        augment: bool = True,
    ):
        self.norm_stats = norm_stats
        self.seq_len = sequence_length
        self.augment = augment

        self.sequences: List[BrowSequence] = []
        self.sample_weights: List[float] = []

        rule_estimator = RuleBasedEstimator()

        for path in session_paths:
            frames = self._load_session(path, rule_estimator)
            if len(frames) < sequence_length + 1:
                continue

            weight = 1.0 if frames[0].has_labels else unlabelled_weight

            for start in range(0, len(frames) - sequence_length, stride):
                window = frames[start: start + sequence_length]
                label_frame = frames[start + sequence_length - 1]

                seq = BrowSequence(
                    frames=np.stack([f.inputs for f in window]),   # (seq, feat)
                    target=label_frame.targets,
                    has_labels=label_frame.has_labels,
                    session_id=label_frame.session_id,
                )
                self.sequences.append(seq)
                self.sample_weights.append(weight)

    def _load_session(self, path: Path, rule_est: RuleBasedEstimator) -> List[BrowFrame]:
        frames = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frame = BrowFrame.from_dict(json.loads(line))

                # For unlabelled frames, fill targets from rule estimator
                if not frame.has_labels:
                    frame.targets = rule_est.estimate(frame)

                frames.append(frame)
        return frames

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        seq = self.sequences[idx]
        weight = self.sample_weights[idx]

        # Normalise inputs
        normed = self.norm_stats.normalise(seq.frames)   # (seq, feat)

        # Augmentation (training only)
        if self.augment:
            normed = self._augment(normed)

        x = torch.tensor(normed, dtype=torch.float32)
        y = torch.tensor(seq.target, dtype=torch.float32)
        return x, y, weight

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """
        Lightweight augmentation to improve generalisation:
        - Gaussian noise on inputs
        - Random time-warp (stretch/compress slightly)
        - Random feature dropout (simulates missing tracker data)
        """
        x = x.copy()

        # Gaussian noise
        if random.random() < 0.7:
            x += np.random.normal(0, 0.02, x.shape).astype(np.float32)

        # Feature dropout — randomly zero some input channels for a frame range
        if random.random() < 0.3:
            drop_feat = random.randint(0, x.shape[1] - 1)
            x[:, drop_feat] = 0.0

        # Temporal jitter — slight random shift of the sequence start
        if random.random() < 0.2 and x.shape[0] > 4:
            shift = random.randint(1, 3)
            x = np.concatenate([x[shift:], x[-shift:]], axis=0)

        return x

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        weights = torch.tensor(self.sample_weights, dtype=torch.float32)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class BrowSyncLoss(nn.Module):
    """
    Combined loss for brow AU prediction.

    Components:
      1. MSE on predicted vs target AU values
      2. Temporal smoothness penalty — penalises jittery frame-to-frame changes
         (computed across batch by comparing adjacent samples in the batch,
          which approximates temporal order for shuffled batches)
      3. Asymmetric penalty — raising brows incorrectly is more visible than
         lowering them slightly wrong, so we weight raise errors more
    """

    def __init__(
        self,
        smoothness_weight: float = 0.1,
        raise_error_weight: float = 1.3,   # BrowInnerUp, BrowOuterUp outputs weighted higher
    ):
        super().__init__()
        self.smoothness_weight = smoothness_weight

        # Build per-output asymmetric weights
        from data.schema import OUTPUT_INDEX
        weights = torch.ones(NUM_BROW_OUTPUTS)
        for name in ["BrowInnerUpLeft", "BrowInnerUpRight", "BrowOuterUpLeft", "BrowOuterUpRight"]:
            weights[OUTPUT_INDEX[name]] = raise_error_weight
        self.register_buffer("output_weights", weights)

    def forward(
        self,
        pred: torch.Tensor,         # (batch, NUM_BROW_OUTPUTS)  residuals
        rule: torch.Tensor,         # (batch, NUM_BROW_OUTPUTS)  rule base
        target: torch.Tensor,       # (batch, NUM_BROW_OUTPUTS)  ground truth
        sample_weights: torch.Tensor,  # (batch,)
        residual_scale: float = 0.4,
    ) -> Tuple[torch.Tensor, dict]:

        # Combined prediction
        combined = torch.clamp(rule + pred * residual_scale, 0.0, 1.0)

        # Weighted MSE
        sq_err = (combined - target) ** 2                   # (batch, outputs)
        weighted_err = sq_err * self.output_weights          # per-output weights
        sample_weighted = weighted_err.mean(dim=1) * sample_weights
        mse_loss = sample_weighted.mean()

        # Temporal smoothness — penalise large consecutive differences
        if combined.shape[0] > 1:
            diffs = (combined[1:] - combined[:-1]) ** 2
            smooth_loss = diffs.mean()
        else:
            smooth_loss = torch.tensor(0.0)

        total = mse_loss + self.smoothness_weight * smooth_loss

        return total, {
            "mse": mse_loss.item(),
            "smoothness": smooth_loss.item(),
            "total": total.item(),
        }


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    train_session_dir: Path = Path("data/sessions/train")
    val_session_dir: Path = Path("data/sessions/val")
    output_dir: Path = Path("models/checkpoints")
    onnx_output: Path = Path("models/browsync.onnx")

    # Model
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    residual_scale: float = 0.4

    # Training
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lr_patience: int = 8        # ReduceLROnPlateau patience
    early_stop_patience: int = 15
    grad_clip: float = 1.0

    # Loss
    smoothness_weight: float = 0.1
    raise_error_weight: float = 1.3
    unlabelled_weight: float = 0.25

    # Misc
    seed: int = 42
    num_workers: int = 2


# ---------------------------------------------------------------------------
# Norm stats computation
# ---------------------------------------------------------------------------

def compute_norm_stats(session_paths: List[Path]) -> NormStats:
    """Compute per-feature mean and std from all training sessions."""
    all_inputs = []
    for path in session_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    frame = BrowFrame.from_dict(json.loads(line))
                    all_inputs.append(frame.inputs)

    if not all_inputs:
        print("[BrowSync] WARNING: No training data found, using identity norm stats")
        return NormStats.identity()

    data = np.stack(all_inputs)   # (N, NUM_INPUTS)
    mean = data.mean(axis=0).astype(np.float32)
    std = data.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)   # avoid division by zero for constant features
    return NormStats(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: TrainConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")   # intentionally CPU — model must run in VRChat on CPU
    print(f"[BrowSync] Training on {device}")

    # Collect session files
    train_paths = list(config.train_session_dir.glob("*.jsonl"))
    val_paths = list(config.val_session_dir.glob("*.jsonl"))
    print(f"[BrowSync] Sessions: {len(train_paths)} train, {len(val_paths)} val")

    if not train_paths:
        print("[BrowSync] No training data found. Run data collection first.")
        return

    # Compute normalisation stats from training data only
    norm_stats = compute_norm_stats(train_paths)
    norm_path = config.output_dir / "norm_stats.json"
    with open(norm_path, "w") as f:
        json.dump(norm_stats.to_dict(), f)
    print(f"[BrowSync] Norm stats saved → {norm_path}")

    # Datasets
    train_ds = BrowSequenceDataset(
        train_paths, norm_stats, augment=True,
        unlabelled_weight=config.unlabelled_weight
    )
    val_ds = BrowSequenceDataset(
        val_paths, norm_stats, augment=False,
        unlabelled_weight=config.unlabelled_weight
    )
    print(f"[BrowSync] Sequences: {len(train_ds)} train, {len(val_ds)} val")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=train_ds.get_weighted_sampler(),
        num_workers=config.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )

    # Model
    model = BrowSyncGRU(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    print(f"[BrowSync] Model parameters: {model.count_parameters():,}")

    # Optimiser + scheduler
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=config.lr_patience, factor=0.5
    )
    criterion = BrowSyncLoss(
        smoothness_weight=config.smoothness_weight,
        raise_error_weight=config.raise_error_weight,
    )

    # Rule estimator — used to compute rule base for loss during training
    rule_est = RuleBasedEstimator()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        # -- Train -----------------------------------------------------------
        model.train()
        train_losses = []

        for x, y, w in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)

            # Compute rule base for this batch
            # (we use the last frame of each sequence as the rule estimate)
            # In practice this runs fast since rule_est is pure numpy
            rule_batch = []
            for i in range(x.shape[0]):
                # Denormalise last frame back to raw inputs for rule estimator
                last_frame_normed = x[i, -1].cpu().numpy()
                last_raw = last_frame_normed * norm_stats.std + norm_stats.mean
                dummy_frame = BrowFrame(
                    timestamp_ms=0, inputs=last_raw, has_labels=False
                )
                rule_batch.append(rule_est.estimate(dummy_frame))
            rule_tensor = torch.tensor(
                np.stack(rule_batch), dtype=torch.float32, device=device
            )

            optimiser.zero_grad()
            pred = model(x)
            loss, info = criterion(pred, rule_tensor, y, w, config.residual_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            train_losses.append(info["total"])

        # -- Validate --------------------------------------------------------
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y, w in val_loader:
                x, y, w = x.to(device), y.to(device), w.to(device)

                rule_batch = []
                for i in range(x.shape[0]):
                    last_raw = x[i, -1].cpu().numpy() * norm_stats.std + norm_stats.mean
                    dummy = BrowFrame(timestamp_ms=0, inputs=last_raw, has_labels=False)
                    rule_batch.append(rule_est.estimate(dummy))
                rule_tensor = torch.tensor(np.stack(rule_batch), dtype=torch.float32).to(device)

                pred = model(x)
                _, info = criterion(pred, rule_tensor, y, w, config.residual_scale)
                val_losses.append(info["total"])

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        print(
            f"[BrowSync] Epoch {epoch+1:3d}/{config.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {optimiser.param_groups[0]['lr']:.2e}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "norm_stats": norm_stats.to_dict(),
                    "config": config.__dict__,
                },
                config.output_dir / "best_model.pt",
            )
            print(f"[BrowSync]   ✓ New best model saved (val={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"[BrowSync] Early stopping at epoch {epoch+1}")
                break

    # -- Export best model to ONNX -------------------------------------------
    checkpoint = torch.load(config.output_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    export_to_onnx(model, norm_stats, config.onnx_output)
    print(f"[BrowSync] Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train(TrainConfig())
