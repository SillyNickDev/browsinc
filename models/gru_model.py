"""
BrowSync — GRU model definition and ONNX export.

Architecture:
  Input:  (batch, SEQUENCE_LENGTH, NUM_INPUTS)   — sliding window of frames
  GRU:    2 layers, hidden_size=64, dropout=0.2
  Head:   Linear(64 → 32) → ReLU → Linear(32 → NUM_BROW_OUTPUTS) → Sigmoid

  The model predicts RESIDUALS on top of the rule-based estimator output,
  so the final output is:  sigmoid(gru_out) + rule_estimate, clipped to [0,1].
  This means the model only needs to learn corrections, not the full mapping.

ONNX export targets opset 17 and embeds normalisation stats + BROW_OUTPUTS
names in the model metadata so the C# consumer doesn't need a separate config.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from data.schema import (
    NUM_INPUTS, NUM_BROW_OUTPUTS, SEQUENCE_LENGTH,
    BROW_OUTPUTS, ALL_INPUT_FEATURES, NormStats
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BrowSyncGRU(nn.Module):
    """
    Lightweight GRU for brow AU residual prediction.
    Designed to run at 90fps on CPU — kept deliberately small.
    """

    def __init__(
        self,
        input_size: int = NUM_INPUTS,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = NUM_BROW_OUTPUTS,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Input projection — helps the GRU by giving it a denser representation
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=48,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Prediction head — only uses the last hidden state
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, output_size),
            nn.Tanh(),   # outputs residuals in (-1, 1); rule base handles [0,1] range
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch, output_size)  — residuals in (-1, 1)
        """
        # Project each timestep
        batch, seq, _ = x.shape
        x_flat = x.view(batch * seq, -1)
        proj = self.input_proj(x_flat).view(batch, seq, -1)

        # GRU — we only care about the last timestep's hidden state
        _, h_n = self.gru(proj)          # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]            # (batch, hidden)

        residuals = self.head(last_hidden)   # (batch, output_size)
        return residuals

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Combined inference: rule base + GRU residual
# ---------------------------------------------------------------------------

class BrowSyncInference:
    """
    Wraps the GRU model + rule estimator for live inference.
    This is what the WebSocket server calls every frame.
    """

    def __init__(self, model: BrowSyncGRU, norm_stats: NormStats, residual_scale: float = 0.4):
        self.model = model
        self.norm_stats = norm_stats
        self.residual_scale = residual_scale  # how much the ML residual contributes
        self.model.eval()

        # Rolling window buffer
        self._buffer = np.zeros((SEQUENCE_LENGTH, NUM_INPUTS), dtype=np.float32)

    def push_frame(self, raw_inputs: np.ndarray, rule_estimate: np.ndarray) -> np.ndarray:
        """
        Push one frame and return combined brow estimate.

        raw_inputs: (NUM_INPUTS,)      — unnormalised input features
        rule_estimate: (NUM_BROW_OUTPUTS,) — output from RuleBasedEstimator
        returns: (NUM_BROW_OUTPUTS,)   — final brow values in [0, 1]
        """
        normed = self.norm_stats.normalise(raw_inputs)

        # Shift buffer left and append new frame
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = normed

        seq = torch.tensor(self._buffer[np.newaxis], dtype=torch.float32)  # (1, seq, feat)

        with torch.no_grad():
            residuals = self.model(seq).squeeze(0).numpy()   # (NUM_BROW_OUTPUTS,)

        combined = rule_estimate + residuals * self.residual_scale
        return np.clip(combined, 0.0, 1.0).astype(np.float32)

    def reset_buffer(self):
        self._buffer[:] = 0.0


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model: BrowSyncGRU,
    norm_stats: NormStats,
    output_path: Path,
    opset: int = 17,
):
    """
    Export the trained GRU to ONNX with embedded metadata.

    The ONNX file is self-contained:
      - Normalisation mean/std embedded as metadata strings
      - Input/output feature names embedded as metadata
      - Single input: float32[batch, SEQUENCE_LENGTH, NUM_INPUTS]
      - Single output: float32[batch, NUM_BROW_OUTPUTS]  (residuals, Tanh)

    The C# consumer should:
      1. Load mean/std from metadata, normalise its rolling window
      2. Run GRU inference to get residuals
      3. Add residuals (scaled) to its own rule-based estimate
      4. Clip to [0, 1] and write to VRCFT parameters
    """
    model.eval()

    dummy_input = torch.zeros(1, SEQUENCE_LENGTH, NUM_INPUTS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input_sequence"],
        output_names=["brow_residuals"],
        dynamic_axes={
            "input_sequence": {0: "batch_size"},
            "brow_residuals": {0: "batch_size"},
        },
    )

    # Embed metadata using onnx library
    import onnx
    model_proto = onnx.load(str(output_path))

    def add_meta(key: str, value: str):
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = value

    add_meta("browsync_version", "0.1.0")
    add_meta("sequence_length", str(SEQUENCE_LENGTH))
    add_meta("num_inputs", str(NUM_INPUTS))
    add_meta("num_outputs", str(NUM_BROW_OUTPUTS))
    add_meta("input_features", json.dumps(ALL_INPUT_FEATURES))
    add_meta("output_features", json.dumps(BROW_OUTPUTS))
    add_meta("norm_mean", json.dumps(norm_stats.mean.tolist()))
    add_meta("norm_std", json.dumps(norm_stats.std.tolist()))
    add_meta("output_type", "residuals_tanh")  # tells C# consumer how to combine

    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, str(output_path))

    size_kb = output_path.stat().st_size / 1024
    print(f"[BrowSync] Exported ONNX model → {output_path}  ({size_kb:.1f} KB)")
    print(f"[BrowSync] Parameters: {model.count_parameters():,}")

    return output_path


def load_from_onnx_metadata(onnx_path: Path) -> tuple[NormStats, list, list]:
    """
    Load normalisation stats and feature names from ONNX metadata.
    Returns (NormStats, input_feature_names, output_feature_names).
    """
    import onnx
    model_proto = onnx.load(str(onnx_path))
    meta = {e.key: e.value for e in model_proto.metadata_props}

    norm = NormStats(
        mean=np.array(json.loads(meta["norm_mean"]), dtype=np.float32),
        std=np.array(json.loads(meta["norm_std"]), dtype=np.float32),
    )
    inputs = json.loads(meta["input_features"])
    outputs = json.loads(meta["output_features"])
    return norm, inputs, outputs
