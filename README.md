# BrowSync

**Inferred eyebrow tracking for VRChat — no Quest Pro required.**

Estimates VRCFT Unified Expression brow parameters from eye tracking,
lower face tracking, and microphone prosody data using a rule-based
pipeline with a learned GRU residual model.

---

## Project Structure

```
browsync/
├── data/
│   ├── schema.py          # Feature definitions, BrowFrame, NormStats
│   └── sessions/          # Training data (.jsonl session files)
│       ├── train/
│       ├── val/
│       └── donated/       # Opt-in Quest Pro labelled sessions
├── inference/
│   ├── rules.py           # Rule-based baseline estimator (no ML)
│   └── smoother.py        # Spring-damper output smoother
├── models/
│   ├── gru_model.py       # GRU architecture + ONNX export
│   ├── checkpoints/       # PyTorch training checkpoints
│   └── browsync.onnx      # Production model (self-contained)
├── training/
│   └── train.py           # Full training pipeline
├── ws_server/
│   └── server.py          # WebSocket inference server
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run in rules-only mode (no trained model needed)
python -m ws_server.server

# Run with trained model
python ws_server/server.py --model models/browsync.onnx
```

Default WebSocket endpoint: `ws://localhost:7720`

---

## Input Features (41 total)

| Group | Features | Source |
|-------|----------|--------|
| Eye tracking | Openness, wide, squint, lid tightener, gaze, blink | Eye tracker (Vive, Pimax, etc.) |
| Lower face | Jaw, lip corners, cheek raise, lip stretch | VRCFT face tracker |
| Prosody | Pitch, energy, speech rate, VAD | Microphone |
| Emotion | Valence, arousal, confidence | SER model (optional) |
| Temporal | Delta features for key signals | Computed |

Missing features default to 0.0 — partial tracker setups are fully supported.

---

## Output Features (7 total)

All outputs are VRCFT Unified Expression parameters in [0, 1]:
- `BrowInnerUp`
- `BrowOuterUpLeft` / `BrowOuterUpRight`
- `BrowLowererLeft` / `BrowLowererRight`
- `BrowPinchLeft` / `BrowPinchRight`

---

## Architecture

```
Raw inputs → Rule-based estimator (fast, deterministic)
                    ↓
           + GRU residual (learned correction)
                    ↓
           Spring-damper smoother (per-AU physics)
                    ↓
           VRCFT brow parameters
```

The GRU learns *residuals* on top of the rule base, not the full mapping.
This means v0.1 (rules only) is already usable, and the ML model improves on top.

---

## Training

```bash
# Place .jsonl session files in data/sessions/train/ and data/sessions/val/
# Each line in a session file is a BrowFrame JSON object

python training/train.py
# Outputs: models/checkpoints/best_model.pt + models/browsync.onnx
```

Labelled sessions (Quest Pro users with opt-in enabled) have `has_labels: true`
and real brow AU targets. Unlabelled sessions use rule-based pseudo-labels
and contribute with reduced loss weight.

---

## WebSocket Protocol

See `ws_server/server.py` for full protocol documentation.

**Frame (client → server):**
```json
{
  "type": "frame",
  "ts": 1234567890.123,
  "inputs": {
    "EyeOpenessLeft": 0.82,
    "LipCornerPullLeft": 0.3
  }
}
```

**Response (server → client):**
```json
{
  "type": "brow",
  "ts": 1234567890.123,
  "outputs": {
    "BrowInnerUp": 0.34,
    "BrowOuterUpLeft": 0.28,
    "BrowOuterUpRight": 0.31,
    "BrowLowererLeft": 0.05,
    "BrowLowererRight": 0.06,
    "BrowPinchLeft": 0.02,
    "BrowPinchRight": 0.02
  },
  "mode": "ml"
}
```

---

## ONNX Model

The exported `browsync.onnx` is self-contained:
- Normalisation stats embedded in metadata
- Feature names embedded in metadata
- Single input: `float32[batch, 30, 41]`
- Single output: `float32[batch, 7]` (Tanh residuals)

The C# VRCFT module can load this directly via `Microsoft.ML.OnnxRuntime`
without any separate config files.

---

## Data Donation

Quest Pro users can opt in via the VRCFT module UI. Donated sessions are:
- Stored locally, never shared externally
- Paired input + ground-truth brow AU data
- Used to fine-tune the model on future releases
- Deletable at any time from the UI
