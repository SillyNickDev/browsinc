# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BrowSync is a real-time eyebrow tracking system for VRChat using a hybrid rule-based + ML approach. It estimates VRCFT Unified Expression brow parameters from eye tracking, lower face tracking, and microphone prosody data — enabling expressive brow animation without a Quest Pro headset.

## Two-component system

BrowSync has two independent pieces that work together:

1. **Python server** (`ws_server/`) — runs inference at 90fps, exposes a WebSocket on port 7720
2. **C# VRCFT plugin** (`BrowSyncModule/`) — connects to the server, writes inferred brow shapes into VRCFT so they reach VRChat via OSC

Start the Python server first, then launch VRCFT.

## Commands

### Python server

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference server (rules-only, no TUI)
python -m ws_server.server

# Run inference server with ONNX model and TUI
python -m ws_server.server --model models/browsync.onnx

# Headless mode (no TUI, log to console)
python -m ws_server.server --model models/browsync.onnx --no-tui

# Train model (requires labelled session data in data/sessions/)
python training/train.py

# Launch TUI standalone (server embedded inside TUI)
python TUI/tui.py
```

### C# VRCFT module

```bash
cd BrowSyncModule
dotnet build -c Release
# Post-build step copies DLL to %APPDATA%\VRCFaceTracking\CustomLibs\ automatically
```

No dedicated test runner or lint config is present in this repo.

## Architecture

### Data Flow

```
Input sources (90fps loop):
  VRCFT client         → 14 eye + 13 face features    (WebSocket, push from client)
  MicrophoneProcessor  → 6 prosody features            (background thread, 16kHz audio)
  HeadMotionTracker    → 11 head motion features       (background thread, OpenXR poll)
  Emotion context      → 3 features (optional SER)
  Delta features       → 5 features (computed in-place)
                         = 52 total inputs (schema.py)

Assembly (ws_server/server.py InferenceEngine)
  → RuleBasedEstimator (inference/rules.py)     deterministic baseline → 8 AU values
  → BrowSyncGRU residual (models/gru_model.py)  adds correction in (-1,1) range, scaled 0.4x
  → BrowSmoother (inference/smoother.py)        spring-damper physics per AU
  → 8 VRCFT Unified Expression brow AU outputs  [0, 1]

Output pushed via asyncio queues to all connected WebSocket clients
```

### Inference Modes (automatic fallback)

The server selects the best available mode at runtime:

| Mode | Sources |
|------|---------|
| `ml` | eye + face + mic + head + GRU model |
| `rules_only` | eye + face + mic + head, rule base only |
| `mic_head` | mic + head only (no face/eye tracker) |
| `head_only` | head motion only |
| `noise_only` | procedural anti-freeze noise |

Mode degrades automatically when data sources become unavailable or stale (VRCFT data expires after 0.5s).

### Key Modules

**`data/schema.py`** — Single source of truth for all data shapes. Defines the 52 input features (indices, names, normalization), 8 output AUs, `BrowFrame` dataclass (one timestep), `BrowSequence` (30-frame sliding window = 333ms context), and `NormStats` (z-score params embedded in ONNX metadata).

**`inference/rules.py`** — `RuleBasedEstimator`: fully deterministic eye/face/prosody/emotion → AU mapping. `RuleWeights` dataclass holds all tunable coupling parameters. Runs every frame even in `ml` mode (GRU predicts residuals on top).

**`models/gru_model.py`** — `BrowSyncGRU`: input projection 52→48, 2-layer GRU (hidden=64), residual head 64→32→8 with Tanh. ~15K parameters, designed for CPU inference. `BrowSyncInference` wraps the model with a rolling 30-frame buffer and combines rule + residual outputs. ONNX export embeds norm_stats and feature names as metadata.

**`ws_server/server.py`** — Core server. `InferenceEngine` runs the 90fps clock in a background thread, assembles frames from all sources, and pushes outputs to asyncio queues. `ClientSession` tracks per-client VRCFT inputs and donation state. Handles control messages: `ping`, `reset`, `recalibrate_head`, `set_mode`, `get_status`.

**`inference/microphone.py`** — Background thread; extracts PitchNorm, PitchDelta, EnergyNorm, EnergyDelta, SpeechRate, IsSpeaking via librosa. Optional `MicrophoneProcessorWithSER` integrates SpeechBrain for emotion context.

**`inference/head_motion.py`** — Background thread polling OpenXR at ~90fps. Calibrates neutral pose from the first 2.5s average. Outputs pitch/roll/yaw, Y/Z translation, linear/angular velocity and acceleration, normalised to [-1,1].

**`inference/smoother.py`** — Per-AU second-order spring-damper with asymmetric attack/decay (raises fast, lowers slow). Maintains position and velocity state between frames.

**`training/train.py`** — `BrowSequenceDataset` loads `.jsonl` sessions with stride-3 sliding windows. `BrowSyncLoss` = MSE + temporal smoothness penalty + asymmetric per-output weights (raises 1.3x). Unlabelled (rule pseudo-label) frames contribute at 0.25 weight. Exports final model to ONNX with embedded normalization stats. Config: 60 epochs, batch=64, LR=3e-4, early stop patience=15.

**`TUI/tui.py`** — Textual app embedding the server as an asyncio worker. `InferencePanel` shows mode (color-coded), FPS, per-AU bar meters, and source status indicators. Keys: `q`=quit, `r`=recalibrate head, `Ctrl+L`=dev log.

**`BrowSyncModule/`** — C# VRCFT plugin (net7.0). `BrowSyncClient.cs` maintains a persistent WebSocket connection to the Python server with auto-reconnect (3s backoff) and a 5s ping keepalive. `BrowSyncModule.cs` inherits `ExtTrackingModule`, declares `SupportsExpression: true` / `SupportsEye: false`, and writes the 8 brow `UnifiedExpressions` shapes at ~90fps. Only brow shapes are written — eye and lower-face are left to the user's existing VRCFT modules. On disconnect, brow shapes are zeroed. On reconnect, a `reset` is sent to clear the Python server's GRU buffer. All connection config (host, port, timeouts) is hardcoded in those two files.

### Data Formats

**Session files** (`data/sessions/{train,val,donated}/`, `.jsonl`): one `BrowFrame` JSON per line.
```json
{"timestamp_ms": 123.4, "inputs": [52 floats], "targets": [8 floats], "has_labels": true, "session_id": "uuid"}
```
`targets` and `has_labels` are absent for unlabelled sessions.

**WebSocket protocol**:
- Client → Server: `{"type": "frame", "ts": <ms>, "inputs": {<feature_name>: <value>, ...}}`
- Server → Client: `{"type": "brow", "ts": <ms>, "outputs": {<au_name>: <value>, ...}, "mode": "ml", "head_tracking": true, "mic_active": true}`

### Design Constraints

- All inputs are z-score normalised using stats computed from the training set and embedded in the ONNX file — do not change input scaling without retraining and re-exporting.
- The GRU residual is scaled by 0.4 before adding to the rule estimate; changing this affects how aggressively the model overrides the rule base.
- Sequence length is fixed at 30 frames (one `SEQUENCE_LENGTH` constant in `schema.py`); changing it requires re-training.
- The ONNX model at `models/browsync.onnx` is self-contained (weights split into `.data` sidecar); both files must stay together.
- `data/sessions/donated/` is gitignored — it holds opt-in Quest Pro user data for fine-tuning.
