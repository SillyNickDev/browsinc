# BrowSync

**Inferred eyebrow tracking for VRChat — no Quest Pro required.**

Estimates VRCFT Unified Expression brow parameters from eye tracking, lower face tracking, and microphone prosody data using a rule-based pipeline with a learned GRU residual model. Works with any OpenXR-compatible headset.

---

## Requirements

- Python 3.11+
- An OpenXR runtime active before launch (SteamVR, Oculus, WMR, Pico, or any compliant runtime)
- [VRCFaceTracking](https://github.com/benaclejames/VRCFaceTracking) (VRCFT) for eye/face input
- Microphone (optional, improves prosody-driven brow movement)

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Rules-only mode — no trained model needed
python -m ws_server.server

# With ONNX model and terminal UI
python -m ws_server.server --model models/browsync.onnx

# Headless (log to console, no TUI)
python -m ws_server.server --model models/browsync.onnx --no-tui

# Custom port (default: 7720)
python -m ws_server.server --port 7721
```

> **Note:** SteamVR (and most OpenXR runtimes) require a valid graphics context even for pose-only use. Start your VR application before launching BrowSync, or head motion tracking will be disabled and the server will fall back to a lower inference mode.

---

## VRCFT Module

`BrowSyncModule/` contains the C# plugin that connects VRCFT to the Python server. It handles the WebSocket connection, receives inferred brow values, and writes them into VRCFT's `UnifiedTracking` shape buffer so VRChat picks them up via OSC.

The module deliberately only writes brow shapes — it does **not** claim eye tracking or lower-face data. This means it stacks cleanly with whatever eye/face tracker you already use in VRCFT.

### Prerequisites

- [VRCFaceTracking](https://github.com/benaclejames/VRCFaceTracking) v5+
- .NET 7 SDK

### Build and install

```bash
cd BrowSyncModule
dotnet build -c Release
```

The post-build step automatically copies `BrowSyncModule.dll` to `%APPDATA%\VRCFaceTracking\CustomLibs\`. VRCFT discovers modules there on startup. To install manually, copy the DLL yourself:

```
BrowSyncModule\bin\Release\net7.0\BrowSyncModule.dll
  → %APPDATA%\VRCFaceTracking\CustomLibs\BrowSyncModule.dll
```

### Usage

1. Start the Python server (see [Quick Start](#quick-start))
2. Launch VRCFaceTracking — the module loads automatically and connects to `ws://localhost:7720`
3. If the server isn't running yet, the module will retry in the background and connect once it's up

The module sends a `reset` on connect to clear the GRU buffer and pings every 5 seconds to keep the connection alive. If the Python server disconnects, brow shapes are zeroed and the module reconnects automatically with a 3-second backoff.

> The host and port are hardcoded to `localhost:7720`. If you need to change them, edit `BrowSyncModule.cs` and rebuild.

---

## Terminal UI

The TUI launches automatically unless `--no-tui` is passed. It shows live inference mode, FPS, per-AU bar meters, and source status.

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Recalibrate head tracking neutral pose |
| `Ctrl+L` | Toggle dev log |

---

## Inference Modes

The server selects the best available mode automatically and degrades gracefully as sources become unavailable. VRCFT data is considered stale after 0.5 seconds.

| Mode | Eye + Face | Mic | Head | ML Model |
|------|-----------|-----|------|----------|
| `ml` | ✓ | ✓ | ✓ | ✓ |
| `rules_only` | ✓ | ✓ | ✓ | — |
| `mic_head` | — | ✓ | ✓ | — |
| `head_only` | — | — | ✓ | — |
| `noise_only` | — | — | — | — |

`noise_only` outputs procedural animation to prevent the avatar from freezing when no sources are available.

---

## Input Features (52 total)

| Group | Count | Features |
|-------|-------|---------|
| Eye tracking | 14 | Openness, wide, squint, lid tightener, gaze X/Y, blink (per side) |
| Lower face | 13 | Jaw, lip corners, cheek raise, lip stretch, and others |
| Prosody | 6 | PitchNorm, PitchDelta, EnergyNorm, EnergyDelta, SpeechRate, IsSpeaking |
| Emotion | 3 | Valence, arousal, confidence (requires SpeechBrain, optional) |
| Temporal deltas | 5 | Computed from key signals each frame |
| Head motion | 11 | Pitch/roll/yaw, Y/Z translation, linear/angular velocity and acceleration |

Missing features default to `0.0` — partial tracker setups are fully supported.

---

## Output Features (8 total)

All outputs are VRCFT Unified Expression parameters in `[0, 1]`:

- `BrowInnerUpLeft` / `BrowInnerUpRight`
- `BrowOuterUpLeft` / `BrowOuterUpRight`
- `BrowLowererLeft` / `BrowLowererRight`
- `BrowPinchLeft` / `BrowPinchRight`

---

## WebSocket Protocol

Default endpoint: `ws://localhost:7720`

### Frame (client → server)

Send eye/face feature values from VRCFT. Only include features that are available — missing keys default to `0.0`.

```json
{
  "type": "frame",
  "ts": 1234567890.123,
  "inputs": {
    "EyeOpenessLeft": 0.82,
    "EyeOpenessRight": 0.79,
    "LipCornerPullLeft": 0.3
  }
}
```

### Brow output (server → client)

Pushed at ~90fps regardless of whether the client sends frames.

```json
{
  "type": "brow",
  "ts": 1234567890.123,
  "outputs": {
    "BrowInnerUpLeft": 0.34,
    "BrowInnerUpRight": 0.34,
    "BrowOuterUpLeft": 0.28,
    "BrowOuterUpRight": 0.31,
    "BrowLowererLeft": 0.05,
    "BrowLowererRight": 0.06,
    "BrowPinchLeft": 0.02,
    "BrowPinchRight": 0.02
  },
  "mode": "ml",
  "head_tracking": true,
  "mic_active": true
}
```

### Control messages

| Message | Description | Acknowledgement |
|---------|-------------|-----------------|
| `{"type": "ping"}` | Keepalive | `{"type": "pong"}` |
| `{"type": "reset"}` | Clear GRU buffer and recalibrate head | `{"type": "reset_ack"}` |
| `{"type": "recalibrate_head"}` | Restart head neutral pose calibration | `{"type": "recalibrate_head_ack"}` |
| `{"type": "set_mode", "mode": "rules_only"}` | Force a specific inference mode | `{"type": "mode_ack", "mode": "..."}` |
| `{"type": "get_status"}` | Request current server status | `{"type": "status", ...}` |

---

## Architecture

```
Input sources (90fps polling loop):
  VRCFT client         →  14 eye + 13 face features   (WebSocket frames from client)
  MicrophoneProcessor  →  6 prosody features           (background thread, 16kHz audio)
  HeadMotionTracker    →  11 head motion features      (background thread, OpenXR poll)
  Emotion context      →  3 features                   (optional SpeechBrain SER)
  Delta features       →  5 features                   (computed per-frame)
                          ─────────────────────────────
                          52 inputs total

  → RuleBasedEstimator    deterministic baseline → 8 AU values [0, 1]
  → GRU residual          learned correction in (−1, 1), scaled by 0.4×
  → Spring-damper smoother  per-AU physics (asymmetric attack/decay)
  → 8 VRCFT brow AU outputs, pushed to all connected clients
```

The GRU learns *residuals* on top of the rule base, not the full mapping. This means rules-only mode is immediately usable, and the ML model improves on top incrementally.

Head tracking calibrates on the first 2.5 seconds of runtime, treating the average pose as neutral. All subsequent values are offsets from that baseline.

---

## Training

```bash
# Place .jsonl session files in:
#   data/sessions/train/
#   data/sessions/val/

python training/train.py
# Outputs: models/checkpoints/best_model.pt + models/browsync.onnx
```

Each line in a session file is a `BrowFrame` JSON object:

```json
{"timestamp_ms": 123.4, "inputs": [52 floats], "targets": [8 floats], "has_labels": true, "session_id": "uuid"}
```

Sessions without Quest Pro ground truth omit `targets` and set `has_labels: false`. They are trained with rule-based pseudo-labels at a reduced loss weight (`0.25`).

Training config: 60 epochs, batch size 64, LR 3e-4, early stop patience 15. Loss = MSE + temporal smoothness penalty + asymmetric output weights (brow raises weighted 1.3× vs lowers).

---

## ONNX Model

The exported `models/browsync.onnx` is self-contained:

- Normalisation stats (mean/std per feature) embedded in ONNX metadata
- Feature names embedded in ONNX metadata
- Input: `float32[batch, 30, 52]` — 30-frame window at 90fps (≈333ms context)
- Output: `float32[batch, 8]` — Tanh residuals, scaled 0.4× at inference time

The companion `models/browsync.onnx.data` file holds the weights — both files must remain together.

---

## Data Donation

Quest Pro users can opt in via the VRCFT module UI. Donated sessions provide real brow AU ground truth paired with the full input feature vector. They are:

- Stored locally, never shared externally
- Used to fine-tune future model releases
- Deletable at any time from the UI
- Saved to `data/sessions/donated/` (gitignored)
