# Changelog

## v0.1.5 — *Lively Lodjur*

> "Lodjur" is Swedish for lynx — an animal renowned for its sharp, expressive eyes. Fitting for a first release.

### Initial Release

BrowSync is a real-time eyebrow tracking system for VRChat that estimates VRCFT Unified Expression brow parameters without requiring a Quest Pro headset. It uses a hybrid rule-based + ML approach, combining eye tracking, lower face tracking, and microphone prosody to drive expressive brow animation at 90fps.

---

### Features

**Inference Pipeline**
- Hybrid architecture: deterministic rule base (`RuleBasedEstimator`) plus a lightweight GRU residual model (~15K parameters) for learned corrections
- 52-feature input schema spanning eye/face tracking, microphone prosody, head motion, and computed deltas
- 8 VRCFT Unified Expression brow output AUs, all clamped to [0, 1]
- Spring-damper smoother with asymmetric attack/decay (raises fast, lowers slow) per AU

**Automatic Mode Fallback**
The server selects the best available inference mode at runtime and degrades gracefully as sources go offline:

| Mode | Sources active |
|------|---------------|
| `ml` | Eye + face + mic + head + GRU model |
| `rules_only` | Eye + face + mic + head |
| `mic_head` | Mic + head only |
| `head_only` | Head motion only |
| `noise_only` | Procedural anti-freeze noise |

**Input Sources**
- VRCFT eye/face tracking (14 eye + 13 face features, WebSocket push)
- Microphone prosody via librosa: pitch, energy, speech rate, speaking detection
- Head motion via OpenXR: pitch/roll/yaw, translation, velocity, acceleration — self-calibrating from first 2.5s of data
- Optional SpeechBrain emotion context (SER)

**Server**
- Async WebSocket server on port 7720 (`ws_server/`)
- 90fps inference clock on a background thread
- Control messages: `ping`, `reset`, `recalibrate_head`, `set_mode`, `get_status`
- ONNX model is self-contained with embedded normalization stats

**TUI**
- Textual-based terminal UI with live per-AU bar meters, FPS counter, source status indicators, and color-coded mode display
- Keys: `q` quit, `r` recalibrate head, `Ctrl+L` dev log

**VRCFT Plugin** (`BrowSyncModule/`)
- C# net7.0 plugin; drop-in install to `%APPDATA%\VRCFaceTracking\CustomLibs\`
- Persistent WebSocket connection with auto-reconnect (3s backoff) and 5s ping keepalive
- Writes only brow shapes — leaves eye and lower-face tracking to your existing VRCFT modules
- Zeroes brow shapes on disconnect; sends `reset` on reconnect to clear the GRU buffer

**Training**
- Supervised training from labelled `.jsonl` session files with unlabelled pseudo-label support (0.25 weight)
- Custom loss: MSE + temporal smoothness penalty + asymmetric raise/lower weighting
- Exports to ONNX with embedded normalization for zero-config deployment

---

### Known Limitations

- Head motion tracking requires an OpenXR runtime to be active
- VRCFT data expires after 0.5s — a slow tracker will cause fallback mode switching
- The GRU residual scale (0.4×) and sequence length (30 frames) are fixed; changing either requires retraining
- No GUI installer — manual setup required (see README)

---

### Getting Started

```bash
pip install -r requirements.txt
python -m ws_server.server --model models/browsync.onnx
```

Then build and install the VRCFT plugin:
```bash
cd BrowSyncModule && dotnet build -c Release
```
