# BrowSyncModule — VRCFT Plugin (C#)

The VRCFT integration layer for BrowSync. Connects to the BrowSync Python inference server over WebSocket, receives inferred brow action unit values each frame, and writes them to `UnifiedTracking.Data.Shapes` so VRCFT can forward them to VRChat via OSC.

All inference logic lives in Python. This plugin is intentionally a thin pipe — it does no ML, no signal processing, and no rule evaluation.

---

## How it fits in

```
BrowSync Python server  ──WebSocket──►  BrowSyncModule (VRCFT plugin)  ──OSC──►  VRChat
  (inference at ~90fps)   ws://localhost:7720   (writes 8 brow shapes)
```

The module only claims brow expressions. Eye tracking and lower-face expressions are left entirely to whatever modules you already use (Vive Facial Tracker, ALVR, etc.), so there are no conflicts.

---

## Prerequisites

- [VRCFaceTracking v5+](https://github.com/benaclejames/VRCFaceTracking)
- .NET 7 SDK
- BrowSync Python server (`python -m ws_server.server` from the repo root)

---

## Build

```bash
cd BrowSyncModule
dotnet build -c Release
```

The post-build step copies `BrowSyncModule.dll` to `%APPDATA%\VRCFaceTracking\CustomLibs\` automatically. VRCFT loads everything in that directory on startup — no further installation steps needed.

To install manually instead:

```
BrowSyncModule\bin\Release\net7.0\BrowSyncModule.dll
  → %APPDATA%\VRCFaceTracking\CustomLibs\BrowSyncModule.dll
```

---

## Usage

1. Start the BrowSync Python server:
   ```bash
   python -m ws_server.server --model models/browsync.onnx
   ```
2. Launch VRCFaceTracking — BrowSyncModule appears in the module list automatically.
3. Launch VRChat.

The module connects to `ws://localhost:7720`. If the server isn't up yet, it retries in the background and connects once it's available.

---

## Shapes written

Only these eight `UnifiedExpressions` values are touched. Everything else in VRCFT's shape buffer is left alone.

| UnifiedExpression | Description |
|-------------------|-------------|
| `BrowInnerUpLeft` | Inner brow raise, left side |
| `BrowInnerUpRight` | Inner brow raise, right side |
| `BrowOuterUpLeft` | Outer brow raise, left side |
| `BrowOuterUpRight` | Outer brow raise, right side |
| `BrowLowererLeft` | Brow furrow / lower, left side |
| `BrowLowererRight` | Brow furrow / lower, right side |
| `BrowPinchLeft` | Inner brow scrunch, left side |
| `BrowPinchRight` | Inner brow scrunch, right side |

All values are in `[0, 1]` and are clamped before being written.

---

## Connection behaviour

| Event | Behaviour |
|-------|-----------|
| Module initialises | Connects with 8-second timeout; succeeds or logs a warning and retries |
| Server not yet running | Retries automatically with 3-second backoff — no action needed |
| Server disconnects | All 8 brow shapes zeroed immediately; reconnects in background |
| Reconnected | Sends `reset` to clear the Python server's GRU buffer |
| Idle | Sends `ping` every 5 seconds to keep the connection alive |

---

## Configuration

All connection settings are hardcoded. To change them, edit `BrowSyncModule/BrowSyncModule.cs` and rebuild.

| Setting | Default | Location |
|---------|---------|----------|
| Host | `localhost` | `BrowSyncModule.cs` |
| Port | `7720` | `BrowSyncModule.cs` |
| Init timeout | 8 seconds | `BrowSyncModule.cs` |
| Update rate | ~90fps (11ms sleep) | `BrowSyncModule.cs` |
| Reconnect backoff | 3 seconds | `BrowSyncClient.cs` |
| Ping interval | 5 seconds | `BrowSyncClient.cs` |
