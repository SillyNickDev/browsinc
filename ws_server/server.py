"""
BrowSync — WebSocket inference server (push-mode with fallback).

Output modes (automatic, based on available data sources):
  ml          — eye + face + mic + head + GRU model  (full pipeline)
  rules_only  — eye + face + mic + head, rule base only
  mic_head    — no face/eye tracker; mic + head motion only
  head_only   — no face/eye, no mic; head motion only
  noise_only  — nothing available; procedural noise only

The server runs its own 90fps clock and pushes output continuously.
VRCFT client frames are merged in when available — they don't drive
the clock. This means the preview works even with no face tracker,
and mic/head data is always used regardless of VRCFT status.

WebSocket protocol unchanged — see previous server.py for full docs.
New control messages:
  { "type": "recalibrate_head" }   — restart head motion calibration
  { "type": "get_status" }         — returns current status JSON

Port: 7720
"""

import asyncio
import json
import logging
import time
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import websockets

from data.schema import (
    BrowFrame, NormStats, NUM_INPUTS, INPUT_INDEX, BROW_OUTPUTS,
    HEAD_MOTION_FEATURES, PROSODY_FEATURES, SEQUENCE_LENGTH
)
from inference.rules import RuleBasedEstimator, apply_head_motion_rules
from inference.smoother import BrowSmoother
from inference.head_motion import HeadMotionTracker
from inference.microphone import MicrophoneProcessor, MicrophoneProcessorWithSER
from inference.preview import PreviewStatus


class _NullPreview:
    """No-op preview for --no-tui / headless mode (no Textual dependency)."""
    def start(self) -> None: pass
    def stop(self)  -> None: pass
    def update(self, _status) -> None: pass


def _make_mic_processor() -> MicrophoneProcessor:
    """Use SpeechBrain-enhanced processor if speechbrain is installed."""
    try:
        import speechbrain  # noqa: F401
        log.info("SpeechBrain detected — using emotion-aware mic processor.")
        return MicrophoneProcessorWithSER()
    except ImportError:
        log.info("SpeechBrain not installed — using basic prosody mic processor.")
        return MicrophoneProcessor()
from models.gru_model import BrowSyncGRU, BrowSyncInference, load_from_onnx_metadata

logging.basicConfig(
    level=logging.INFO,
    format="[BrowSync] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("browsync")

DEFAULT_PORT = 7720
DONATION_DIR = Path("data/sessions/donated")
DONATION_DIR.mkdir(parents=True, exist_ok=True)

# How long a VRCFT frame is considered "fresh" before we treat eye/face as absent
VRCFT_STALE_THRESHOLD = 0.5   # seconds


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------

_HEAD_NAMES    = set(HEAD_MOTION_FEATURES)
_PROSODY_NAMES = set(PROSODY_FEATURES)


def assemble_inputs(
    vrcft_inputs: dict,           # latest from VRCFT client (may be empty)
    prosody_dict: dict,           # from MicrophoneProcessor
    head_array: np.ndarray,       # (11,) from HeadMotionTracker
) -> np.ndarray:
    """
    Merge all data sources into a single (NUM_INPUTS,) feature vector.
    VRCFT provides eye/face; prosody comes from mic; head from SteamVR.
    Each source fills only its own feature slots — no conflicts.
    """
    vec = np.zeros(NUM_INPUTS, dtype=np.float32)

    # VRCFT eye + face features
    for name, idx in INPUT_INDEX.items():
        if name not in _HEAD_NAMES and name not in _PROSODY_NAMES:
            if name in vrcft_inputs:
                vec[idx] = float(vrcft_inputs[name])

    # Mic prosody features
    for name, val in prosody_dict.items():
        idx = INPUT_INDEX.get(name)
        if idx is not None:
            vec[idx] = float(val)

    # Head motion features
    for i, name in enumerate(HEAD_MOTION_FEATURES):
        idx = INPUT_INDEX.get(name)
        if idx is not None:
            vec[idx] = head_array[i]

    return vec


def determine_mode(
    has_model: bool,
    eye_face_active: bool,
    mic_active: bool,
    head_active: bool,
    requested_mode: str,
) -> str:
    """
    Determine the effective inference mode based on available sources.
    'requested_mode' is what the user set — actual mode may be downgraded.
    """
    if requested_mode == "noise_only":
        return "noise_only"

    if not eye_face_active and not mic_active and not head_active:
        return "noise_only"

    if not eye_face_active:
        # Fallback: mic and/or head only
        return "mic_head"

    # We have eye/face data
    if has_model and requested_mode == "ml":
        return "ml"

    return "rules_only"


# ---------------------------------------------------------------------------
# Per-client session (now receives pushes, optionally sends VRCFT data)
# ---------------------------------------------------------------------------

class ClientSession:
    def __init__(self):
        self.last_vrcft_inputs: dict = {}
        self.last_vrcft_ts: float = 0.0
        self.requested_mode: str = "ml"
        self.donation_session_id: Optional[str] = None
        self.donation_file = None

    @property
    def vrcft_fresh(self) -> bool:
        return (time.monotonic() - self.last_vrcft_ts) < VRCFT_STALE_THRESHOLD

    def close(self):
        if self.donation_file:
            self.donation_file.close()
            self.donation_file = None


# ---------------------------------------------------------------------------
# Core inference engine (shared across all clients)
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Runs the inference pipeline at 90fps in a background thread.
    Clients subscribe and receive output via asyncio queues.
    """

    TARGET_FPS = 90.0
    TARGET_DT  = 1.0 / TARGET_FPS

    def __init__(
        self,
        onnx_path: Optional[Path],
        head_tracker: HeadMotionTracker,
        mic: MicrophoneProcessor,
        preview: PreviewWindow,
    ):
        self.head_tracker = head_tracker
        self.mic          = mic
        self.preview      = preview

        self.has_model    = False
        self._inference: Optional[BrowSyncInference] = None
        self._load_model(onnx_path)

        self._rule_est = RuleBasedEstimator()
        self._smoother = BrowSmoother()

        # Shared latest output — written by inference thread, read by WS handler
        self._latest_outputs: np.ndarray = np.zeros(len(BROW_OUTPUTS), dtype=np.float32)
        self._latest_mode: str = "noise_only"
        self._lock = threading.Lock()

        # VRCFT input — updated by WS handler, read by inference thread
        self._vrcft_inputs: dict = {}
        self._vrcft_ts: float = 0.0
        self._vrcft_lock = threading.Lock()

        # Subscriber queues (asyncio) — one per connected WS client
        self._subscribers: list[asyncio.Queue] = []
        self._sub_lock = threading.Lock()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # FPS tracking
        self._frame_count: int = 0
        self._fps_window: deque = deque()

    def _load_model(self, onnx_path: Optional[Path]):
        if not onnx_path or not Path(onnx_path).exists():
            log.warning("No ONNX model — rules/fallback only.")
            return
        try:
            import onnxruntime as ort
            norm_stats, _, _ = load_from_onnx_metadata(onnx_path)
            ort_sess = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            engine = self

            class _OrtInf(BrowSyncInference):
                def __init__(self_i):
                    self_i.norm_stats = norm_stats
                    self_i.residual_scale = 0.4
                    self_i._ort = ort_sess
                    self_i._buffer = np.zeros(
                        (SEQUENCE_LENGTH, NUM_INPUTS), dtype=np.float32
                    )
                def push_frame(self_i, raw_inputs, rule_est):
                    normed = self_i.norm_stats.normalise(raw_inputs)
                    self_i._buffer[:-1] = self_i._buffer[1:]
                    self_i._buffer[-1] = normed
                    seq = self_i._buffer[np.newaxis]
                    res = self_i._ort.run(None, {"input_sequence": seq})[0].squeeze(0)
                    return np.clip(rule_est + res * self_i.residual_scale, 0.0, 1.0).astype(np.float32)
                def reset_buffer(self_i):
                    self_i._buffer[:] = 0.0

            self._inference = _OrtInf()
            self.has_model = True
            log.info(f"Model loaded from {onnx_path}")
        except Exception as e:
            log.error(f"Failed to load ONNX model: {e}")

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def update_vrcft(self, inputs: dict):
        with self._vrcft_lock:
            self._vrcft_inputs = inputs
            self._vrcft_ts = time.monotonic()

    def reset_buffer(self):
        if self._inference:
            self._inference.reset_buffer()

    def recalibrate(self, target: str) -> dict:
        """
        Reset one or more subsystems. Returns a dict describing the ack payload.
        target: "head" | "mic" | "gru" | "all"
        If all sources are offline (noise_only), returns status "deferred".
        """
        head_avail = self.head_tracker.is_available
        mic_avail  = self.mic.is_available
        any_avail  = head_avail or mic_avail

        do_head = target in ("head", "all")
        do_mic  = target in ("mic",  "all")
        do_gru  = target in ("gru",  "all")

        if not any_avail and target == "all":
            # No sources online — subsystems will self-calibrate when they reconnect
            return {"target": target, "status": "deferred", "ready_in_ms": None}

        if do_head:
            self.head_tracker.recalibrate()
        if do_mic:
            self.mic.recalibrate()
        if do_gru:
            self.reset_buffer()

        if target == "gru":
            return {"target": "gru", "status": "ready", "ready_in_ms": 0}
        if target == "head":
            return {"target": "head", "status": "settling", "ready_in_ms": 2500}
        if target == "mic":
            return {"target": "mic", "status": "settling",
                    "ready_in_ms": self.mic.ready_in_ms or 10000}
        # all
        return {"target": "all", "status": "settling", "ready_in_ms": 10000}

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="BrowSync-Engine", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    # -- Inference loop (background thread) ----------------------------------

    def _run(self):
        last_t = time.monotonic()

        while not self._stop.is_set():
            now   = time.monotonic()
            dt    = now - last_t
            last_t = now

            outputs, mode = self._step(dt)

            with self._lock:
                self._latest_outputs = outputs
                self._latest_mode    = mode

            # Push to all subscribers via their asyncio queues
            if self._loop and self._loop.is_running():
                msg = self._build_message(outputs, mode)
                with self._sub_lock:
                    for q in list(self._subscribers):
                        try:
                            self._loop.call_soon_threadsafe(
                                q.put_nowait, msg
                            )
                        except Exception:
                            pass

            # Update preview
            self._update_preview(outputs, mode)

            # Frame budget sleep
            elapsed = time.monotonic() - now
            sleep = self.TARGET_DT - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _step(self, dt: float) -> tuple[np.ndarray, str]:
        # Gather sources
        head_frame  = self.head_tracker.latest
        head_active = self.head_tracker.is_available   # true even during settling (raw frames)
        head_array  = head_frame.to_array()

        prosody     = self.mic.latest
        mic_active  = self.mic.is_available and not self.mic.settling

        with self._vrcft_lock:
            vrcft_inputs = dict(self._vrcft_inputs)
            vrcft_fresh  = (time.monotonic() - self._vrcft_ts) < VRCFT_STALE_THRESHOLD

        eye_face_active = vrcft_fresh and bool(vrcft_inputs)

        # Requested mode comes from latest connected client; default ml
        requested = "ml"

        mode = determine_mode(
            self.has_model, eye_face_active, mic_active, head_active, requested
        )

        if mode == "noise_only":
            # Pure procedural noise — at least brows aren't frozen
            dummy = BrowFrame(timestamp_ms=0, inputs=np.zeros(NUM_INPUTS, dtype=np.float32))
            out = self._rule_est.estimate(dummy, dt_seconds=dt)
            return self._smoother.smooth(out, dt), mode

        # Assemble inputs from all available sources
        prosody_dict = prosody.to_feature_dict() if mic_active else {}
        inputs = assemble_inputs(
            vrcft_inputs if eye_face_active else {},
            prosody_dict,
            head_array,
        )

        frame = BrowFrame(timestamp_ms=time.time() * 1000, inputs=inputs)
        rule_out = self._rule_est.estimate(frame, dt_seconds=dt)

        # Apply head motion additive rules if head is active
        if head_active:
            rule_out = apply_head_motion_rules(inputs, rule_out)

        if mode == "mic_head":
            # No face/eye — scale down raises, emphasise mic-driven motion
            # Prosody still drives pitch/energy → brow raise via rule base
            rule_out *= 0.7   # slightly conservative without eye confirmation
            return self._smoother.smooth(rule_out, dt), mode

        if mode == "ml" and self._inference is not None:
            combined = self._inference.push_frame(inputs, rule_out)
            return self._smoother.smooth(combined, dt), mode

        return self._smoother.smooth(rule_out, dt), "rules_only"

    def _build_message(self, outputs: np.ndarray, mode: str) -> str:
        return json.dumps({
            "type": "brow",
            "ts": time.time() * 1000,
            "outputs": {
                name: round(float(outputs[i]), 4)
                for i, name in enumerate(BROW_OUTPUTS)
            },
            "mode": mode,
            "head_tracking": self.head_tracker.is_available,
            "mic_active": self.mic.is_available,
        })

    def _update_preview(self, outputs: np.ndarray, mode: str):
        now = time.monotonic()
        self._fps_window.append(now)
        while self._fps_window and now - self._fps_window[0] > 2.0:
            self._fps_window.popleft()

        status = PreviewStatus(
            mode=mode,
            eye_face_active=(time.monotonic() - self._vrcft_ts) < VRCFT_STALE_THRESHOLD,
            mic_active=self.mic.is_available and not self.mic.settling,
            head_active=self.head_tracker.is_available and self.head_tracker.calibrated,
            calibrating=self.head_tracker.settling,
            mic_settling=self.mic.settling,
            head_ready_in_ms=self.head_tracker.ready_in_ms,
            mic_ready_in_ms=self.mic.ready_in_ms,
            frames_per_sec=len(self._fps_window) / 2.0,
            outputs={name: float(outputs[i]) for i, name in enumerate(BROW_OUTPUTS)},
            model_loaded=self.has_model,
        )
        self.preview.update(status)

    @property
    def latest(self) -> tuple[np.ndarray, str]:
        with self._lock:
            return self._latest_outputs.copy(), self._latest_mode


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

class BrowSyncServer:

    def __init__(self, onnx_path: Optional[Path] = None, port: int = DEFAULT_PORT, preview=None):
        self.port = port

        self._head    = HeadMotionTracker(target_fps=90.0)
        self._mic     = _make_mic_processor()
        self._preview = preview if preview is not None else _NullPreview()
        self._engine  = InferenceEngine(onnx_path, self._head, self._mic, self._preview)

        self._head.start()
        self._mic.start()

        log.info("Head motion tracker started.")
        log.info("Microphone processor started.")

    def recalibrate_head(self) -> None:
        self._engine.recalibrate("head")

    def recalibrate(self, target: str) -> None:
        self._engine.recalibrate(target)

    async def handle_client(self, ws):
        log.info(f"Client connected: {ws.remote_address}")
        session = ClientSession()
        sub_q = self._engine.subscribe()

        # Send initial reset
        self._engine.reset_buffer()

        recv_task = asyncio.create_task(self._recv_loop(ws, session))
        send_task = asyncio.create_task(self._send_loop(ws, sub_q))

        done, pending = await asyncio.wait(
            [recv_task, send_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        self._engine.unsubscribe(sub_q)
        session.close()
        log.info(f"Client disconnected: {ws.remote_address}")

    async def _send_loop(self, ws, q: asyncio.Queue):
        """Push inference output to the client as fast as the engine produces it."""
        try:
            while True:
                msg = await q.get()
                await ws.send(msg)
        except Exception:
            pass

    async def _recv_loop(self, ws, session: ClientSession):
        """Receive VRCFT frames and control messages from the client."""
        try:
            async for message in ws:
                try:
                    msg = json.loads(message)
                    t   = msg.get("type", "frame")

                    if t == "ping":
                        await ws.send(json.dumps({"type": "pong"}))

                    elif t == "reset":
                        self._engine.reset_buffer()
                        self._head.recalibrate()
                        await ws.send(json.dumps({"type": "reset_ack"}))

                    elif t == "recalibrate_head":
                        # Legacy message — kept for VRCFT plugin backward compat
                        self._engine.recalibrate("head")
                        await ws.send(json.dumps({"type": "recalibrate_head_ack"}))

                    elif t == "recalibrate":
                        target = msg.get("target", "all")
                        ack = self._engine.recalibrate(target)
                        await ws.send(json.dumps({"type": "recalibrate_ack", **ack}))

                    elif t == "set_mode":
                        session.requested_mode = msg.get("mode", "ml")
                        await ws.send(json.dumps({"type": "mode_ack", "mode": session.requested_mode}))

                    elif t == "get_status":
                        _, mode = self._engine.latest
                        await ws.send(json.dumps({
                            "type": "status",
                            "mode": mode,
                            "head_tracking": self._head.is_available,
                            "mic_active": self._mic.is_available,
                            "model_loaded": self._engine.has_model,
                        }))

                    elif t in ("frame", "labelled_frame"):
                        inputs = msg.get("inputs", {})
                        self._engine.update_vrcft(inputs)

                        # Donation handling
                        if t == "labelled_frame":
                            await self._handle_donation(msg, session, inputs)

                except (json.JSONDecodeError, Exception) as e:
                    log.debug(f"Recv error: {e}")

        except websockets.exceptions.ConnectionClosed:
            pass

    async def _handle_donation(self, msg: dict, session: ClientSession, inputs: dict):
        gt      = msg.get("brow_ground_truth", {})
        consent = msg.get("consent_version")
        if not (consent and gt):
            return

        if not session.donation_session_id:
            sid = str(uuid.uuid4())
            session.donation_session_id = sid
            session.donation_file = open(DONATION_DIR / f"{sid}.jsonl", "w")

        head_array = self._head.latest.to_array()
        prosody    = self._mic.latest.to_feature_dict()
        full_inputs = assemble_inputs(inputs, prosody, head_array)

        record = {
            "timestamp_ms": msg.get("ts"),
            "inputs": full_inputs.tolist(),
            "targets": [gt.get(name, 0.0) for name in BROW_OUTPUTS],
            "has_labels": True,
            "session_id": session.donation_session_id,
            "consent_version": consent,
        }
        if session.donation_file:
            session.donation_file.write(json.dumps(record) + "\n")
            session.donation_file.flush()

    async def run(self):
        loop = asyncio.get_running_loop()
        self._engine.set_loop(loop)
        self._engine.start()

        log.info(f"BrowSync server starting on ws://localhost:{self.port}")
        async with websockets.serve(self.handle_client, "localhost", self.port):
            log.info(
                f"Listening | Model: {'loaded' if self._engine.has_model else 'not found'} | "
                f"SteamVR: {'active' if self._head.is_available else 'unavailable'} | "
                f"Mic: {'active' if self._mic.is_available else 'unavailable'}"
            )
            await asyncio.Future()

    def shutdown(self):
        self._engine.stop()
        self._head.stop()
        self._mic.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BrowSync inference server")
    parser.add_argument("--model",   type=Path, default=Path("models/browsync.onnx"))
    parser.add_argument("--port",    type=int,  default=DEFAULT_PORT)
    parser.add_argument("--no-tui",  action="store_true",
                        help="Headless mode: disable terminal UI, log to console instead")
    args = parser.parse_args()

    if args.no_tui:
        # Headless mode — plain asyncio loop, logs stay on stdout
        server = BrowSyncServer(onnx_path=args.model, port=args.port)
        try:
            asyncio.run(server.run())
        except KeyboardInterrupt:
            log.info("Shutting down.")
        finally:
            server.shutdown()
    else:
        # TUI mode — Textual is the main loop; server runs as an asyncio worker inside it.
        # Server errors are shown as TUI notifications; all logs also go to browsync.log.
        from TUI.tui import BrowSyncApp, _TUIBridge
        from textual.logging import TextualHandler

        file_handler = logging.FileHandler("browsync.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "[BrowSync] %(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
        ))
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        logging.root.addHandler(TextualHandler())
        logging.root.addHandler(file_handler)

        bridge = _TUIBridge()
        server = BrowSyncServer(onnx_path=args.model, port=args.port, preview=bridge)
        app    = BrowSyncApp(
            server_run_coro=server.run(),
            on_recalibrate=server.recalibrate,
        )
        bridge.set_app(app)
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()


if __name__ == "__main__":
    main()
