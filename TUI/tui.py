"""
BrowSync — Textual TUI.

Live inference panel running in the server terminal — no separate window needed.
The Textual app is the main process; the WebSocket server runs as an asyncio worker.

Future expansion (from project TODO):
  - Data tab:    view / organise session files
  - Prepare tab: slice, label, and vectorise sessions for training
  - Train tab:   kick off training runs, view loss curves
  - Eval tab:    evaluate model on val set, compare to rule baseline
  - Deploy tab:  hot-swap the ONNX model without restarting the server

Controls
--------
  q   Quit
  r   Recalibrate head tracker (hold neutral pose)
  Ctrl+L  Toggle Textual developer log panel
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static
from textual.worker import Worker

from inference.preview import PreviewStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BAR_WIDTH = 12

_MODE_STYLE: dict[str, str] = {
    "ml":         "bold green",
    "rules_only": "bold yellow",
    "mic_head":   "bold cyan",
    "head_only":  "bold blue",
    "noise_only": "bold red",
}

# AU pairs: (display label, left key, right key)
_AU_PAIRS = [
    ("InnerUp", "BrowInnerUpLeft",  "BrowInnerUpRight"),
    ("OuterUp", "BrowOuterUpLeft",  "BrowOuterUpRight"),
    ("Lowerer", "BrowLowererLeft",  "BrowLowererRight"),
    ("Pinch",   "BrowPinchLeft",    "BrowPinchRight"),
]


def _bar(value: float) -> str:
    filled = round(max(0.0, min(1.0, value)) * _BAR_WIDTH)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _src(active: bool, label: str) -> str:
    dot = "[green]●[/]" if active else "[red]○[/]"
    return f"{dot} {label}"


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class InferencePanel(Static):
    """Live brow AU display: mode, sources, per-AU bar meters."""

    DEFAULT_CSS = """
    InferencePanel {
        height: 1fr;
        padding: 1 2;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._status: Optional[PreviewStatus] = None

    def set_status(self, status: PreviewStatus) -> None:
        self._status = status
        self.update(self._render())

    def _render(self) -> str:
        s = self._status
        if s is None:
            return "\n  [dim]Waiting for inference engine…[/]"

        # ── header ──────────────────────────────────────────────────────
        mode_style = _MODE_STYLE.get(s.mode, "bold white")
        model_txt  = "[green]loaded[/]" if s.model_loaded else "[dim]no model[/]"
        header = (
            f"\n  Mode: [{mode_style}]{s.mode}[/]"
            f"   FPS: [cyan]{s.frames_per_sec:.1f}[/]"
            f"   Model: {model_txt}"
        )

        # ── sources ─────────────────────────────────────────────────────
        calib = "  [yellow](calibrating…)[/]" if s.calibrating else ""
        sources = (
            f"\n  {_src(s.eye_face_active, 'Eye/Face')}"
            f"   {_src(s.mic_active, 'Mic')}"
            f"   {_src(s.head_active or s.calibrating, 'Head')}{calib}"
        )

        # ── AU bars ─────────────────────────────────────────────────────
        divider = "\n  [dim]" + "─" * 58 + "[/]"
        au_lines = [divider]
        for label, lkey, rkey in _AU_PAIRS:
            lv = s.outputs.get(lkey, 0.0)
            rv = s.outputs.get(rkey, 0.0)
            au_lines.append(
                f"  {label:<8} L  {_bar(lv)}  [cyan]{lv:.2f}[/]"
                f"    {label:<8} R  {_bar(rv)}  [cyan]{rv:.2f}[/]"
            )

        return "\n".join([header, divider, sources, *au_lines, ""])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class BrowSyncApp(App):
    """
    BrowSync Textual application.

    Runs the WebSocket inference server as an asyncio worker and displays
    live brow AU values in the terminal.
    """

    CSS = """
    Screen { background: $surface; }
    InferencePanel { border: round $accent; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "recalibrate", "Recalibrate Head"),
    ]

    TITLE = "BrowSync"
    SUB_TITLE = "Eyebrow inference — ws://localhost:7720"

    def __init__(
        self,
        server_run_coro,
        on_recalibrate: Callable[[], None],
    ) -> None:
        super().__init__()
        self._server_coro    = server_run_coro
        self._on_recalibrate = on_recalibrate
        self._panel: Optional[InferencePanel] = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield InferencePanel()
        yield Footer()

    async def on_mount(self) -> None:
        self._panel = self.query_one(InferencePanel)
        self.run_worker(
            self._server_coro,
            thread=False,
            exclusive=False,
            exit_on_error=False,
        )

    def on_worker_failed(self, event: Worker.Failed) -> None:
        self.notify(
            str(event.error),
            title="Server error — check browsync.log",
            severity="error",
            timeout=30,
        )

    def update_status(self, status: PreviewStatus) -> None:
        if self._panel is not None:
            self._panel.set_status(status)

    def action_recalibrate(self) -> None:
        self._on_recalibrate()
        self.notify("Head tracker recalibrating — hold neutral pose.", title="Recalibrate")


# ---------------------------------------------------------------------------
# Bridge (drop-in replacement for PreviewWindow, used by server.py)
# ---------------------------------------------------------------------------

class _TUIBridge:
    """
    Passes PreviewStatus updates from the inference thread to the Textual app.
    Has the same start/stop/update interface as PreviewWindow so server.py
    doesn't need to know whether a GUI or TUI is in use.

    Updates are throttled to ~30 fps to avoid saturating Textual's event loop.
    """

    _PREVIEW_HZ = 30.0
    _PREVIEW_DT = 1.0 / _PREVIEW_HZ

    def __init__(self) -> None:
        self._app: Optional[BrowSyncApp] = None
        self._last_update: float = 0.0

    def set_app(self, app: BrowSyncApp) -> None:
        self._app = app

    def start(self) -> None:
        pass  # app started externally in main()

    def stop(self) -> None:
        pass  # app stopped externally in main()

    def update(self, status: PreviewStatus) -> None:
        now = time.monotonic()
        if now - self._last_update < self._PREVIEW_DT:
            return
        self._last_update = now
        if self._app is not None:
            try:
                self._app.call_from_thread(self._app.update_status, status)
            except Exception:
                pass

