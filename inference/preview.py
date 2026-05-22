"""
BrowSync — Real-time preview window.

Shows a live bar graph of all 8 brow AU outputs, current mode,
data source status (eye/face/mic/head), and a simple face diagram.

Runs in its own thread. Call .update(outputs, status) from the server
and the UI redraws at ~30fps independently.

Uses only tkinter (stdlib) — no extra dependencies.
"""

import threading
import time
import queue
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("browsync.preview")

# ---------------------------------------------------------------------------
# Status dataclass passed from server to UI
# ---------------------------------------------------------------------------

@dataclass
class PreviewStatus:
    mode: str = "rules_only"
    eye_face_active: bool = False
    mic_active: bool = False
    head_active: bool = False
    calibrating: bool = False
    frames_per_sec: float = 0.0
    outputs: dict = field(default_factory=dict)   # {AU_name: float}


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

BG         = "#1a1a2e"
PANEL_BG   = "#16213e"
ACCENT     = "#e94560"
GREEN      = "#0f9b58"
AMBER      = "#f5a623"
TEXT_DIM   = "#8899aa"
TEXT_MAIN  = "#e0e8f0"
BAR_FILL   = "#4fc3f7"
BAR_EMPTY  = "#1e3a4a"

AU_LABELS = [
    ("BrowInnerUpLeft",  "Inner ↑ L"),
    ("BrowInnerUpRight", "Inner ↑ R"),
    ("BrowOuterUpLeft",  "Outer ↑ L"),
    ("BrowOuterUpRight", "Outer ↑ R"),
    ("BrowLowererLeft",  "Lower  L"),
    ("BrowLowererRight", "Lower  R"),
    ("BrowPinchLeft",    "Pinch  L"),
    ("BrowPinchRight",   "Pinch  R"),
]

SOURCE_LABELS = [
    ("eye_face_active", "Eye/Face"),
    ("mic_active",      "Mic"),
    ("head_active",     "Head (6DoF)"),
]


# ---------------------------------------------------------------------------
# Preview window
# ---------------------------------------------------------------------------

class PreviewWindow:
    """
    Spawns a tkinter window in a daemon thread.
    Thread-safe update via a queue.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=5)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="BrowSync-Preview", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._running = False

    def update(self, status: PreviewStatus):
        """Non-blocking update — drops frames if the queue is full."""
        try:
            self._queue.put_nowait(status)
        except queue.Full:
            pass

    # -- UI thread -----------------------------------------------------------

    def _run(self):
        try:
            import tkinter as tk
            from tkinter import font as tkfont
        except ImportError:
            log.warning("[Preview] tkinter not available — preview window disabled.")
            return

        root = tk.Tk()
        root.title("BrowSync — Live Preview")
        root.configure(bg=BG)
        root.resizable(False, False)

        W, H = 480, 540
        root.geometry(f"{W}x{H}")

        # -- Fonts -----------------------------------------------------------
        try:
            mono = tkfont.Font(family="Consolas", size=10)
            mono_sm = tkfont.Font(family="Consolas", size=9)
            title_font = tkfont.Font(family="Consolas", size=13, weight="bold")
        except Exception:
            mono = mono_sm = title_font = None

        canvas = tk.Canvas(root, width=W, height=H, bg=BG, highlightthickness=0)
        canvas.pack()

        # -- Static layout ---------------------------------------------------
        # Title
        canvas.create_text(W//2, 22, text="BrowSync  Live Preview",
                           fill=TEXT_MAIN, font=title_font, anchor="center")
        canvas.create_line(20, 38, W-20, 38, fill=ACCENT, width=1)

        # AU bars — left column (L) and right column (R) in pairs
        BAR_X      = 28
        BAR_W      = W - 56
        BAR_H      = 18
        BAR_SPACING = 30
        BAR_Y_START = 54

        bar_rects = {}   # AU name → canvas rect id
        bar_texts = {}   # AU name → value text id

        for i, (au_name, label) in enumerate(AU_LABELS):
            y = BAR_Y_START + i * BAR_SPACING

            # Label
            canvas.create_text(BAR_X, y + BAR_H//2, text=label,
                                fill=TEXT_DIM, font=mono_sm, anchor="w")

            # Background track
            canvas.create_rectangle(
                BAR_X + 80, y, BAR_X + 80 + BAR_W - 100, y + BAR_H,
                fill=BAR_EMPTY, outline="", tags=f"track_{au_name}"
            )

            # Fill bar (starts at 0 width)
            rect_id = canvas.create_rectangle(
                BAR_X + 80, y, BAR_X + 80, y + BAR_H,
                fill=BAR_FILL, outline="", tags=f"bar_{au_name}"
            )
            bar_rects[au_name] = rect_id

            # Value text
            txt_id = canvas.create_text(
                BAR_X + 80 + BAR_W - 95, y + BAR_H//2,
                text="0.000", fill=TEXT_MAIN, font=mono_sm, anchor="e"
            )
            bar_texts[au_name] = txt_id

        # Separator
        sep_y = BAR_Y_START + len(AU_LABELS) * BAR_SPACING + 4
        canvas.create_line(20, sep_y, W-20, sep_y, fill=PANEL_BG, width=1)

        # Status section
        STATUS_Y = sep_y + 16

        # Mode label
        mode_id = canvas.create_text(
            BAR_X, STATUS_Y, text="Mode: —",
            fill=AMBER, font=mono, anchor="w"
        )

        # FPS
        fps_id = canvas.create_text(
            W - BAR_X, STATUS_Y, text="0 fps",
            fill=TEXT_DIM, font=mono_sm, anchor="e"
        )

        # Source indicators
        source_ids = {}
        for si, (key, label) in enumerate(SOURCE_LABELS):
            x = BAR_X + si * 150
            dot_id = canvas.create_oval(
                x, STATUS_Y + 20, x + 10, STATUS_Y + 30,
                fill=BAR_EMPTY, outline=""
            )
            txt_id = canvas.create_text(
                x + 14, STATUS_Y + 25, text=label,
                fill=TEXT_DIM, font=mono_sm, anchor="w"
            )
            source_ids[key] = (dot_id, txt_id)

        # Calibration notice
        calib_id = canvas.create_text(
            W//2, STATUS_Y + 50, text="",
            fill=AMBER, font=mono, anchor="center"
        )

        # Simple brow face diagram
        FACE_Y = STATUS_Y + 75
        FACE_CX = W // 2
        FACE_R  = 52

        canvas.create_oval(
            FACE_CX - FACE_R, FACE_Y,
            FACE_CX + FACE_R, FACE_Y + FACE_R * 2,
            outline=TEXT_DIM, fill=PANEL_BG, width=1
        )

        # Eyes (static circles)
        for ex in [FACE_CX - 18, FACE_CX + 18]:
            canvas.create_oval(ex-7, FACE_Y+38, ex+7, FACE_Y+52,
                               outline=TEXT_DIM, fill=BG, width=1)

        # Brow lines — these move
        brow_lines = {}
        for side, ex in [("Left", FACE_CX - 18), ("Right", FACE_CX + 18)]:
            ln = canvas.create_line(
                ex - 12, FACE_Y + 30,
                ex + 12, FACE_Y + 30,
                fill=TEXT_MAIN, width=2
            )
            brow_lines[side] = ln

        # Mouth arc (static)
        canvas.create_arc(
            FACE_CX - 20, FACE_Y + 68,
            FACE_CX + 20, FACE_Y + 88,
            start=200, extent=140,
            outline=TEXT_DIM, style="arc", width=1
        )

        # -- Animation loop --------------------------------------------------
        last_status = PreviewStatus()
        frame_times: list = []
        MAX_BAR_PX = BAR_W - 100

        def tick():
            nonlocal last_status

            # Drain queue — use latest
            while True:
                try:
                    last_status = self._queue.get_nowait()
                except queue.Empty:
                    break

            s = last_status

            # FPS tracking
            now = time.monotonic()
            frame_times.append(now)
            while frame_times and now - frame_times[0] > 2.0:
                frame_times.pop(0)
            fps = len(frame_times) / 2.0

            # Update bars
            for au_name, _ in AU_LABELS:
                val = s.outputs.get(au_name, 0.0)
                fill_w = int(val * MAX_BAR_PX)
                x0 = BAR_X + 80
                y0_coord = BAR_Y_START + [a for a, _ in AU_LABELS].index(au_name) * BAR_SPACING

                # Colour: green when active, blue otherwise
                colour = GREEN if val > 0.05 else BAR_FILL
                canvas.coords(bar_rects[au_name],
                               x0, y0_coord,
                               x0 + fill_w, y0_coord + BAR_H)
                canvas.itemconfig(bar_rects[au_name], fill=colour)
                canvas.itemconfig(bar_texts[au_name], text=f"{val:.3f}")

            # Mode
            mode_colours = {
                "ml":         GREEN,
                "rules_only": AMBER,
                "noise_only": TEXT_DIM,
                "mic_head":   "#a78bfa",
            }
            canvas.itemconfig(mode_id,
                               text=f"Mode: {s.mode.upper()}",
                               fill=mode_colours.get(s.mode, TEXT_DIM))

            # FPS
            canvas.itemconfig(fps_id, text=f"{fps:.0f} fps")

            # Source dots
            source_vals = {
                "eye_face_active": s.eye_face_active,
                "mic_active":      s.mic_active,
                "head_active":     s.head_active,
            }
            for key, (dot_id, txt_id) in source_ids.items():
                active = source_vals.get(key, False)
                canvas.itemconfig(dot_id, fill=GREEN if active else BAR_EMPTY)
                canvas.itemconfig(txt_id, fill=TEXT_MAIN if active else TEXT_DIM)

            # Calibration
            canvas.itemconfig(calib_id,
                               text="⟳ Calibrating head — hold neutral pose..." if s.calibrating else "")

            # Brow diagram
            outputs = s.outputs
            for side in ["Left", "Right"]:
                ex = FACE_CX - 18 if side == "Left" else FACE_CX + 18
                inner_up = outputs.get(f"BrowInnerUp{side}", 0.0)
                outer_up = outputs.get(f"BrowOuter Up{side}", outputs.get(f"BrowOuterUp{side}", 0.0))
                lower    = outputs.get(f"BrowLowerer{side}", 0.0)

                # Inner end rises with inner_up, outer end with outer_up
                # Lowerer pushes both down
                inner_y = FACE_Y + 30 - int(inner_up * 10) + int(lower * 5)
                outer_y = FACE_Y + 30 - int(outer_up * 8)  + int(lower * 5)

                if side == "Left":
                    canvas.coords(brow_lines[side],
                                   ex - 12, outer_y, ex + 12, inner_y)
                else:
                    canvas.coords(brow_lines[side],
                                   ex - 12, inner_y, ex + 12, outer_y)

            if self._running:
                root.after(33, tick)   # ~30fps

        root.after(100, tick)

        try:
            root.mainloop()
        except Exception:
            pass
