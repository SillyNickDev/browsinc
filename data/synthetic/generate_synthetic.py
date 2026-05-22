"""
BrowSync — Synthetic session generator.

When you have zero real data and zero OpenFace recordings, this generates
procedurally plausible training sessions from FACS co-occurrence statistics.

This is explicitly a bootstrap tool — it gives the model a reasonable prior
so it doesn't start from random weights when real data arrives. Synthetic
data is never used as a substitute for real labels; it fills the unlabelled
session pool (has_labels=False) and trains the model to beat the rule base
on realistic input distributions.

The generator simulates natural conversation expression patterns by:
  1. Sampling "expression events" from a catalogue of common conversational
     states (neutral, smile, surprise, question, concentration, etc.)
  2. Interpolating between them with realistic transition timing
  3. Adding per-feature noise to simulate sensor imperfection
  4. Computing targets from the rule base (so the GRU starts with a prior
     that at least matches the rule base's co-occurrence model)

Usage:
    python data/synthetic/generate_synthetic.py --sessions 50 --frames 2700
    # 50 sessions × 2700 frames = 50 × 30 seconds at 90fps
"""

import argparse
import json
import math
import random
import uuid
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.schema import (
    BrowFrame, INPUT_INDEX, NUM_INPUTS, NUM_BROW_OUTPUTS,
)
from inference.rules import RuleBasedEstimator


# ---------------------------------------------------------------------------
# Expression state catalogue
# Each state is a dict of {feature_name: target_value} for key features.
# Features not listed default to 0.0.
# These represent archetypes — the generator blends between them.
# ---------------------------------------------------------------------------

EXPRESSION_STATES = {
    "neutral": {
        "EyeOpenessLeft": 0.65, "EyeOpenessRight": 0.65,
    },
    "smile_soft": {
        "EyeOpenessLeft": 0.70, "EyeOpenessRight": 0.70,
        "LipCornerPullLeft": 0.55, "LipCornerPullRight": 0.55,
        "CheekRaiserLeft": 0.35, "CheekRaiserRight": 0.35,
    },
    "smile_big": {
        "EyeOpenessLeft": 0.75, "EyeOpenessRight": 0.75,
        "LipCornerPullLeft": 0.85, "LipCornerPullRight": 0.85,
        "CheekRaiserLeft": 0.70, "CheekRaiserRight": 0.70,
        "MouthOpen": 0.30,
    },
    "surprise": {
        "EyeOpenessLeft": 0.95, "EyeOpenessRight": 0.95,
        "EyeWideLeft": 0.80, "EyeWideRight": 0.80,
        "JawOpen": 0.60, "MouthOpen": 0.55,
        "HeadPitch": 0.10,   # slight head-back
        "HeadY": 0.05,
    },
    "question": {
        "EyeOpenessLeft": 0.72, "EyeOpenessRight": 0.72,
        "HeadRoll": 0.25,         # curious tilt
        "HeadPitch": 0.08,
        "PitchDelta": 0.45,       # rising intonation
        "IsSpeaking": 1.0,
    },
    "concentration": {
        "EyeOpenessLeft": 0.55, "EyeOpenessRight": 0.55,
        "EyeSquintLeft": 0.30, "EyeSquintRight": 0.30,
        "LipCornerDepressorLeft": 0.20, "LipCornerDepressorRight": 0.20,
        "HeadPitch": -0.20,       # slight downward tilt
    },
    "frown": {
        "EyeOpenessLeft": 0.60, "EyeOpenessRight": 0.60,
        "EyeSquintLeft": 0.25, "EyeSquintRight": 0.25,
        "LipCornerDepressorLeft": 0.55, "LipCornerDepressorRight": 0.55,
        "LipStretchLeft": 0.20, "LipStretchRight": 0.20,
        "EmotionValence": -0.6, "EmotionArousal": 0.4, "EmotionConfidence": 0.7,
    },
    "talking_neutral": {
        "EyeOpenessLeft": 0.68, "EyeOpenessRight": 0.68,
        "JawOpen": 0.25, "MouthOpen": 0.30,
        "IsSpeaking": 1.0, "EnergyNorm": 0.55,
        "PitchNorm": 0.50,
    },
    "talking_animated": {
        "EyeOpenessLeft": 0.75, "EyeOpenessRight": 0.75,
        "LipCornerPullLeft": 0.30, "LipCornerPullRight": 0.30,
        "JawOpen": 0.35, "MouthOpen": 0.40,
        "IsSpeaking": 1.0, "EnergyNorm": 0.75, "PitchNorm": 0.60,
        "PitchDelta": 0.25,
        "EmotionArousal": 0.55, "EmotionConfidence": 0.65,
    },
    "listening": {
        "EyeOpenessLeft": 0.67, "EyeOpenessRight": 0.67,
        "HeadRoll": 0.10,
        "IsSpeaking": 0.0,
    },
    "head_tilt_curiosity": {
        "EyeOpenessLeft": 0.70, "EyeOpenessRight": 0.70,
        "HeadRoll": 0.45,
        "HeadPitch": 0.05,
    },
    "nod_agreement": {
        "EyeOpenessLeft": 0.67, "EyeOpenessRight": 0.67,
        "HeadPitch": 0.15,
        "HeadPitchVel": 0.40,
    },
    "blink": {
        "EyeOpenessLeft": 0.05, "EyeOpenessRight": 0.05,
        "BlinkLeft": 1.0, "BlinkRight": 1.0,
    },
}

# Transition weights — how likely each state is to follow in conversation
STATE_TRANSITIONS = {
    "neutral":            {"talking_neutral": 0.3, "smile_soft": 0.15, "listening": 0.2, "head_tilt_curiosity": 0.1, "neutral": 0.15, "concentration": 0.1},
    "talking_neutral":    {"talking_animated": 0.25, "neutral": 0.2, "question": 0.15, "smile_soft": 0.15, "talking_neutral": 0.15, "blink": 0.1},
    "talking_animated":   {"talking_neutral": 0.3, "smile_big": 0.15, "question": 0.15, "surprise": 0.1, "talking_animated": 0.2, "blink": 0.1},
    "smile_soft":         {"smile_big": 0.2, "talking_animated": 0.2, "neutral": 0.3, "listening": 0.15, "smile_soft": 0.15},
    "smile_big":          {"smile_soft": 0.4, "talking_animated": 0.2, "neutral": 0.25, "smile_big": 0.15},
    "surprise":           {"neutral": 0.3, "talking_animated": 0.3, "smile_soft": 0.2, "question": 0.2},
    "question":           {"talking_neutral": 0.3, "listening": 0.25, "neutral": 0.2, "head_tilt_curiosity": 0.15, "question": 0.1},
    "concentration":      {"neutral": 0.35, "frown": 0.20, "talking_neutral": 0.25, "concentration": 0.2},
    "frown":              {"neutral": 0.35, "concentration": 0.25, "talking_neutral": 0.25, "frown": 0.15},
    "listening":          {"neutral": 0.3, "nod_agreement": 0.2, "head_tilt_curiosity": 0.2, "talking_neutral": 0.2, "listening": 0.1},
    "head_tilt_curiosity":{"listening": 0.3, "neutral": 0.3, "question": 0.25, "head_tilt_curiosity": 0.15},
    "nod_agreement":      {"listening": 0.4, "neutral": 0.35, "talking_neutral": 0.25},
    "blink":              {"talking_neutral": 0.3, "neutral": 0.3, "listening": 0.2, "talking_animated": 0.2},
}

# Typical duration range for each state (frames at 90fps)
STATE_DURATIONS = {
    "neutral":            (60,  270),
    "talking_neutral":    (90,  360),
    "talking_animated":   (60,  240),
    "smile_soft":         (45,  180),
    "smile_big":          (30,  120),
    "surprise":           (15,   60),
    "question":           (45,  150),
    "concentration":      (90,  360),
    "frown":              (45,  200),
    "listening":          (90,  360),
    "head_tilt_curiosity":(45,  150),
    "nod_agreement":      (20,   60),
    "blink":              (4,    10),
}


def sample_next_state(current: str) -> str:
    transitions = STATE_TRANSITIONS.get(current, {"neutral": 1.0})
    states = list(transitions.keys())
    weights = list(transitions.values())
    return random.choices(states, weights=weights, k=1)[0]


def state_to_vector(state_name: str, noise_scale: float = 0.03) -> np.ndarray:
    """Convert a named state to a feature vector with optional noise."""
    vec = np.zeros(NUM_INPUTS, dtype=np.float32)
    state = EXPRESSION_STATES.get(state_name, {})
    for feat, val in state.items():
        idx = INPUT_INDEX.get(feat)
        if idx is not None:
            noisy = val + random.gauss(0, noise_scale)
            vec[idx] = float(np.clip(noisy, 0.0, 1.0))
    return vec


def interpolate(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Smooth interpolation using a cubic ease curve."""
    t_smooth = t * t * (3 - 2 * t)   # smoothstep
    return a + (b - a) * t_smooth


def add_micro_variation(vec: np.ndarray, t: float, freq: float = 0.3) -> np.ndarray:
    """Add subtle per-feature Perlin-like variation to prevent static frames."""
    result = vec.copy()
    # Subtle sinusoidal variation on key features
    variation = math.sin(2 * math.pi * freq * t) * 0.015
    for feat in ["EyeOpenessLeft", "EyeOpenessRight", "HeadRoll"]:
        idx = INPUT_INDEX.get(feat)
        if idx is not None:
            result[idx] = float(np.clip(result[idx] + variation, 0.0, 1.0))
    return result


# ---------------------------------------------------------------------------
# Session generator
# ---------------------------------------------------------------------------

def generate_session(
    n_frames: int = 2700,
    fps: float = 90.0,
    session_id: str = "",
    noise_scale: float = 0.03,
) -> list[BrowFrame]:
    """
    Generate one synthetic session of n_frames frames.
    Returns a list of BrowFrame objects with targets set by the rule base.
    """
    rule_est = RuleBasedEstimator()
    dt = 1.0 / fps

    # Initialise state machine
    current_state = "neutral"
    current_vec = state_to_vector(current_state)
    target_state = sample_next_state(current_state)
    target_vec   = state_to_vector(target_state)

    duration_min, duration_max = STATE_DURATIONS[current_state]
    transition_end = random.randint(duration_min, duration_max)
    transition_start = 0

    frames = []
    t_sec = 0.0

    for i in range(n_frames):
        # State machine update
        progress = (i - transition_start) / max(1, transition_end - transition_start)

        if progress >= 1.0:
            # Arrived at target state — pick next
            current_state = target_state
            current_vec   = target_vec.copy()
            target_state  = sample_next_state(current_state)
            target_vec    = state_to_vector(target_state, noise_scale)
            transition_start = i
            dur_min, dur_max = STATE_DURATIONS.get(target_state, (60, 180))
            transition_end   = i + random.randint(dur_min, dur_max)
            progress = 0.0

        # Interpolated feature vector
        interp = interpolate(current_vec, target_vec, progress)
        interp = add_micro_variation(interp, t_sec)

        # Compute delta features
        if frames:
            prev = frames[-1].inputs
            eye_delta_l = float(np.clip(interp[INPUT_INDEX["EyeOpenessLeft"]]  - prev[INPUT_INDEX["EyeOpenessLeft"]],  -1, 1))
            eye_delta_r = float(np.clip(interp[INPUT_INDEX["EyeOpenessRight"]] - prev[INPUT_INDEX["EyeOpenessRight"]], -1, 1))
            jaw_delta   = float(np.clip(interp[INPUT_INDEX["JawOpen"]]         - prev[INPUT_INDEX["JawOpen"]],         -1, 1))
        else:
            eye_delta_l = eye_delta_r = jaw_delta = 0.0

        interp[INPUT_INDEX["EyeOpenessDeltaLeft"]]  = eye_delta_l
        interp[INPUT_INDEX["EyeOpenessDeltaRight"]] = eye_delta_r
        interp[INPUT_INDEX["JawOpenDelta"]]         = jaw_delta

        # Get rule-based target (pseudo-label)
        frame_tmp = BrowFrame(timestamp_ms=t_sec * 1000, inputs=interp)
        targets = rule_est.estimate(frame_tmp, dt_seconds=dt)

        frame = BrowFrame(
            timestamp_ms=t_sec * 1000,
            inputs=interp,
            targets=targets,
            has_labels=False,    # synthetic — not real Quest Pro labels
            session_id=session_id,
        )
        frames.append(frame)
        t_sec += dt

    return frames


def write_session(frames: list[BrowFrame], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for frame in frames:
            f.write(json.dumps(frame.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic BrowSync training sessions"
    )
    parser.add_argument("--sessions",    type=int, default=40,
                        help="Number of synthetic sessions to generate (default 40)")
    parser.add_argument("--frames",      type=int, default=2700,
                        help="Frames per session at 90fps (default 2700 = 30s)")
    parser.add_argument("--output",      type=Path, default=Path("data/sessions/train"),
                        help="Output directory")
    parser.add_argument("--val_split",   type=float, default=0.15,
                        help="Fraction of sessions for val set")
    parser.add_argument("--noise",       type=float, default=0.03,
                        help="Per-feature noise scale (default 0.03)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_train = args.output
    output_val   = args.output.parent / "val"
    n_val = max(1, int(args.sessions * args.val_split))

    print(f"[BrowSync] Generating {args.sessions} synthetic sessions "
          f"({args.frames} frames each = {args.frames/90:.0f}s at 90fps)")
    print(f"  Train: {args.sessions - n_val}  Val: {n_val}")

    for i in range(args.sessions):
        sid = f"synthetic_{uuid.uuid4().hex[:12]}"
        frames = generate_session(
            n_frames=args.frames,
            session_id=sid,
            noise_scale=args.noise,
        )
        dest = output_val if i < n_val else output_train
        path = dest / f"{sid}.jsonl"
        write_session(frames, path)

        if (i + 1) % 10 == 0 or i == args.sessions - 1:
            print(f"  {i+1}/{args.sessions} sessions generated...")

    total_frames = args.sessions * args.frames
    print(f"[BrowSync] Done — {total_frames:,} synthetic frames written.")
    print(f"  Train → {output_train}")
    print(f"  Val   → {output_val}")
    print()
    print("These sessions have has_labels=False — they train the model to beat")
    print("the rule base. Add real OpenFace or Quest Pro sessions for better results.")
    print()
    print("Next step: python training/train.py")


if __name__ == "__main__":
    main()
