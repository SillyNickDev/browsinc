"""
BrowSync — Data schema and constants.

Defines the input feature vector, output target vector, and all
named indices so the rest of the codebase never uses raw magic numbers.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np

# ---------------------------------------------------------------------------
# Output targets  (VRCFT Unified Expressions — brow AUs only)
# ---------------------------------------------------------------------------

BROW_OUTPUTS = [
    # Unified Expressions uses per-side inner AND outer — 8 outputs total.
    # Key names here MUST match the C# BrowOutputs JSON property names exactly.
    "BrowInnerUpLeft",      # AU1L — left inner corner raises
    "BrowInnerUpRight",     # AU1R — right inner corner raises
    "BrowOuterUpLeft",      # AU2L — left outer corner raises
    "BrowOuterUpRight",     # AU2R — right outer corner raises
    "BrowLowererLeft",      # AU4L — left brow presses down / furrows
    "BrowLowererRight",     # AU4R — right brow presses down / furrows
    "BrowPinchLeft",        # AU4L variant — inner pinch / scrunch
    "BrowPinchRight",       # AU4R variant — inner pinch / scrunch
]

NUM_BROW_OUTPUTS = len(BROW_OUTPUTS)  # 8

# ---------------------------------------------------------------------------
# Input features
# ---------------------------------------------------------------------------

# --- Eye tracking (per eye, then combined) ---
EYE_FEATURES = [
    "EyeOpenessLeft",           # 0-1, lid aperture
    "EyeOpenessRight",
    "EyeWideLeft",              # AU5 — lid retraction beyond neutral open
    "EyeWideRight",
    "EyeSquintLeft",            # AU7 — lid tightening / squint
    "EyeSquintRight",
    "EyeLidTightenerLeft",      # composite lid tension
    "EyeLidTightenerRight",
    "GazeVerticalLeft",         # -1 (down) to 1 (up)
    "GazeVerticalRight",
    "GazeHorizontalLeft",
    "GazeHorizontalRight",
    "BlinkLeft",                # binary blink event
    "BlinkRight",
]

# --- Lower face (VRCFT unified expression subset) ---
FACE_FEATURES = [
    "JawOpen",                  # AU26/27
    "LipCornerPullLeft",        # AU12L — smile
    "LipCornerPullRight",       # AU12R
    "LipCornerDepressorLeft",   # AU15L — frown
    "LipCornerDepressorRight",  # AU15R
    "CheekPuffLeft",            # AU36L
    "CheekPuffRight",
    "CheekRaiserLeft",          # AU6L — Duchenne marker
    "CheekRaiserRight",
    "LipStretchLeft",           # AU20L
    "LipStretchRight",
    "MouthOpen",                # composite mouth aperture
    "TongueOut",
]

# --- Prosody features (derived from microphone, updated ~100ms) ---
PROSODY_FEATURES = [
    "PitchNorm",                # fundamental frequency, normalised 0-1
    "PitchDelta",               # first derivative of pitch (rising/falling)
    "EnergyNorm",               # RMS energy, normalised 0-1
    "EnergyDelta",              # first derivative of energy
    "SpeechRate",               # syllables/sec estimate, normalised
    "IsSpeaking",               # binary voice activity detection
]

# --- Emotion context (SER model output, slow — updated ~500ms) ---
EMOTION_FEATURES = [
    "EmotionValence",           # -1 (negative) to 1 (positive)
    "EmotionArousal",           # 0 (calm) to 1 (excited)
    "EmotionConfidence",        # how confident the SER model is
]

# --- Temporal delta features (computed from rolling window) ---
# Rate-of-change of key signals — helps the model understand motion direction
DELTA_FEATURES = [
    "EyeOpenessDeltaLeft",
    "EyeOpenessDeltaRight",
    "JawOpenDelta",
    "PitchDelta2",              # second derivative of audio pitch (acceleration)
    "EnergyDelta2",
]

# --- Head motion features (from SteamVR HMD pose, polled at ~90fps) ---
# All rotation values normalised over their expected natural range.
# Translation values are in metres relative to calibrated neutral pose.
# Velocity/acceleration computed over a rolling 50ms (~5 frame) window.
HEAD_MOTION_FEATURES = [
    # Raw orientation offsets from calibrated neutral
    "HeadPitch",            # nod up(+)/down(-),    ±45° mapped to [-1, 1]
    "HeadRoll",             # tilt right(+)/left(-), ±30° mapped to [-1, 1]
    "HeadYaw",              # turn right(+)/left(-), ±60° mapped to [-1, 1]

    # Raw translation offsets from calibrated neutral (metres, clamped ±0.3m)
    "HeadY",                # vertical: up(+)/down(-)   — recoil / lean magnitude
    "HeadZ",                # forward(-)/back(+)         — engagement lean / recoil

    # First derivatives — angular velocity (degrees/sec, normalised ±180°/s → [-1,1])
    "HeadPitchVel",         # nodding speed: fast snap = surprise / emphasis
    "HeadRollVel",          # tilting speed: quick tilt = curiosity flash
    "HeadYawVel",           # turning speed: context only
    "HeadYVel",             # vertical recoil speed: sudden drop = aversion

    # Second derivatives — angular acceleration (degrees/sec², normalised)
    # These capture impulsive "snap" movements that trigger brow flashes
    "HeadPitchAccel",       # sudden nod snap → strong brow raise flash
    "HeadYAccel",           # sudden vertical recoil → surprise brow raise
]

ALL_INPUT_FEATURES = (
    EYE_FEATURES
    + FACE_FEATURES
    + PROSODY_FEATURES
    + EMOTION_FEATURES
    + DELTA_FEATURES
    + HEAD_MOTION_FEATURES
)

NUM_INPUTS = len(ALL_INPUT_FEATURES)  # 52

# Build lookup dicts for fast indexing
INPUT_INDEX = {name: i for i, name in enumerate(ALL_INPUT_FEATURES)}
OUTPUT_INDEX = {name: i for i, name in enumerate(BROW_OUTPUTS)}


# ---------------------------------------------------------------------------
# Frame dataclass — one timestep of data
# ---------------------------------------------------------------------------

@dataclass
class BrowFrame:
    """
    A single timestep of paired input/output data.
    Used both for training samples and live inference input.
    """
    timestamp_ms: float
    inputs: np.ndarray          # shape (NUM_INPUTS,)   float32
    targets: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_BROW_OUTPUTS, dtype=np.float32)
    )
    # targets are only populated for labelled (Quest Pro) sessions
    has_labels: bool = False
    session_id: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp_ms": self.timestamp_ms,
            "inputs": self.inputs.tolist(),
            "targets": self.targets.tolist(),
            "has_labels": self.has_labels,
            "session_id": self.session_id,
        }

    @staticmethod
    def from_dict(d: dict) -> "BrowFrame":
        return BrowFrame(
            timestamp_ms=d["timestamp_ms"],
            inputs=np.array(d["inputs"], dtype=np.float32),
            targets=np.array(d["targets"], dtype=np.float32),
            has_labels=d["has_labels"],
            session_id=d.get("session_id", ""),
        )


# ---------------------------------------------------------------------------
# Sequence dataclass — fixed-length window for the recurrent model
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 30   # 30 frames @ ~90fps ≈ 333ms of context

@dataclass
class BrowSequence:
    """
    A sliding window of BrowFrames fed to the GRU model.
    Shape: (SEQUENCE_LENGTH, NUM_INPUTS)
    """
    frames: np.ndarray          # (SEQUENCE_LENGTH, NUM_INPUTS)
    target: np.ndarray          # (NUM_BROW_OUTPUTS,) — label for the LAST frame
    has_labels: bool = False
    session_id: str = ""


# ---------------------------------------------------------------------------
# Normalisation constants (updated after each training run, stored in ONNX metadata)
# ---------------------------------------------------------------------------

@dataclass
class NormStats:
    """Per-feature mean and std for z-score normalisation."""
    mean: np.ndarray    # (NUM_INPUTS,)
    std: np.ndarray     # (NUM_INPUTS,)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "NormStats":
        return NormStats(
            mean=np.array(d["mean"], dtype=np.float32),
            std=np.array(d["std"], dtype=np.float32),
        )

    @staticmethod
    def identity() -> "NormStats":
        return NormStats(
            mean=np.zeros(NUM_INPUTS, dtype=np.float32),
            std=np.ones(NUM_INPUTS, dtype=np.float32),
        )
