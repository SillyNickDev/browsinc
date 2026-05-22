"""
BrowSync — Rule-based baseline estimator.

This runs independently of the ML model and serves two purposes:
  1. Cold-start / fallback: works out of the box on v0.1 before any
     trained model exists.
  2. Residual target: the ML model learns to correct *this output*,
     not raw zeros, which dramatically reduces what it needs to learn.

All mappings are grounded in FACS co-occurrence literature.
Tuning constants are named and grouped so they can be exposed in the UI.
"""

import numpy as np
from dataclasses import dataclass
from data.schema import (
    INPUT_INDEX, OUTPUT_INDEX, NUM_BROW_OUTPUTS,
    BROW_OUTPUTS, BrowFrame
)


# ---------------------------------------------------------------------------
# Tuning constants  (candidate UI sliders)
# ---------------------------------------------------------------------------

@dataclass
class RuleWeights:
    # Eye → brow coupling
    eye_open_to_raise: float        = 0.45   # eye wide open → inner/outer raise
    eye_squint_to_lower: float      = 0.50   # squint → lowerer
    eye_wide_to_raise: float        = 0.60   # AU5 wide → strong raise
    blink_suppress_raise: float     = 0.30   # during blink, suppress raise

    # Face → brow coupling
    smile_to_relax: float           = 0.25   # AU12 smile → brow relaxation
    frown_to_furrow: float          = 0.40   # AU15 depress → furrow/pinch
    cheek_raise_to_relax: float     = 0.20   # AU6 Duchenne → relax
    jaw_open_to_raise: float        = 0.30   # surprise mouth → brow raise
    lip_stretch_to_tense: float     = 0.15   # fear/tension marker

    # Prosody → brow coupling
    pitch_rise_to_raise: float      = 0.35   # rising intonation
    energy_burst_to_raise: float    = 0.30   # emphasis flash
    speaking_baseline_lift: float   = 0.05   # subtle lift while talking

    # Emotion context → brow bias (slow)
    arousal_to_raise: float         = 0.20   # high arousal lifts brows
    negative_valence_to_furrow: float = 0.25 # negative emotion → furrow

    # Procedural noise (anti-freeze)
    noise_amplitude: float          = 0.025
    noise_frequency_hz: float       = 0.5


class RuleBasedEstimator:
    """
    Fast, deterministic brow estimator from raw input features.
    Output is in [0, 1] range matching VRCFT Unified Expressions convention.
    """

    def __init__(self, weights: RuleWeights = None):
        self.w = weights or RuleWeights()
        self._noise_phase = 0.0
        self._time_acc = 0.0

    def estimate(self, frame: BrowFrame, dt_seconds: float = 0.011) -> np.ndarray:
        """
        Given a BrowFrame, return estimated brow AU values.
        dt_seconds: time since last call, used for procedural noise.
        Returns np.ndarray of shape (NUM_BROW_OUTPUTS,), values in [0, 1].
        """
        x = frame.inputs
        w = self.w
        out = np.zeros(NUM_BROW_OUTPUTS, dtype=np.float32)

        # -- Convenience accessors -----------------------------------------
        def f(name: str) -> float:
            return float(x[INPUT_INDEX[name]])

        def o(name: str) -> int:
            return OUTPUT_INDEX[name]

        # -- Eye signals -------------------------------------------------------
        eye_open_avg = (f("EyeOpenessLeft") + f("EyeOpenessRight")) * 0.5
        eye_wide_avg = (f("EyeWideLeft") + f("EyeWideRight")) * 0.5
        eye_squint_avg = (f("EyeSquintLeft") + f("EyeSquintRight")) * 0.5
        eye_squint_l = f("EyeSquintLeft")
        eye_squint_r = f("EyeSquintRight")
        blink_avg = (f("BlinkLeft") + f("BlinkRight")) * 0.5

        gaze_v_avg = (f("GazeVerticalLeft") + f("GazeVerticalRight")) * 0.5

        # Eye openness above neutral (0.6) → raise signal
        open_raise = max(0.0, (eye_open_avg - 0.5) * 2.0) * w.eye_open_to_raise
        wide_raise = eye_wide_avg * w.eye_wide_to_raise
        squint_lower = eye_squint_avg * w.eye_squint_to_lower
        blink_suppress = blink_avg * w.blink_suppress_raise

        # Upward gaze slightly correlates with brow raise
        gaze_raise = max(0.0, gaze_v_avg) * 0.15

        # -- Face signals ------------------------------------------------------
        jaw_open = f("JawOpen")
        lip_corner_l = f("LipCornerPullLeft")
        lip_corner_r = f("LipCornerPullRight")
        lip_depress_l = f("LipCornerDepressorLeft")
        lip_depress_r = f("LipCornerDepressorRight")
        cheek_raise_avg = (f("CheekRaiserLeft") + f("CheekRaiserRight")) * 0.5
        lip_stretch_avg = (f("LipStretchLeft") + f("LipStretchRight")) * 0.5

        smile_avg = (lip_corner_l + lip_corner_r) * 0.5
        frown_avg = (lip_depress_l + lip_depress_r) * 0.5

        jaw_raise = jaw_open * w.jaw_open_to_raise
        smile_relax = smile_avg * w.smile_to_relax          # suppresses raise
        frown_furrow = frown_avg * w.frown_to_furrow
        cheek_relax = cheek_raise_avg * w.cheek_raise_to_relax
        stretch_tense = lip_stretch_avg * w.lip_stretch_to_tense

        # -- Prosody signals ---------------------------------------------------
        pitch_delta = f("PitchDelta")
        energy_norm = f("EnergyNorm")
        energy_delta = f("EnergyDelta")
        is_speaking = f("IsSpeaking")

        pitch_raise = max(0.0, pitch_delta) * w.pitch_rise_to_raise
        energy_flash = max(0.0, energy_delta) * w.energy_burst_to_raise
        speaking_lift = is_speaking * w.speaking_baseline_lift

        # -- Emotion context ---------------------------------------------------
        valence = f("EmotionValence")      # -1..1
        arousal = f("EmotionArousal")      # 0..1
        confidence = f("EmotionConfidence")

        arousal_raise = arousal * w.arousal_to_raise * confidence
        neg_furrow = max(0.0, -valence) * w.negative_valence_to_furrow * confidence

        # -- Procedural noise (anti-freeze baseline) ---------------------------
        self._time_acc += dt_seconds
        noise_inner = (
            np.sin(2 * np.pi * w.noise_frequency_hz * self._time_acc) * 0.6
            + np.sin(2 * np.pi * w.noise_frequency_hz * 1.3 * self._time_acc) * 0.4
        ) * w.noise_amplitude
        noise_outer = (
            np.sin(2 * np.pi * w.noise_frequency_hz * 0.7 * self._time_acc + 1.1) * 0.7
            + np.sin(2 * np.pi * w.noise_frequency_hz * 1.8 * self._time_acc) * 0.3
        ) * w.noise_amplitude * 0.6

        # -- Compose outputs ---------------------------------------------------

        # BrowInnerUp L/R — driven by eye openness, jaw drop, pitch, arousal
        # Both sides driven by shared signals; slight asymmetry via per-eye squint
        inner_base = (
            open_raise * 0.8
            + wide_raise * 1.0
            + jaw_raise * 0.7
            + pitch_raise * 0.6
            + energy_flash * 0.5
            + speaking_lift
            + arousal_raise * 0.6
            + gaze_raise * 0.4
            - smile_relax * 0.5
            - blink_suppress * 0.4
            + noise_inner
        )
        out[o("BrowInnerUpLeft")]  = np.clip(inner_base - eye_squint_l * 0.1, 0.0, 1.0)
        out[o("BrowInnerUpRight")] = np.clip(inner_base - eye_squint_r * 0.1, 0.0, 1.0)

        # BrowOuterUp L/R — slightly less driven than inner
        outer_base = (
            open_raise * 0.6
            + wide_raise * 0.8
            + jaw_raise * 0.5
            + pitch_raise * 0.4
            + arousal_raise * 0.5
            - smile_relax * 0.3
            + noise_outer
        )
        out[o("BrowOuterUpLeft")] = np.clip(
            outer_base + (max(0.0, gaze_v_avg)) * 0.1, 0.0, 1.0
        )
        out[o("BrowOuterUpRight")] = np.clip(
            outer_base + (max(0.0, gaze_v_avg)) * 0.1, 0.0, 1.0
        )

        # BrowLowerer L/R — squint, frown, negative emotion
        lower_base = (
            squint_lower * 0.9
            + frown_furrow * 0.8
            + neg_furrow * 0.7
            + stretch_tense * 0.4
            - open_raise * 0.3   # opposes raise
            - wide_raise * 0.4
        )
        out[o("BrowLowererLeft")] = np.clip(
            lower_base + eye_squint_l * 0.1, 0.0, 1.0
        )
        out[o("BrowLowererRight")] = np.clip(
            lower_base + eye_squint_r * 0.1, 0.0, 1.0
        )

        # BrowPinch L/R — inner scrunch, driven by frown + negative valence
        pinch_base = (
            frown_furrow * 0.6
            + neg_furrow * 0.8
            + stretch_tense * 0.3
            - inner_base * 0.4  # opposes inner raise
        )
        out[o("BrowPinchLeft")] = np.clip(pinch_base, 0.0, 1.0)
        out[o("BrowPinchRight")] = np.clip(pinch_base, 0.0, 1.0)

        return out

# ---------------------------------------------------------------------------
# Head-motion-aware rule weights extension
# ---------------------------------------------------------------------------
# These weights are separate so they can be tuned or disabled independently
# from the face/prosody rules, and to keep the RuleWeights dataclass clean.

HEAD_MOTION_RULE_WEIGHTS = {
    # Roll (head tilt) — highest confidence head motion signal for brows
    # Head tilt almost universally accompanies curiosity / empathy → raise
    "roll_to_inner_raise":   0.50,
    "roll_to_outer_raise":   0.35,

    # Pitch up (looking up) → outer brow raise; pitch down → subtle lower
    "pitch_up_to_raise":     0.40,
    "pitch_down_to_lower":   0.25,

    # Vertical recoil (sudden backward head movement) → surprise raise
    "head_y_recoil_to_raise": 0.35,

    # Forward lean (engagement) → subtle inner raise
    "head_z_lean_to_raise":  0.15,

    # Velocity signals — fast motion overrides slow position signal
    "pitch_vel_raise":       0.40,   # upward snap → flash raise
    "roll_vel_raise":        0.30,   # quick tilt → curiosity flash
    "head_y_vel_recoil":     0.35,   # fast recoil pop → surprise raise

    # Acceleration signals — impulsive "snap" events
    "pitch_accel_flash":     0.45,   # sharp nod snap → brow flash
    "head_y_accel_flash":    0.40,   # sharp recoil → surprise flash
}


def apply_head_motion_rules(
    x: np.ndarray,
    out: np.ndarray,
    w: dict = None,
) -> np.ndarray:
    """
    Apply head motion rules as ADDITIVE contributions on top of the existing
    rule estimate. Takes and returns a copy — does not mutate in place.

    x:   full input feature vector (NUM_INPUTS,)
    out: current rule estimate (NUM_BROW_OUTPUTS,) before head motion
    w:   weight dict, defaults to HEAD_MOTION_RULE_WEIGHTS
    """
    if w is None:
        w = HEAD_MOTION_RULE_WEIGHTS

    result = out.copy()

    def f(name: str) -> float:
        from data.schema import INPUT_INDEX
        idx = INPUT_INDEX.get(name)
        return float(x[idx]) if idx is not None else 0.0

    def o(name: str) -> int:
        from data.schema import OUTPUT_INDEX
        return OUTPUT_INDEX[name]

    # -- Read head motion features ------------------------------------------
    pitch      = f("HeadPitch")        # normalised ±1, up = positive
    roll       = f("HeadRoll")         # normalised ±1, right tilt = positive
    head_y     = f("HeadY")            # vertical offset, up = positive
    head_z     = f("HeadZ")            # depth offset, back = positive (recoil)
    pitch_vel  = f("HeadPitchVel")     # upward velocity = positive
    roll_vel   = f("HeadRollVel")
    head_y_vel = f("HeadYVel")
    pitch_accel  = f("HeadPitchAccel")
    head_y_accel = f("HeadYAccel")

    # -- Head roll → bilateral inner + outer raise ---------------------------
    # Curiosity / empathy tilt — abs() because both directions are expressive
    roll_mag = abs(roll)
    roll_inner_add = roll_mag * w["roll_to_inner_raise"]
    roll_outer_add = roll_mag * w["roll_to_outer_raise"]

    # -- Pitch up → outer raise; pitch down → lowerer -----------------------
    pitch_up   = max(0.0, pitch)   # positive only
    pitch_down = max(0.0, -pitch)  # negative only (magnitude)

    pitch_raise_add = pitch_up   * w["pitch_up_to_raise"]
    pitch_lower_add = pitch_down * w["pitch_down_to_lower"]

    # -- Vertical recoil (backward head movement) → surprise raise -----------
    # head_z positive = leaning back, head_y_vel negative = dropping down
    recoil = max(0.0, head_z) * w["head_y_recoil_to_raise"]

    # -- Engagement lean (forward) → subtle inner raise ----------------------
    engagement = max(0.0, -head_z) * w["head_z_lean_to_raise"]

    # -- Velocity contributions (reactive, faster timescale) -----------------
    pitch_vel_add  = max(0.0, pitch_vel)    * w["pitch_vel_raise"]
    roll_vel_add   = abs(roll_vel)          * w["roll_vel_raise"]
    recoil_vel_add = max(0.0, -head_y_vel)  * w["head_y_vel_recoil"]

    # -- Acceleration flash (impulsive snaps, very short-lived) --------------
    # Use max(0, signal) — we only want the positive (upward) snap direction
    pitch_flash   = max(0.0, pitch_accel)   * w["pitch_accel_flash"]
    recoil_flash  = max(0.0, -head_y_accel) * w["head_y_accel_flash"]

    # -- Compose additive contributions per output ---------------------------
    # Inner brow — most responsive to all head signals
    inner_add = (
        roll_inner_add
        + pitch_raise_add * 0.8
        + recoil * 0.8
        + engagement
        + pitch_vel_add * 0.7
        + roll_vel_add * 0.6
        + recoil_vel_add * 0.8
        + pitch_flash
        + recoil_flash
    )

    # Outer brow — driven by roll and pitch, less by recoil
    outer_add = (
        roll_outer_add
        + pitch_raise_add * 0.6
        + recoil * 0.5
        + pitch_vel_add * 0.5
        + roll_vel_add * 0.5
        + pitch_flash * 0.7
        + recoil_flash * 0.5
    )

    # Lowerer — head pitched down → concentration furrow
    lower_add = pitch_lower_add

    result[o("BrowInnerUpLeft")]  = np.clip(result[o("BrowInnerUpLeft")]  + inner_add, 0.0, 1.0)
    result[o("BrowInnerUpRight")] = np.clip(result[o("BrowInnerUpRight")] + inner_add, 0.0, 1.0)
    result[o("BrowOuterUpLeft")]  = np.clip(result[o("BrowOuterUpLeft")]  + outer_add, 0.0, 1.0)
    result[o("BrowOuterUpRight")] = np.clip(result[o("BrowOuterUpRight")] + outer_add, 0.0, 1.0)
    result[o("BrowLowererLeft")]  = np.clip(result[o("BrowLowererLeft")]  + lower_add, 0.0, 1.0)
    result[o("BrowLowererRight")] = np.clip(result[o("BrowLowererRight")] + lower_add, 0.0, 1.0)

    return result


# ----------------------- Nick's Perspective -------------------------
#
#                      ------------------------------------
#                     |                                   |
#                     | This project is so fun            |
#                     |                            /s     |
#            ,        | ----------------------------------
#            |`-.__   |/
#            / ' _/
#           ****` 
#          /    }
#         /  \ /
#     \ /`   \\\
#      `\    /_\\
#       `~~~~~``~`
# --------------------------------------------------------------------