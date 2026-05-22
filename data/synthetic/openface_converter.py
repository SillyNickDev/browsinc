"""
BrowSync — OpenFace CSV → training session converter.

Takes the CSV output from OpenFace 2.0 (FeatureExtraction or FaceLandmarkVidMulti)
and converts it into BrowSync .jsonl session files suitable for training.

OpenFace outputs one row per frame with columns including:
  - AU01_r, AU02_r, AU04_r  (brow AUs, intensity 0-5)
  - AU05_r, AU06_r, AU07_r  (eye-region AUs)
  - AU12_r, AU15_r, AU20_r, AU25_r, AU26_r  (lower face AUs)
  - eye_lmk_x_*, eye_lmk_y_*  (eye landmarks)
  - gaze_0_x, gaze_0_y, gaze_1_x, gaze_1_y  (gaze direction)
  - pose_Rx, pose_Ry, pose_Rz  (head rotation, radians)
  - confidence, success  (tracking quality)

Usage:
    python data/synthetic/openface_converter.py \\
        --input path/to/openface_output.csv \\
        --output data/sessions/train/ \\
        --subject_id volunteer_01

    # Batch convert a directory of CSVs:
    python data/synthetic/openface_converter.py \\
        --input_dir path/to/openface_outputs/ \\
        --output data/sessions/train/

OpenFace 2.0: https://github.com/TadasBaltrusaitis/OpenFace
Run OpenFace first:  FeatureExtraction.exe -f your_video.mp4 -aus
"""

import argparse
import csv
import json
import math
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from data.schema import (
    BrowFrame, INPUT_INDEX, OUTPUT_INDEX, BROW_OUTPUTS,
    NUM_INPUTS, NUM_BROW_OUTPUTS,
)
from inference.rules import RuleBasedEstimator

# ---------------------------------------------------------------------------
# AU intensity normalisation
# OpenFace AU intensities are 0-5. We normalise to 0-1 for our schema.
# ---------------------------------------------------------------------------

AU_MAX = 5.0

# ---------------------------------------------------------------------------
# Confidence threshold — frames with low OpenFace confidence are discarded
# ---------------------------------------------------------------------------

MIN_CONFIDENCE = 0.85


# ---------------------------------------------------------------------------
# OpenFace column → BrowSync feature mapping
# ---------------------------------------------------------------------------

def build_input_vector(row: dict) -> Optional[np.ndarray]:
    """
    Map one OpenFace CSV row to a BrowSync input feature vector.
    Returns None if the frame should be discarded (low confidence, tracking failure).
    """
    try:
        confidence = float(row.get("confidence", 0))
        success    = int(row.get("success", 0))
    except (ValueError, KeyError):
        return None

    if confidence < MIN_CONFIDENCE or success == 0:
        return None

    vec = np.zeros(NUM_INPUTS, dtype=np.float32)

    def au(col: str, default: float = 0.0) -> float:
        """Read AU intensity column, normalise to [0,1]."""
        try:
            return min(1.0, max(0.0, float(row[col]) / AU_MAX))
        except (KeyError, ValueError):
            return default

    def col(name: str, default: float = 0.0) -> float:
        try:
            return float(row[name])
        except (KeyError, ValueError):
            return default

    def set_feat(name: str, value: float):
        idx = INPUT_INDEX.get(name)
        if idx is not None:
            vec[idx] = float(np.clip(value, -1.0, 1.0))

    # -- Eye openness --
    # OpenFace gives eyelid aperture via AU45 (blink) and AU05 (wide).
    # Approximate openness from the eye landmark vertical span.
    # eye_lmk columns: indices 0-55 for left eye, 56-111 for right (x then y)
    # Vertical span of upper vs lower lid landmarks approximates openness.
    try:
        # Left eye vertical landmarks (approximate — indices 1,5 are top/bottom)
        ly_top = col(" eye_lmk_y_1")
        ly_bot = col(" eye_lmk_y_5")
        lx_span = abs(col(" eye_lmk_x_3") - col(" eye_lmk_x_0")) + 1e-6
        left_openness = min(1.0, max(0.0, abs(ly_bot - ly_top) / lx_span))

        ry_top = col(" eye_lmk_y_57")
        ry_bot = col(" eye_lmk_y_61")
        rx_span = abs(col(" eye_lmk_x_59") - col(" eye_lmk_x_56")) + 1e-6
        right_openness = min(1.0, max(0.0, abs(ry_bot - ry_top) / rx_span))
    except Exception:
        left_openness = right_openness = 0.5  # neutral fallback

    set_feat("EyeOpenessLeft",  left_openness)
    set_feat("EyeOpenessRight", right_openness)
    set_feat("EyeWideLeft",     au(" AU05_r"))
    set_feat("EyeWideRight",    au(" AU05_r"))      # AU05 is bilateral in OpenFace
    set_feat("EyeSquintLeft",   au(" AU07_r"))
    set_feat("EyeSquintRight",  au(" AU07_r"))
    set_feat("EyeLidTightenerLeft",  au(" AU07_r"))
    set_feat("EyeLidTightenerRight", au(" AU07_r"))

    # -- Gaze --
    # OpenFace gaze: gaze_0 = left eye, gaze_1 = right eye
    # Gaze direction vector components, already normalised approximately
    set_feat("GazeVerticalLeft",   col(" gaze_0_y") * -1)   # flip: OpenFace +y is down
    set_feat("GazeVerticalRight",  col(" gaze_1_y") * -1)
    set_feat("GazeHorizontalLeft", col(" gaze_0_x"))
    set_feat("GazeHorizontalRight",col(" gaze_1_x"))

    # Blink approximation from eye openness
    blink_thresh = 0.15
    set_feat("BlinkLeft",  1.0 if left_openness < blink_thresh else 0.0)
    set_feat("BlinkRight", 1.0 if right_openness < blink_thresh else 0.0)

    # -- Lower face --
    set_feat("JawOpen",                  au(" AU26_r"))
    set_feat("LipCornerPullLeft",        au(" AU12_r"))
    set_feat("LipCornerPullRight",       au(" AU12_r"))   # AU12 bilateral
    set_feat("LipCornerDepressorLeft",   au(" AU15_r"))
    set_feat("LipCornerDepressorRight",  au(" AU15_r"))
    set_feat("CheekRaiserLeft",          au(" AU06_r"))
    set_feat("CheekRaiserRight",         au(" AU06_r"))
    set_feat("LipStretchLeft",           au(" AU20_r"))
    set_feat("LipStretchRight",          au(" AU20_r"))
    set_feat("MouthOpen",                au(" AU25_r"))

    # -- Head pose → head motion features --
    # OpenFace pose_Rx/Ry/Rz are rotation in radians (camera-relative)
    # We map these to our HeadPitch/Roll/Yaw features directly.
    # No calibration needed here — OpenFace already gives camera-relative pose,
    # which is approximately neutral-relative for a seated webcam user.
    pitch_rad = col(" pose_Rx")
    roll_rad  = col(" pose_Rz")
    yaw_rad   = col(" pose_Ry")

    pitch_deg = math.degrees(pitch_rad)
    roll_deg  = math.degrees(roll_rad)
    yaw_deg   = math.degrees(yaw_rad)

    set_feat("HeadPitch", pitch_deg / 45.0)
    set_feat("HeadRoll",  roll_deg  / 30.0)
    set_feat("HeadYaw",   yaw_deg   / 60.0)
    # HeadY, HeadZ, velocities, accelerations: not available from static OpenFace output
    # They remain zero — the model will learn to use them when live SteamVR data is present

    # -- Prosody / emotion: not available from video alone --
    # Leave at zero. The model learns these correlations from real sessions.

    return vec


def build_target_vector(row: dict) -> Optional[np.ndarray]:
    """
    Map OpenFace AU outputs to BrowSync target brow values.
    Returns None if targets are unreliable.
    """
    try:
        confidence = float(row.get("confidence", 0))
        if confidence < MIN_CONFIDENCE:
            return None
    except (ValueError, KeyError):
        return None

    targets = np.zeros(NUM_BROW_OUTPUTS, dtype=np.float32)

    def au(col: str) -> float:
        try:
            return min(1.0, max(0.0, float(row[col]) / AU_MAX))
        except (KeyError, ValueError):
            return 0.0

    def set_out(name: str, value: float):
        idx = OUTPUT_INDEX.get(name)
        if idx is not None:
            targets[idx] = float(np.clip(value, 0.0, 1.0))

    # AU01 = inner brow raise (bilateral in OpenFace)
    au01 = au(" AU01_r")
    set_out("BrowInnerUpLeft",  au01)
    set_out("BrowInnerUpRight", au01)

    # AU02 = outer brow raise (bilateral)
    au02 = au(" AU02_r")
    set_out("BrowOuterUpLeft",  au02)
    set_out("BrowOuterUpRight", au02)

    # AU04 = brow lowerer / furrow (bilateral)
    au04 = au(" AU04_r")
    set_out("BrowLowererLeft",  au04 * 0.75)   # lowerer gets 75%
    set_out("BrowLowererRight", au04 * 0.75)
    set_out("BrowPinchLeft",    au04 * 0.45)   # pinch gets 45% (less extreme)
    set_out("BrowPinchRight",   au04 * 0.45)

    return targets


# ---------------------------------------------------------------------------
# Delta feature computation (computed across frames, not per-frame)
# ---------------------------------------------------------------------------

def compute_delta_features(
    frames: list[np.ndarray],
    idx: int,
    window: int = 3,
) -> tuple[float, float]:
    """Compute first and second derivative for a feature at position idx."""
    values = [f[idx] for f in frames]
    n = len(values)
    if n < 3:
        return 0.0, 0.0

    start = max(0, n - window)
    span = values[start:]
    if len(span) < 2:
        return 0.0, 0.0

    vel = span[-1] - span[0]
    if len(span) >= 3:
        vel1 = span[len(span)//2] - span[0]
        vel2 = span[-1] - span[len(span)//2]
        accel = vel2 - vel1
    else:
        accel = 0.0

    return float(vel), float(accel)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def convert_csv(
    csv_path: Path,
    output_dir: Path,
    subject_id: str = "",
    min_frames: int = 100,
) -> Optional[Path]:
    """
    Convert one OpenFace CSV to a BrowSync .jsonl session file.
    Returns the output path on success, None if skipped.
    """
    session_id = f"openface_{subject_id}_{uuid.uuid4().hex[:8]}"
    output_path = output_dir / f"{session_id}.jsonl"

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < min_frames:
        print(f"  [skip] {csv_path.name}: only {len(rows)} frames (min {min_frames})")
        return None

    # Pass 1: build raw vectors
    raw_inputs  = []
    raw_targets = []
    valid_rows  = []

    for row in rows:
        inp = build_input_vector(row)
        tgt = build_target_vector(row)
        if inp is not None and tgt is not None:
            raw_inputs.append(inp)
            raw_targets.append(tgt)
            valid_rows.append(row)

    if len(raw_inputs) < min_frames:
        print(f"  [skip] {csv_path.name}: only {len(raw_inputs)} valid frames after confidence filter")
        return None

    # Pass 2: inject delta features
    eye_open_l_idx = INPUT_INDEX["EyeOpenessLeft"]
    eye_open_r_idx = INPUT_INDEX["EyeOpenessRight"]
    jaw_open_idx   = INPUT_INDEX["JawOpen"]

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(output_path, "w") as out_f:
        for i, (inp, tgt) in enumerate(zip(raw_inputs, raw_targets)):
            history = raw_inputs[max(0, i-5): i+1]

            eye_vel_l, _   = compute_delta_features(history, eye_open_l_idx)
            eye_vel_r, _   = compute_delta_features(history, eye_open_r_idx)
            jaw_vel, _     = compute_delta_features(history, jaw_open_idx)

            inp[INPUT_INDEX["EyeOpenessDeltaLeft"]]  = float(np.clip(eye_vel_l, -1, 1))
            inp[INPUT_INDEX["EyeOpenessDeltaRight"]] = float(np.clip(eye_vel_r, -1, 1))
            inp[INPUT_INDEX["JawOpenDelta"]]         = float(np.clip(jaw_vel,   -1, 1))

            frame = BrowFrame(
                timestamp_ms=i * (1000.0 / 30.0),   # assume 30fps from OpenFace
                inputs=inp,
                targets=tgt,
                has_labels=True,
                session_id=session_id,
            )
            out_f.write(json.dumps(frame.to_dict()) + "\n")
            written += 1

    print(f"  [ok] {csv_path.name} → {output_path.name} ({written} frames)")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenFace 2.0 CSV output to BrowSync training sessions"
    )
    parser.add_argument("--input",      type=Path, help="Single OpenFace CSV file")
    parser.add_argument("--input_dir",  type=Path, help="Directory of OpenFace CSV files")
    parser.add_argument("--output",     type=Path, default=Path("data/sessions/train"),
                        help="Output directory for .jsonl session files")
    parser.add_argument("--subject_id", type=str, default="",
                        help="Subject identifier prefix for session IDs")
    parser.add_argument("--val_split",  type=float, default=0.15,
                        help="Fraction of sessions to put in val set (default 0.15)")
    args = parser.parse_args()

    import random
    output_train = args.output
    output_val   = args.output.parent / "val"

    csv_files = []
    if args.input:
        csv_files = [args.input]
    elif args.input_dir:
        csv_files = sorted(args.input_dir.glob("*.csv"))
    else:
        parser.error("Provide --input or --input_dir")

    print(f"[BrowSync] Converting {len(csv_files)} CSV file(s)...")

    converted = []
    for csv_path in csv_files:
        sid = args.subject_id or csv_path.stem
        result = convert_csv(csv_path, output_train, subject_id=sid)
        if result:
            converted.append(result)

    # Shuffle a val split out of the converted sessions
    if len(converted) > 1 and args.val_split > 0:
        random.shuffle(converted)
        n_val = max(1, int(len(converted) * args.val_split))
        val_files = converted[:n_val]
        output_val.mkdir(parents=True, exist_ok=True)
        for f in val_files:
            dest = output_val / f.name
            f.rename(dest)
        print(f"[BrowSync] Moved {n_val} session(s) to val set → {output_val}")

    print(f"[BrowSync] Done. {len(converted)} sessions converted.")
    print(f"  Train: {output_train}")
    print(f"  Val:   {output_val}")
    print()
    print("Next step: python training/train.py")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    main()
