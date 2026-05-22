"""
BrowSync — Head motion input layer (OpenXR).

Polls HMD pose from the active OpenXR runtime at ~90fps using pyopenxr.
Works with SteamVR, Oculus, WMR, Pico, and any other OpenXR-compliant runtime.

OpenXR gives us a quaternion + position directly — cleaner than OpenVR's
3x4 matrix decomposition. We convert quaternion → Euler (pitch/roll/yaw),
compute velocity and acceleration via rolling differentiation, normalise
everything to [-1, 1], and produce a HeadMotionFrame ready for inference.

Calibration: neutral pose is averaged over the first CALIBRATION_SECONDS
after successful init. All subsequent values are offsets from that neutral.

Install:  pip install pyopenxr
Docs:     https://github.com/cmbruns/pyopenxr
"""

import math
import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("browsync.headmotion")

# ---------------------------------------------------------------------------
# Normalisation ranges
# ---------------------------------------------------------------------------

PITCH_RANGE_DEG   = 45.0
ROLL_RANGE_DEG    = 30.0
YAW_RANGE_DEG     = 60.0
TRANS_Y_RANGE_M   = 0.30
TRANS_Z_RANGE_M   = 0.30
ROT_VEL_RANGE     = 180.0
TRANS_VEL_RANGE   = 0.50
ROT_ACCEL_RANGE   = 600.0
TRANS_ACCEL_RANGE = 2.0

CALIBRATION_SECONDS    = 2.5
CALIBRATION_MIN_FRAMES = 30
DERIVATIVE_WINDOW      = 5


# ---------------------------------------------------------------------------
# HeadMotionFrame — matches HEAD_MOTION_FEATURES order in schema.py
# ---------------------------------------------------------------------------

@dataclass
class HeadMotionFrame:
    pitch:        float = 0.0
    roll:         float = 0.0
    yaw:          float = 0.0
    head_y:       float = 0.0
    head_z:       float = 0.0
    pitch_vel:    float = 0.0
    roll_vel:     float = 0.0
    yaw_vel:      float = 0.0
    head_y_vel:   float = 0.0
    pitch_accel:  float = 0.0
    head_y_accel: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.pitch, self.roll, self.yaw,
            self.head_y, self.head_z,
            self.pitch_vel, self.roll_vel, self.yaw_vel, self.head_y_vel,
            self.pitch_accel, self.head_y_accel,
        ], dtype=np.float32)

    @staticmethod
    def zero() -> "HeadMotionFrame":
        return HeadMotionFrame()


# ---------------------------------------------------------------------------
# Quaternion → Euler (YXZ order, degrees)
# ---------------------------------------------------------------------------

def quat_to_euler_deg(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """
    Convert a unit quaternion to Euler angles in degrees (YXZ / pitch-yaw-roll order).

    OpenXR coordinate system: +X right, +Y up, -Z forward (right-handed).

    Returns (pitch_deg, roll_deg, yaw_deg):
      pitch = rotation around X (nod up/down)
      roll  = rotation around Z (tilt left/right)
      yaw   = rotation around Y (turn left/right)
    """
    # Pitch (X axis) — arcsin of the cross term
    sin_pitch = 2.0 * (w * x - y * z)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch_rad = math.asin(sin_pitch)

    cos_pitch = math.cos(pitch_rad)

    if abs(cos_pitch) > 1e-6:
        # Roll (Z axis)
        sin_roll = 2.0 * (w * z + x * y)
        cos_roll = 1.0 - 2.0 * (x * x + z * z)
        roll_rad = math.atan2(sin_roll, cos_roll)

        # Yaw (Y axis)
        sin_yaw = 2.0 * (w * y + x * z)
        cos_yaw = 1.0 - 2.0 * (x * x + y * y)
        yaw_rad = math.atan2(sin_yaw, cos_yaw)
    else:
        # Gimbal lock at ±90° pitch
        roll_rad = math.atan2(2.0 * (w * z - x * y),
                               1.0 - 2.0 * (y * y + z * z))
        yaw_rad = 0.0

    return math.degrees(pitch_rad), math.degrees(roll_rad), math.degrees(yaw_rad)


def normalise(value: float, range_: float) -> float:
    return max(-1.0, min(1.0, value / range_))


# ---------------------------------------------------------------------------
# Rolling derivative
# ---------------------------------------------------------------------------

class RollingDerivative:
    def __init__(self, window: int = DERIVATIVE_WINDOW):
        self._vals:  deque = deque(maxlen=window)
        self._times: deque = deque(maxlen=window)

    def push(self, value: float, t: float) -> tuple[float, float]:
        self._vals.append(value)
        self._times.append(t)
        n = len(self._vals)
        if n < 3:
            return 0.0, 0.0
        dt = self._times[-1] - self._times[0]
        if dt < 1e-6:
            return 0.0, 0.0
        v = list(self._vals)
        t_list = list(self._times)
        vel = (v[-1] - v[0]) / dt
        mid = n // 2
        dt1 = t_list[mid] - t_list[0]
        dt2 = t_list[-1] - t_list[mid]
        if dt1 < 1e-6 or dt2 < 1e-6:
            return vel, 0.0
        vel1 = (v[mid] - v[0]) / dt1
        vel2 = (v[-1] - v[mid]) / dt2
        accel = (vel2 - vel1) / ((dt1 + dt2) * 0.5)
        return vel, accel

    def reset(self):
        self._vals.clear()
        self._times.clear()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationState:
    samples:        list  = field(default_factory=list)
    complete:       bool  = False
    neutral_pitch:  float = 0.0
    neutral_roll:   float = 0.0
    neutral_yaw:    float = 0.0
    neutral_ty:     float = 0.0
    neutral_tz:     float = 0.0

    def add(self, pitch, roll, yaw, ty, tz):
        self.samples.append((pitch, roll, yaw, ty, tz))

    def finalise(self) -> bool:
        if len(self.samples) < CALIBRATION_MIN_FRAMES:
            return False
        arr = np.array(self.samples)
        self.neutral_pitch = float(np.median(arr[:, 0]))
        self.neutral_roll  = float(np.median(arr[:, 1]))
        self.neutral_yaw   = float(np.median(arr[:, 2]))
        self.neutral_ty    = float(np.median(arr[:, 3]))
        self.neutral_tz    = float(np.median(arr[:, 4]))
        self.complete = True
        log.info(
            f"[HeadMotion] Calibration complete — "
            f"pitch={self.neutral_pitch:.1f}° roll={self.neutral_roll:.1f}° "
            f"yaw={self.neutral_yaw:.1f}°"
        )
        return True


# ---------------------------------------------------------------------------
# HeadMotionTracker — OpenXR backend
# ---------------------------------------------------------------------------

class HeadMotionTracker:
    """
    Polls the OpenXR HMD pose and exposes HeadMotionFrame via .latest.
    Falls back to zeros gracefully if OpenXR is unavailable.
    """

    def __init__(self, target_fps: float = 90.0):
        self._target_dt   = 1.0 / target_fps
        self._latest      = HeadMotionFrame.zero()
        self._lock        = threading.Lock()
        self._stop        = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available   = False

        self._d_pitch = RollingDerivative()
        self._d_roll  = RollingDerivative()
        self._d_yaw   = RollingDerivative()
        self._d_ty    = RollingDerivative()

        self._calib:       CalibrationState = CalibrationState()
        self._calib_start: Optional[float]  = None

    @property
    def latest(self) -> HeadMotionFrame:
        with self._lock:
            return self._latest

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def calibrated(self) -> bool:
        return self._calib.complete

    @property
    def settling(self) -> bool:
        """True while OpenXR is available but calibration is still accumulating samples."""
        return self._available and not self._calib.complete

    @property
    def ready_in_ms(self) -> int:
        """Milliseconds until calibration is expected to lock; 0 when calibrated."""
        if not self.settling or self._calib_start is None:
            return 0
        elapsed = time.monotonic() - self._calib_start
        remaining = CALIBRATION_SECONDS - elapsed
        return max(0, int(remaining * 1000))

    def recalibrate(self):
        self._calib       = CalibrationState()
        self._calib_start = None
        for d in (self._d_pitch, self._d_roll, self._d_yaw, self._d_ty):
            d.reset()
        log.info("[HeadMotion] Recalibration requested.")

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="BrowSync-HeadMotion", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    # -- Background thread ---------------------------------------------------

    def _run(self):
        try:
            import xr
        except ImportError:
            log.warning(
                "[HeadMotion] pyopenxr not installed — head motion disabled. "
                "Install with:  pip install pyopenxr"
            )
            return

        # Try to initialise OpenXR
        try:
            instance, session, space_local, space_view = self._init_openxr(xr)
        except Exception as e:
            log.warning(f"[HeadMotion] OpenXR init failed: {e}. Head motion disabled.")
            return

        self._available = True
        log.info("[HeadMotion] OpenXR connected. Starting HMD pose polling.")

        try:
            self._poll_loop(xr, instance, session, space_local, space_view)
        finally:
            try:
                xr.destroy_space(space_view)
                xr.destroy_space(space_local)
                xr.destroy_session(session)
                xr.destroy_instance(instance)
            except Exception:
                pass
            self._available = False
            log.info("[HeadMotion] OpenXR disconnected.")

    def _init_openxr(self, xr):
        """
        Create a minimal headless OpenXR instance for pose reading only.
        We don't render anything — we just need spatial tracking.

        NOTE: SteamVR and most OpenXR runtimes require graphics context validation
        even for headless pose-only applications. This is a known limitation.
        If OpenXR fails to initialize, head motion tracking is disabled gracefully.
        """
        try:
            # Create application info with correct API version object
            app_info = xr.ApplicationInfo(
                application_name="BrowSync",
                application_version=1,
                engine_name="BrowSync",
                engine_version=1,
                api_version=xr.Version(1, 0, 0),
            )

            instance_ci = xr.InstanceCreateInfo(application_info=app_info)
            instance = xr.create_instance(instance_ci)

            # Get the system (HMD)
            system_id = xr.get_system(
                instance,
                xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY)
            )

            # Create session. Most runtimes require graphics binding validation.
            # We attempt without binding first, then fail gracefully if needed.
            session_ci = xr.SessionCreateInfo(system_id=system_id)
            session = xr.create_session(instance, session_ci)

            # Begin session
            xr.begin_session(session, xr.SessionBeginInfo(
                primary_view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO
            ))

            # Reference spaces: LOCAL (room-scale origin) and VIEW (HMD pose)
            space_local = xr.create_reference_space(
                session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=xr.ReferenceSpaceType.LOCAL,
                    pose_in_reference_space=xr.Posef(),
                )
            )
            space_view = xr.create_reference_space(
                session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=xr.ReferenceSpaceType.VIEW,
                    pose_in_reference_space=xr.Posef(),
                )
            )

            return instance, session, space_local, space_view

        except Exception as e:
            # If we can't set up OpenXR, provide a helpful diagnostic message
            err_str = str(e).lower()
            if "graphics" in err_str:
                raise RuntimeError(
                    f"OpenXR session creation failed (graphics issue): {e}. "
                    f"This usually means the XR runtime requires a valid graphics context. "
                    f"Try starting your VR application (e.g., SteamVR Dashboard) before running BrowSync. "
                    f"Head motion tracking will be disabled."
                )
            else:
                raise RuntimeError(f"OpenXR session creation failed: {e}")

    def _poll_loop(self, xr, instance, session, space_local, space_view):
        while not self._stop.is_set():
            t_start = time.monotonic()

            try:
                # Process OpenXR events (keeps session alive)
                while True:
                    try:
                        event = xr.poll_event(instance)
                        if isinstance(event, xr.EventDataSessionStateChanged):
                            if event.state in (
                                xr.SessionState.READY,
                                xr.SessionState.SYNCHRONIZED,
                                xr.SessionState.VISIBLE,
                                xr.SessionState.FOCUSED,
                            ):
                                pass  # session is active, continue polling
                    except xr.exception.EventUnavailable:
                        break

                # Query HMD pose: locate VIEW space relative to LOCAL space
                now_xr = int(time.time_ns())   # OpenXR time in nanoseconds
                location = xr.locate_space(space_view, space_local, now_xr)

                pos_valid = bool(
                    location.location_flags &
                    xr.SpaceLocationFlags.POSITION_VALID_BIT
                )
                ori_valid = bool(
                    location.location_flags &
                    xr.SpaceLocationFlags.ORIENTATION_VALID_BIT
                )

                if pos_valid and ori_valid:
                    q = location.pose.orientation
                    p = location.pose.position

                    pitch, roll, yaw = quat_to_euler_deg(q.x, q.y, q.z, q.w)
                    ty = float(p.y)
                    tz = float(p.z)

                    frame = self._process(pitch, roll, yaw, ty, tz, t_start)
                    with self._lock:
                        self._latest = frame
                else:
                    with self._lock:
                        self._latest = HeadMotionFrame.zero()

            except Exception as e:
                log.debug(f"[HeadMotion] Poll error: {e}")
                with self._lock:
                    self._latest = HeadMotionFrame.zero()

            elapsed = time.monotonic() - t_start
            sleep   = self._target_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _process(self, pitch, roll, yaw, ty, tz, t) -> HeadMotionFrame:
        # Calibration phase
        if not self._calib.complete:
            if self._calib_start is None:
                self._calib_start = t
                log.info("[HeadMotion] Calibrating — hold head in neutral position...")
            self._calib.add(pitch, roll, yaw, ty, tz)
            if t - self._calib_start >= CALIBRATION_SECONDS:
                self._calib.finalise()
            # During settling: emit raw normalized frame (neutral = 0, no velocity history)
            return HeadMotionFrame(
                pitch=normalise(pitch, PITCH_RANGE_DEG),
                roll=normalise(roll,   ROLL_RANGE_DEG),
                yaw=normalise(yaw,     YAW_RANGE_DEG),
                head_y=normalise(ty,   TRANS_Y_RANGE_M),
                head_z=normalise(tz,   TRANS_Z_RANGE_M),
            )

        # Offsets from neutral
        dp = pitch - self._calib.neutral_pitch
        dr = roll  - self._calib.neutral_roll
        dy = yaw   - self._calib.neutral_yaw
        dty = ty   - self._calib.neutral_ty
        dtz = tz   - self._calib.neutral_tz

        pv, pa = self._d_pitch.push(dp,  t)
        rv, _  = self._d_roll.push(dr,   t)
        yv, _  = self._d_yaw.push(dy,    t)
        tv, ta = self._d_ty.push(dty,    t)

        return HeadMotionFrame(
            pitch        = normalise(dp,  PITCH_RANGE_DEG),
            roll         = normalise(dr,  ROLL_RANGE_DEG),
            yaw          = normalise(dy,  YAW_RANGE_DEG),
            head_y       = normalise(dty, TRANS_Y_RANGE_M),
            head_z       = normalise(dtz, TRANS_Z_RANGE_M),
            pitch_vel    = normalise(pv,  ROT_VEL_RANGE),
            roll_vel     = normalise(rv,  ROT_VEL_RANGE),
            yaw_vel      = normalise(yv,  ROT_VEL_RANGE),
            head_y_vel   = normalise(tv,  TRANS_VEL_RANGE),
            pitch_accel  = normalise(pa,  ROT_ACCEL_RANGE),
            head_y_accel = normalise(ta,  TRANS_ACCEL_RANGE),
        )