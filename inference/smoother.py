"""
BrowSync — Output smoother.

Spring-damper per AU output with asymmetric attack/decay.
This is what gives brow animation its natural feel — raw ML output
without smoothing looks twitchy and unnatural.

Each brow AU has:
  - attack_hz:  how fast it rises   (brows raise quickly)
  - decay_hz:   how fast it falls   (brows lower slowly)
  - stiffness:  spring strength     (higher = snappier)
  - damping:    oscillation damping (critically damped by default)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from data.schema import NUM_BROW_OUTPUTS, BROW_OUTPUTS, OUTPUT_INDEX


@dataclass
class AUSmootherParams:
    attack_hz: float    # response speed when value is rising
    decay_hz: float     # response speed when value is falling
    stiffness: float    # spring constant
    damping: float      # 1.0 = critically damped, <1 = underdamped (bouncy)


# Per-AU tuned parameters
# Brow raises are fast and slightly springy; lowers are slower and critically damped
DEFAULT_PARAMS: dict[str, AUSmootherParams] = {
    "BrowInnerUp":       AUSmootherParams(attack_hz=8.0,  decay_hz=3.0,  stiffness=80.0, damping=0.9),
    "BrowOuterUpLeft":   AUSmootherParams(attack_hz=7.0,  decay_hz=2.5,  stiffness=70.0, damping=0.85),
    "BrowOuterUpRight":  AUSmootherParams(attack_hz=7.0,  decay_hz=2.5,  stiffness=70.0, damping=0.85),
    "BrowLowererLeft":   AUSmootherParams(attack_hz=5.0,  decay_hz=4.0,  stiffness=60.0, damping=1.0),
    "BrowLowererRight":  AUSmootherParams(attack_hz=5.0,  decay_hz=4.0,  stiffness=60.0, damping=1.0),
    "BrowPinchLeft":     AUSmootherParams(attack_hz=6.0,  decay_hz=3.5,  stiffness=65.0, damping=1.0),
    "BrowPinchRight":    AUSmootherParams(attack_hz=6.0,  decay_hz=3.5,  stiffness=65.0, damping=1.0),
}


class BrowSmoother:
    """
    Per-AU second-order spring-damper smoother.
    Maintains velocity state between frames for physically plausible motion.
    """

    def __init__(self, params: dict[str, AUSmootherParams] = None):
        self.params = params or DEFAULT_PARAMS
        self._positions = np.zeros(NUM_BROW_OUTPUTS, dtype=np.float32)
        self._velocities = np.zeros(NUM_BROW_OUTPUTS, dtype=np.float32)

    def smooth(self, target: np.ndarray, dt: float) -> np.ndarray:
        """
        Step the spring-damper toward target.

        target: (NUM_BROW_OUTPUTS,) desired AU values
        dt: time since last call in seconds
        returns: smoothed AU values
        """
        dt = np.clip(dt, 0.001, 0.05)   # clamp to sane range (20fps min)

        for i, name in enumerate(BROW_OUTPUTS):
            p = self.params.get(name, AUSmootherParams(6.0, 3.0, 60.0, 1.0))

            # Choose attack or decay frequency based on direction
            is_rising = target[i] > self._positions[i]
            freq_hz = p.attack_hz if is_rising else p.decay_hz

            # Spring-damper update (semi-implicit Euler)
            # omega = 2π * freq
            omega = 2.0 * np.pi * freq_hz
            k = p.stiffness
            c = 2.0 * p.damping * np.sqrt(k)   # critical damping coefficient

            displacement = target[i] - self._positions[i]
            spring_force = k * displacement
            damping_force = -c * self._velocities[i]
            acceleration = spring_force + damping_force

            self._velocities[i] += acceleration * dt
            self._positions[i] += self._velocities[i] * dt

        return np.clip(self._positions.copy(), 0.0, 1.0)

    def reset(self):
        self._positions[:] = 0.0
        self._velocities[:] = 0.0

    def reset_to(self, values: np.ndarray):
        """Snap to a specific state without interpolation (e.g. on avatar load)."""
        self._positions[:] = np.clip(values, 0.0, 1.0)
        self._velocities[:] = 0.0
