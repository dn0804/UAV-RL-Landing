"""
Spawn position sampler for episode resets.

Generates randomized (x, y, z, yaw) spawn tuples within the current
curriculum stage's constraints.  Uses rejection sampling to enforce:
  1. Drone always spawns at or above marker height.
  2. Marker is within the camera's field of view at the spawn pose.

Coordinate conventions (matching SDF):
  - Marker at (-0.40, 0, 1.10), facing +X.
  - Landing pad at (0.05, 0, 0.75).
  - Drone approaches from +X direction.
"""

import math
import numpy as np
from typing import Optional

from rl_uav_package.config.constants import (
    MARKER_CENTER_Z,
    MARKER_WORLD_POS,
    MARKER_HALF_WIDTH,
    CAMERA_HFOV_RAD,
    CAMERA_VFOV_RAD,
    CURRICULUM_STAGES,
    Z_CEILING,
)

# Maximum spawn height — derived from Z_CEILING (operational ceiling).
# Small margin so the drone doesn't spawn right at the ceiling boundary.
Z_SPAWN_MAX = Z_CEILING - 0.05  # 1.90 m

MAX_REJECTION_ATTEMPTS = 200


class Spawner:
    def __init__(self, stage: int = 0, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.set_stage(stage)

    def set_stage(self, stage: int) -> None:
        if stage not in CURRICULUM_STAGES:
            raise ValueError(f"Unknown curriculum stage: {stage}")
        cfg = CURRICULUM_STAGES[stage]
        self._d_min = cfg["d_min"]
        self._d_max = cfg["d_max"]
        self._angle_max = cfg["angle_max"]
        self._stage = stage

    @property
    def stage(self) -> int:
        return self._stage

    _MX = MARKER_WORLD_POS[0]
    _MY = MARKER_WORLD_POS[1]

    def sample(self) -> dict:
        for _ in range(MAX_REJECTION_ATTEMPTS):
            d = self.rng.uniform(self._d_min, self._d_max)
            theta = self.rng.uniform(-self._angle_max, self._angle_max)

            x = self._MX + d * math.cos(theta)
            y = self._MY + d * math.sin(theta)
            z = self.rng.uniform(MARKER_CENTER_Z, Z_SPAWN_MAX)

            if z < MARKER_CENTER_Z:
                continue

            yaw = math.atan2(self._MY - y, self._MX - x)

            if not self._marker_in_fov(x, y, z):
                continue

            return {"x": x, "y": y, "z": z, "yaw": yaw}

        raise RuntimeError(
            f"Spawner failed after {MAX_REJECTION_ATTEMPTS} attempts "
            f"(stage {self._stage}, d=[{self._d_min}, {self._d_max}], "
            f"angle_max={math.degrees(self._angle_max):.1f}°)")

    _TAN_HALF_HFOV = math.tan(CAMERA_HFOV_RAD / 2.0)
    _TAN_HALF_VFOV = math.tan(CAMERA_VFOV_RAD / 2.0)

    def _marker_in_fov(self, drone_x, drone_y, drone_z):
        dx = drone_x - self._MX
        dy = drone_y - self._MY
        d = math.hypot(dx, dy)
        if d < 0.01:
            return False

        hw = MARKER_HALF_WIDTH
        theta = math.atan2(dy, dx)
        min_depth = d - hw * abs(math.sin(theta))
        if min_depth <= 0.0:
            return False

        margin = 0.90

        if hw * abs(math.cos(theta)) > min_depth * self._TAN_HALF_HFOV * margin:
            return False

        z_max = (MARKER_CENTER_Z - hw) + min_depth * self._TAN_HALF_VFOV * margin
        if drone_z > z_max:
            return False

        z_min = (MARKER_CENTER_Z + hw) - min_depth * self._TAN_HALF_VFOV * margin
        if drone_z < z_min:
            return False

        return True