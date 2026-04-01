"""
Spawn position sampler for episode resets.

Generates randomized (x, y, z, yaw) spawn tuples within the current
curriculum stage's constraints.  Uses rejection sampling to enforce:
  1. Drone always spawns at or above marker height.
  2. Marker is within the camera's field of view at the spawn pose.

Coordinate conventions:
  - Wall surface at x = 0, marker at (0, 0, MARKER_Z).
  - +x points away from wall (into room).
  - +y points left when facing the wall.
  - +z points up.
  - Approach angle θ is measured from the wall normal (+x axis).
    θ = 0 means directly in front of the marker.
    θ > 0 means offset in the +y direction (left).
"""

import math
import numpy as np
from typing import Optional

from rl_uav_package.config.constants import (
    MARKER_Z,
    MARKER_HALF_WIDTH,
    CAMERA_HFOV_RAD,
    CAMERA_VFOV_RAD,
    CURRICULUM_STAGES,
    Z_MAX,
)

# Maximum spawn height.  The operational volume ceiling is Z_MAX = 2.5 m,
# but spawning near the ceiling wastes descent time without teaching anything
# useful.  2.0 m gives ample room for high spawns in Stage 3.
Z_SPAWN_MAX = 2.0

# Maximum rejection attempts before giving up.  Low rejection rates are
# expected (most of the spawn cone lies above MARKER_Z), so hitting this
# limit indicates a configuration bug, not bad luck.
MAX_REJECTION_ATTEMPTS = 200


class Spawner:
    """Samples valid spawn positions for a given curriculum stage.

    Parameters
    ----------
    stage : int
        Curriculum stage (1, 2, or 3).  Determines distance range and
        approach angle limits.
    rng : np.random.Generator, optional
        Random number generator.  If None, a new default generator is
        created.  Pass a seeded generator for reproducibility.
    """

    def __init__(self, stage: int = 1, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.set_stage(stage)

    def set_stage(self, stage: int) -> None:
        """Update the spawn distribution to match a curriculum stage."""
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

    def sample(self) -> dict:
        """Sample a valid spawn pose.

        Returns
        -------
        spawn : dict
            Keys: 'x', 'y', 'z', 'yaw' (all float, SI units).
            The position is in world frame, the yaw orients the drone
            toward the marker so it appears near the center of the
            camera frame.

        Raises
        ------
        RuntimeError
            If MAX_REJECTION_ATTEMPTS are exhausted without finding a
            valid spawn.  This should never happen with correctly
            configured stage parameters.
        """
        for _ in range(MAX_REJECTION_ATTEMPTS):
            # 1. Sample horizontal position in polar coordinates
            d = self.rng.uniform(self._d_min, self._d_max)
            theta = self.rng.uniform(-self._angle_max, self._angle_max)

            x = d * math.cos(theta)
            y = d * math.sin(theta)

            # 2. Sample altitude — uniform between marker height and ceiling
            z = self.rng.uniform(MARKER_Z, Z_SPAWN_MAX)

            # 3. Rejection: drone must be at or above marker height
            #    (guaranteed by the sampling range, but guard defensively)
            if z < MARKER_Z:
                continue

            # 4. Compute yaw: orient the drone's nose toward the marker
            yaw = math.atan2(0.0 - y, 0.0 - x)

            # 5. Validate marker is within camera FOV at this pose
            if not self._marker_in_fov(x, y, z):
                continue

            return {"x": x, "y": y, "z": z, "yaw": yaw}

        raise RuntimeError(
            f"Spawner failed after {MAX_REJECTION_ATTEMPTS} attempts "
            f"(stage {self._stage}, d=[{self._d_min}, {self._d_max}], "
            f"angle_max={math.degrees(self._angle_max):.1f}°). "
            f"Check stage configuration."
        )

    # Pre-computed tangent values for FOV checks.
    _TAN_HALF_HFOV = math.tan(CAMERA_HFOV_RAD / 2.0)   # tan(41°) ≈ 0.8693
    _TAN_HALF_VFOV = math.tan(CAMERA_VFOV_RAD / 2.0)    # tan(24.5°) ≈ 0.4557

    def _marker_in_fov(self, drone_x: float, drone_y: float, drone_z: float) -> bool:
        """Check whether all four marker corners are within the camera FOV.

        Uses the exact closed-form inequalities derived from projecting
        each marker corner into the camera frame.  The critical quantity
        is the optical depth to the marker's closest vertical edge:

            min_depth = d - MARKER_HALF_WIDTH × |sin θ|

        where d is horizontal distance and θ is approach angle.  The
        corner at this depth is the geometric bottleneck for both
        horizontal and vertical visibility.

        The three constraints are:
          1. Horizontal:  hw × cos θ  ≤  min_depth × tan(HFOV/2)
          2. Upper z:     z  ≤  (marker_z - hw) + min_depth × tan(VFOV/2)
          3. Lower z:     z  ≥  (marker_z + hw) - min_depth × tan(VFOV/2)

        where hw = MARKER_HALF_WIDTH = 0.10 m.

        A 90% margin is applied to avoid spawns where corners sit at
        the very edge of the frame, which degrades solvePnP accuracy.

        Parameters
        ----------
        drone_x, drone_y, drone_z : float
            Candidate spawn position in world frame.

        Returns
        -------
        visible : bool
        """
        d = math.hypot(drone_x, drone_y)
        if d < 0.01:
            return False

        hw = MARKER_HALF_WIDTH  # 0.10 m
        theta = math.atan2(drone_y, drone_x)

        # Optical depth to the closest vertical edge of the marker
        min_depth = d - hw * abs(math.sin(theta))
        if min_depth <= 0.0:
            return False

        margin = 0.90  # 90% of FOV to avoid edge degradation

        # 1. Horizontal: all four corners within horizontal FOV
        if hw * abs(math.cos(theta)) > min_depth * self._TAN_HALF_HFOV * margin:
            return False

        # 2. Upper altitude limit: bottom corners don't drop below frame
        z_max = (MARKER_Z - hw) + min_depth * self._TAN_HALF_VFOV * margin
        if drone_z > z_max:
            return False

        # 3. Lower altitude limit: top corners don't push above frame
        z_min = (MARKER_Z + hw) - min_depth * self._TAN_HALF_VFOV * margin
        if drone_z < z_min:
            return False

        return True