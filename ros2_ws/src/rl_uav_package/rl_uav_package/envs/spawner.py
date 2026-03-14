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

    def _marker_in_fov(self, drone_x: float, drone_y: float, drone_z: float) -> bool:
        """Check whether the marker would be visible in the camera frame.

        At spawn the drone faces the marker, so the marker is horizontally
        centered.  The only real FOV risk is vertical: if the drone is
        much higher than the marker and close, the marker drops below the
        camera's vertical field of view.

        Parameters
        ----------
        drone_x, drone_y, drone_z : float
            Candidate spawn position in world frame.

        Returns
        -------
        visible : bool
        """
        d_horizontal = math.hypot(drone_x, drone_y)

        # Prevent division by zero when spawning extremely close to the wall
        if d_horizontal < 0.01:
            return False

        # Vertical angle from the drone's optical axis down to the marker.
        # Positive means the marker is below the camera center.
        height_above_marker = drone_z - MARKER_Z
        vertical_angle = math.atan2(height_above_marker, d_horizontal)

        # The camera's vertical FOV extends ±VFOV/2 from the optical axis.
        # The marker must be within this range.  We use a small margin (90%
        # of the half-FOV) to avoid spawns where the marker sits right at
        # the frame edge, which degrades solvePnP accuracy.
        vfov_margin = 0.90
        if abs(vertical_angle) > (CAMERA_VFOV_RAD / 2.0) * vfov_margin:
            return False

        # Horizontal check: the marker should be roughly centered since the
        # drone faces it.  This is a safety net — at spawn, yaw is computed
        # to aim at the marker, so horizontal offset should be near zero.
        # But verify the approach angle doesn't push the marker outside the
        # horizontal FOV (can happen at extreme angles if the marker is at
        # the wall and the drone's FOV clips the wall at a glancing angle).
        hfov_margin = 0.90
        horizontal_angle = abs(math.atan2(abs(drone_y), drone_x))
        # This is the angle of the marker relative to the drone's heading.
        # Since the drone points at (0, 0), the marker is on the optical axis.
        # The actual horizontal offset in the camera frame is near zero.
        # The meaningful check: can the camera *physically see* the wall plane
        # at the marker location?  At steep approach angles, the wall is
        # nearly parallel to the line of sight, but the marker itself is
        # what we're detecting, not the wall, so this rarely fails.
        # We include it for completeness.
        bearing_to_marker = math.atan2(0.0 - drone_y, 0.0 - drone_x)
        yaw = bearing_to_marker  # drone heading at spawn
        # Angle from camera center to marker in the horizontal plane
        h_offset = abs(bearing_to_marker - yaw)  # always 0 by construction
        if h_offset > (CAMERA_HFOV_RAD / 2.0) * hfov_margin:
            return False

        return True