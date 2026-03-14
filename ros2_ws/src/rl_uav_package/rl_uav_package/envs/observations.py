"""
Observation vector assembly.

Builds the 29-dimensional normalized observation from raw sensor data.
Owns the action history ring buffer and applies ToF cosine correction.
The EMA velocity filter is external — drone_env calls it before passing
filtered velocities here.

Channel layout (29 total):
    [0]     x               forward distance to marker (vision or odom)
    [1]     y               lateral offset
    [2]     z               vertical offset
    [3]     yaw             heading relative to marker
    [4]     vx              forward velocity (EMA-filtered)
    [5]     vy              lateral velocity (EMA-filtered)
    [6]     vz              vertical velocity (EMA-filtered)
    [7]     z_tof           cosine-corrected ToF range
    [8]     dropout_timer   steps since last marker detection
    [9..28] action history  5 most recent actions × 4 axes, newest first
"""

import math
from collections import deque

import numpy as np

from rl_uav_package.config.constants import (
    OBS_DIM,
    ACTION_DIM,
    ACTION_HISTORY_STEPS,
    OBS_SCALE,
    OBS_CLIP,
    DROPOUT_TIMER_CAP,
)


class ObservationBuilder:
    """Stateful builder that assembles, normalizes, and clips observations.

    Maintains the action history ring buffer internally.  Call ``push_action``
    each step *before* ``build`` so the observation reflects the most recent
    command the agent issued.
    """

    def __init__(self):
        self._action_history: deque[np.ndarray] = deque(maxlen=ACTION_HISTORY_STEPS)
        self.reset()

    # ── Public API ───────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all internal state for a new episode.

        Fills the action history with zeros — at the start of an episode
        the agent has no prior commands.
        """
        self._action_history.clear()
        for _ in range(ACTION_HISTORY_STEPS):
            self._action_history.append(np.zeros(ACTION_DIM, dtype=np.float32))

    def push_action(self, action: np.ndarray) -> None:
        """Record the latest action into the history ring buffer.

        Call this once per step, *before* ``build()``, so the observation
        the agent receives includes the action it just took as the most
        recent entry in the history.

        Parameters
        ----------
        action : array-like, shape (4,)
            The action sent to the drone this step, in [-1, 1].
        """
        self._action_history.append(np.asarray(action, dtype=np.float32).copy())

    def build(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        vx: float,
        vy: float,
        vz: float,
        z_tof_raw: float,
        roll: float,
        pitch: float,
        dropout_timer: int,
    ) -> np.ndarray:
        """Assemble, normalize, and clip the full observation vector.

        Parameters
        ----------
        x, y, z : float
            Position relative to marker (m).  Source is vision when
            available, odom ground truth for the baseline.
        yaw : float
            Heading relative to marker (rad).
        vx, vy, vz : float
            Velocity (m/s), already passed through the EMA filter.
        z_tof_raw : float
            Raw ToF range reading (m), *before* cosine correction.
        roll, pitch : float
            Drone attitude (rad), from IMU.  Used for ToF correction.
        dropout_timer : int
            Steps since the marker was last detected.  Zero when visible.

        Returns
        -------
        obs : np.ndarray, shape (29,), dtype float32
            Normalized and clipped observation vector.
        """
        # 1. Cosine-correct the ToF range
        z_tof = z_tof_raw * math.cos(pitch) * math.cos(roll)

        # 2. Cap the dropout timer
        dropout_capped = min(dropout_timer, DROPOUT_TIMER_CAP)

        # 3. Assemble the 9 sensor channels
        sensor = np.array([
            x, y, z, yaw,
            vx, vy, vz,
            z_tof,
            float(dropout_capped),
        ], dtype=np.float32)

        # 4. Flatten action history (newest first)
        #    deque order: oldest at [0], newest at [-1]
        #    We want newest first in the obs, so reverse.
        history_flat = np.concatenate(
            [self._action_history[i] for i in range(len(self._action_history) - 1, -1, -1)]
        )

        # 5. Concatenate into the full vector
        obs_raw = np.concatenate([sensor, history_flat])

        assert obs_raw.shape == (OBS_DIM,), (
            f"Observation has {obs_raw.shape[0]} elements, expected {OBS_DIM}"
        )

        # 6. Normalize and clip
        obs_normalized = obs_raw / OBS_SCALE
        obs_clipped = np.clip(obs_normalized, -OBS_CLIP, OBS_CLIP)

        return obs_clipped.astype(np.float32)