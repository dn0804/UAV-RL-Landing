"""
Observation vector assembly.

Builds the 33-dimensional normalized observation from raw sensor data.
Owns the action history ring buffer.
The EMA velocity filter is external — drone_env calls it before passing
filtered velocities here.

Channel layout (33 total):
    [0]     x               forward distance to marker (vision, body frame)
    [1]     y               lateral offset (vision, body frame)
    [2]     z               vertical offset (vision, body frame)
    [3]     yaw             heading relative to marker (odom-derived)
    [4]     vx              forward velocity (EMA-filtered)
    [5]     vy              lateral velocity (EMA-filtered)
    [6]     vz              vertical velocity (EMA-filtered)
    [7]     z_tof           true vertical clearance to surface below
    [8]     dropout_timer   steps since last marker detection
    [9]     roll            body roll angle (IMU)
    [10]    pitch           body pitch angle (IMU)
    [11]    marker_px       marker pixel x in frame, normalized [-1, 1]
    [12]    marker_py       marker pixel y in frame, normalized [-1, 1]
    [13..32] action history 5 most recent actions × 4 axes, newest first
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
        marker_px: float = 0.0,
        marker_py: float = 0.0,
    ) -> np.ndarray:
        """Assemble, normalize, and clip the full observation vector.

        Parameters
        ----------
        marker_px, marker_py : float
            Marker center position in camera frame, normalized so that
            (0, 0) = frame center, (-1, -1) = top-left, (1, 1) = bottom-right.
            Frozen at last-known values during marker dropout.
        """

        # 1. Cap the dropout timer
        dropout_capped = min(dropout_timer, DROPOUT_TIMER_CAP)

        # 2. Assemble the 13 sensor channels
        sensor = np.array([
            x, y, z, yaw,
            vx, vy, vz,
            z_tof_raw,
            float(dropout_capped),
            roll,
            pitch,
            marker_px,
            marker_py,
        ], dtype=np.float32)

        # 3. Flatten action history (newest first)
        history_flat = np.concatenate(
            [self._action_history[i] for i in range(len(self._action_history) - 1, -1, -1)]
        )

        # 4. Concatenate into the full vector
        obs_raw = np.concatenate([sensor, history_flat])

        assert obs_raw.shape == (OBS_DIM,), (
            f"Observation has {obs_raw.shape[0]} elements, expected {OBS_DIM}"
        )

        # 5. Normalize and clip
        obs_normalized = obs_raw / OBS_SCALE
        obs_clipped = np.clip(obs_normalized, -OBS_CLIP, OBS_CLIP)

        return obs_clipped.astype(np.float32)