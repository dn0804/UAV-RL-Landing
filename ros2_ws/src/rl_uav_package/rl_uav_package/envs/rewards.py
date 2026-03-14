"""
Reward computation for UAV precision landing.

Pure function — takes pre-computed state quantities and returns the scalar
reward plus a per-component breakdown dict for TensorBoard logging.
All shaping rewards are progress-based (deltas), not state-based.

Reward equation:
    R_step = R_horizontal + R_descent + R_yaw + R_jerk + R_time
"""

import math
import numpy as np

from rl_uav_package.config.constants import (
    W_HORIZONTAL,
    W_DESCENT, DESCENT_GATE_H_MID, DESCENT_GATE_TAU, DESCENT_GATE_SIGMA,
    W_YAW, YAW_PENALTY_D_MIN,
    W_JERK,
    W_TIME,
)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def compute_reward(
    # Horizontal centering
    d_pad: float,
    d_pad_prev: float,
    # Descent
    z: float,
    z_prev: float,
    h_above_marker: float,
    # Yaw tracking
    yaw_error: float,
    d_marker: float,
    # Jerk
    action: np.ndarray,
    prev_action: np.ndarray,
) -> tuple[float, dict]:
    """Compute the per-step shaping reward.

    Parameters
    ----------
    d_pad : float
        Current horizontal distance to landing pad center (m).
    d_pad_prev : float
        Previous step's horizontal distance to pad center (m).
    z : float
        Current drone altitude, world frame (m).
    z_prev : float
        Previous step's drone altitude (m).
    h_above_marker : float
        Height of drone above the ArUco marker (m).  Computed as
        z_drone − MARKER_Z.  Can be negative if drone is below marker.
    yaw_error : float
        Signed angular error between drone heading and bearing to marker (rad).
        Wrapped to [−π, π].
    d_marker : float
        Horizontal distance to the marker origin (m).  Used to scale the
        yaw penalty — distinct from d_pad because the marker is on the wall
        and the pad is 0.30 m in front of it.
    action : np.ndarray, shape (4,)
        Current action [vx, vy, vz, yaw_rate], each in [−1, 1].
    prev_action : np.ndarray, shape (4,)
        Previous step's action.

    Returns
    -------
    reward : float
        Total per-step reward (sum of all components).
    breakdown : dict
        Individual components keyed by name, for TensorBoard logging.
    """

    # ==================================================================
    # 1. Horizontal centering reward
    #    Positive when horizontal distance to pad decreases.
    # ==================================================================
    r_horizontal = W_HORIZONTAL * (d_pad_prev - d_pad)

    # ==================================================================
    # 2. Gated descent reward
    #
    #    High altitude (h >> h_mid):  gate_blend → 0, gate → 1
    #        → descent is freely rewarded regardless of centering.
    #    Low altitude  (h << h_mid):  gate_blend → 1, gate → Gaussian
    #        → descent rewarded only when horizontally centered.
    #    Ascending (delta_z ≤ 0): always penalized uniformly (ungated)
    #        → prevents "fly away then ascend cheaply" exploit.
    # ==================================================================
    delta_z = z_prev - z                          # positive when descending

    gate_blend = _sigmoid((DESCENT_GATE_H_MID - h_above_marker) / DESCENT_GATE_TAU)
    gaussian = math.exp(-(d_pad ** 2) / (2.0 * DESCENT_GATE_SIGMA ** 2))
    centering_gate = gate_blend * gaussian + (1.0 - gate_blend)

    if delta_z > 0.0:
        # Descending — gated by centering quality at low altitude.
        r_descent = W_DESCENT * delta_z * centering_gate
    else:
        # Ascending — penalized uniformly, no centering discount.
        r_descent = W_DESCENT * delta_z

    # ==================================================================
    # 3. Marker tracking penalty
    #    Scales with inverse distance — more pressure at close range
    #    where losing the marker from the FOV is catastrophic.
    # ==================================================================
    abs_yaw_error = abs(yaw_error)
    proximity_scale = YAW_PENALTY_D_MIN / max(d_marker, YAW_PENALTY_D_MIN)
    r_yaw = -W_YAW * (abs_yaw_error / math.pi) * proximity_scale

    # ==================================================================
    # 4. Jerk penalty
    #    L2 norm of the action difference vector.
    # ==================================================================
    r_jerk = -W_JERK * float(np.linalg.norm(action - prev_action))

    # ==================================================================
    # 5. Time penalty
    #    Constant per step — discourages hovering.
    # ==================================================================
    r_time = -W_TIME

    # ==================================================================
    # Total
    # ==================================================================
    reward = r_horizontal + r_descent + r_yaw + r_jerk + r_time

    breakdown = {
        "reward/horizontal": r_horizontal,
        "reward/descent": r_descent,
        "reward/yaw": r_yaw,
        "reward/jerk": r_jerk,
        "reward/time": r_time,
        "reward/total": reward,
        # Diagnostic values (not reward components, but useful for debugging)
        "diag/d_pad": d_pad,
        "diag/h_above_marker": h_above_marker,
        "diag/centering_gate": centering_gate,
        "diag/gate_blend": gate_blend,
        "diag/delta_z": delta_z,
    }

    return reward, breakdown