"""
Reward computation for UAV precision landing.

Pure function — takes pre-computed state quantities and returns the scalar
reward plus a per-component breakdown dict for TensorBoard logging.
All shaping rewards are progress-based (deltas), not state-based.

Reward equation:
    R_step = R_horizontal + R_descent + R_yaw + R_centering + R_velocity + R_jerk + R_time
"""

import math
import numpy as np

from rl_uav_package.config.constants import (
    W_HORIZONTAL,
    W_DESCENT,
    W_YAW, YAW_PENALTY_D_MIN,
    W_CENTERING, CENTERING_DEADZONE,
    W_VEL_XY, W_VEL_Z, VEL_PENALTY_D_REF,
    W_JERK,
    W_TIME,
    W_VEL_XY_UNI,
    MAX_VEL_XY
)


def compute_reward(
    # Horizontal centering
    d_pad: float,
    d_pad_prev: float,
    # Descent
    z: float,
    z_prev: float,
    # Yaw tracking
    yaw_error: float,
    d_marker: float,
    # Pixel centering
    marker_px: float,
    marker_py: float,
    # Velocity (for proximity-scaled penalty)
    vx: float,
    vy: float,
    vz: float,
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
    yaw_error : float
        Signed angular error between drone heading and bearing to marker (rad).
        Wrapped to [−π, π].
    d_marker : float
        Horizontal distance to the marker origin (m).  Used to scale the
        yaw penalty.
    marker_px : float
        Marker center x in camera frame, normalized to [-1, 1].
        0 = frame center, ±1 = frame edge.
    marker_py : float
        Marker center y in camera frame, normalized to [-1, 1].
    vx : float
        Forward velocity (m/s), world frame.
    vy : float
        Lateral velocity (m/s), world frame.
    vz : float
        Vertical velocity (m/s), world frame.
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
    # 2. Descent reward
    #    Plain delta-z: positive when descending, negative when ascending.
    #    No altitude gating — pixel centering penalty handles the
    #    "center before descending" behavior.
    # ==================================================================
    delta_z = z_prev - z    # positive when descending
    r_descent = W_DESCENT * delta_z

    # ==================================================================
    # 3. Yaw tracking penalty (mild shaping)
    #    Scales with inverse distance — more pressure at close range.
    #    Reduced weight (W_YAW = 0.006) since pixel centering is the
    #    primary "keep marker in frame" signal.
    # ==================================================================
    abs_yaw_error = abs(yaw_error)
    proximity_scale = YAW_PENALTY_D_MIN / max(d_marker, YAW_PENALTY_D_MIN)
    r_yaw = -W_YAW * (abs_yaw_error / math.pi) * proximity_scale

    # ==================================================================
    # 4. Pixel centering penalty
    #    Penalizes the marker drifting from the camera frame center.
    #    Dead zone: no penalty when within CENTERING_DEADZONE of center.
    #    Outside dead zone: penalty scales linearly with distance.
    # ==================================================================
    pixel_dist = math.hypot(marker_px, marker_py)
    centering_excess = max(0.0, pixel_dist - CENTERING_DEADZONE)
    r_centering = -W_CENTERING * centering_excess

    # ==================================================================
    # 5. Velocity penalty (proximity-scaled)
    #    Penalizes speed proportional to closeness to the pad.
    #    proximity = (d_ref / max(d_pad, d_ref))²:
    #      - Inside d_ref (0.5 m): full penalty weight.
    #      - At 2× d_ref: half weight.  At 4× d_ref: quarter weight.
    #    Horizontal and vertical are split so descent reward isn't
    #    undermined at range.
    # ==================================================================
    proximity = (VEL_PENALTY_D_REF / max(d_pad, VEL_PENALTY_D_REF))**2
    v_xy = math.hypot(vx, vy)
    r_vel_xy = -W_VEL_XY * v_xy * proximity
    r_vel_z = -W_VEL_Z * abs(vz) * proximity

    # Universal speed limit — flat penalty for excess above MAX_VEL_XY,
    # applied regardless of proximity.
    excess_xy = max(0.0, v_xy - MAX_VEL_XY)
    r_vel_xy_uni = -W_VEL_XY_UNI * excess_xy

    r_velocity = r_vel_xy + r_vel_z + r_vel_xy_uni

    # ==================================================================
    # 6. Jerk penalty
    #    L2 norm of the action difference vector.
    # ==================================================================
    r_jerk = -W_JERK * float(np.linalg.norm(action - prev_action))

    # ==================================================================
    # 7. Time penalty
    #    Constant per step — discourages hovering.
    # ==================================================================
    r_time = -W_TIME

    # ==================================================================
    # Total
    # ==================================================================
    reward = (r_horizontal + r_descent + r_yaw + r_centering
              + r_velocity + r_jerk + r_time)

    breakdown = {
        "reward/horizontal": r_horizontal,
        "reward/descent": r_descent,
        "reward/yaw": r_yaw,
        "reward/centering": r_centering,
        "reward/vel_xy": r_vel_xy,
        "reward/vel_z": r_vel_z,
        "reward/velocity": r_velocity,
        "reward/vel_xy_uni": r_vel_xy_uni,
        "reward/jerk": r_jerk,
        "reward/time": r_time,
        "reward/total": reward,
        "diag/d_pad": d_pad,
        "diag/delta_z": delta_z,
        "diag/pixel_dist": pixel_dist,
        "diag/v_xy": v_xy,
        "diag/proximity": proximity,
    }

    return reward, breakdown