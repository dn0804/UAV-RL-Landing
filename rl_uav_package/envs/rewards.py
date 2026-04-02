"""
Reward computation for UAV precision landing.

Checkpoint pipeline:
    Before checkpoint:  z-align → marker height, hover bonus active
    After checkpoint:   z-align → pad surface, hover bonus off, descent gradient
    Phase C (latched):  heavy descent, XY hold, no centering/yaw

Dropout freeze (pre-checkpoint only):
    When marker is lost for > DROPOUT_GRACE_STEPS, penalize XYZ action
    commands.  Yaw is free — teaches the drone to stop and search.
"""

import math
import numpy as np

from rl_uav_package.config.constants import (
    W_HORIZONTAL, W_Z_ALIGN,
    W_YAW, YAW_PENALTY_D_MIN,
    W_CENTERING, CENTERING_DEADZONE,
    W_VEL_XY, W_VEL_Z, VEL_PENALTY_D_REF,
    W_VEL_XY_UNI, MAX_VEL_XY,
    W_JERK, W_TIME,
    W_HOVER_BONUS, HOVER_Z_TOLERANCE,
    DESCENT_GATE_D_PAD,
    W_DESCENT_COMMITTED, W_XY_HOLD,
    W_DROPOUT_FREEZE, DROPOUT_GRACE_STEPS,
    PAD_ELEVATION,
)


def compute_reward(
    d_pad: float, d_pad_prev: float,
    z: float, z_prev: float,
    yaw_error: float, d_marker: float,
    marker_px: float, marker_py: float,
    vx: float, vy: float, vz: float,
    action: np.ndarray, prev_action: np.ndarray,
    marker_center_z: float,
    descent_committed: bool,
    hover_checkpoint_reached: bool,
    dropout_timer: int = 0,
) -> tuple[float, dict]:
    """Compute per-step shaping reward."""

    v_xy = math.hypot(vx, vy)
    r_jerk = -W_JERK * float(np.linalg.norm(action - prev_action))
    r_time = -W_TIME

    # ── Dropout freeze (pre-checkpoint only) ─────────────────
    # When marker is lost and checkpoint not yet reached, penalize
    # any XYZ movement commands.  Yaw (action[3]) is free so the
    # drone can search for the marker.
    dropout_freeze_active = (
        dropout_timer >= DROPOUT_GRACE_STEPS
        and not hover_checkpoint_reached
        and not descent_committed
    )
    if dropout_freeze_active:
        xyz_cmd = abs(float(action[0])) + abs(float(action[1])) + abs(float(action[2]))
        r_dropout_freeze = -W_DROPOUT_FREEZE * xyz_cmd
    else:
        r_dropout_freeze = 0.0

    # ==================================================================
    # PHASE C: Committed descent (blind landing)
    # ==================================================================
    if descent_committed:
        delta_z = z_prev - z
        r_descent = W_DESCENT_COMMITTED * delta_z
        r_xy_hold = -W_XY_HOLD * v_xy

        reward = r_descent + r_xy_hold + r_jerk + r_time

        return reward, {
            "reward/descent_committed": r_descent,
            "reward/xy_hold": r_xy_hold,
            "reward/jerk": r_jerk,
            "reward/time": r_time,
            "reward/horizontal": 0.0,
            "reward/z_align": 0.0,
            "reward/yaw": 0.0,
            "reward/centering": 0.0,
            "reward/velocity": 0.0,
            "reward/hover_bonus": 0.0,
            "reward/dropout_freeze": 0.0,
            "reward/total": reward,
            "diag/d_pad": d_pad,
            "diag/delta_z": delta_z,
            "diag/v_xy": v_xy,
            "diag/phase": 3,
        }

    # ==================================================================
    # PHASE A/B: Approach + hover / post-checkpoint descent
    # ==================================================================

    r_horizontal = W_HORIZONTAL * (d_pad_prev - d_pad)

    # Z-alignment: target marker before checkpoint, pad after
    near_pad = d_pad < DESCENT_GATE_D_PAD
    if hover_checkpoint_reached:
        z_target = PAD_ELEVATION
    else:
        z_target = marker_center_z

    prev_z_dist = abs(z_prev - z_target)
    curr_z_dist = abs(z - z_target)
    r_z_align = W_Z_ALIGN * (prev_z_dist - curr_z_dist)

    # Yaw tracking
    abs_yaw_error = abs(yaw_error)
    proximity_scale = YAW_PENALTY_D_MIN / max(d_marker, YAW_PENALTY_D_MIN)
    r_yaw = -W_YAW * (abs_yaw_error / math.pi) * proximity_scale

    # Pixel centering
    pixel_dist = math.hypot(marker_px, marker_py)
    centering_excess = max(0.0, pixel_dist - CENTERING_DEADZONE)
    r_centering = -W_CENTERING * centering_excess

    # Velocity penalty
    proximity = (VEL_PENALTY_D_REF / max(d_pad, VEL_PENALTY_D_REF)) ** 2
    r_vel_xy = -W_VEL_XY * v_xy * proximity
    r_vel_z = -W_VEL_Z * abs(vz) * proximity
    excess_xy = max(0.0, v_xy - MAX_VEL_XY)
    r_vel_xy_uni = -W_VEL_XY_UNI * excess_xy
    r_velocity = r_vel_xy + r_vel_z + r_vel_xy_uni

    # Hover bonus (pre-checkpoint only)
    at_marker_height = abs(z - marker_center_z) < HOVER_Z_TOLERANCE
    if not hover_checkpoint_reached and near_pad and at_marker_height:
        r_hover = W_HOVER_BONUS
        phase = 1
    else:
        r_hover = 0.0
        phase = 2 if hover_checkpoint_reached else 0

    reward = (r_horizontal + r_z_align + r_yaw + r_centering
              + r_velocity + r_jerk + r_time + r_hover + r_dropout_freeze)

    return reward, {
        "reward/horizontal": r_horizontal,
        "reward/z_align": r_z_align,
        "reward/yaw": r_yaw,
        "reward/centering": r_centering,
        "reward/vel_xy": r_vel_xy,
        "reward/vel_z": r_vel_z,
        "reward/velocity": r_velocity,
        "reward/vel_xy_uni": r_vel_xy_uni,
        "reward/jerk": r_jerk,
        "reward/time": r_time,
        "reward/hover_bonus": r_hover,
        "reward/dropout_freeze": r_dropout_freeze,
        "reward/descent_committed": 0.0,
        "reward/xy_hold": 0.0,
        "reward/total": reward,
        "diag/d_pad": d_pad,
        "diag/delta_z": z_prev - z,
        "diag/pixel_dist": pixel_dist,
        "diag/v_xy": v_xy,
        "diag/proximity": proximity,
        "diag/z_dist_to_marker": abs(z - marker_center_z),
        "diag/z_dist_to_target": curr_z_dist,
        "diag/phase": phase,
    }