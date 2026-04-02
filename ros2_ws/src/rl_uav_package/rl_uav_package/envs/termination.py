"""
Episode termination checks.

Operational volume is a 3D cone with tip behind the marker, opening in
+X direction (toward the drone's approach path).  Flat floor at Z_FLOOR,
flat ceiling at Z_CEILING.
"""

import math

from rl_uav_package.config.constants import (
    CONE_TIP_X, CONE_TIP_Y, CONE_TIP_Z,
    CONE_COS_HALF_ANGLE, CONE_LENGTH,
    Z_FLOOR, Z_CEILING,
    PAD_ELEVATION, PAD_WORLD_POS,
    DESK_X_MIN, DESK_X_MAX, DESK_Y_MIN, DESK_Y_MAX,
    TOF_CONTACT_THRESHOLD,
    SUCCESS_D_XY_MAX, SUCCESS_VZ_MAX, SUCCESS_VXY_MAX,
    CRASH_ROLL_MAX, CRASH_PITCH_MAX,
    MAX_STEPS,
    R_SUCCESS, R_SUCCESS_NO_CHECKPOINT, R_CRASH, R_TIMEOUT,
)

OUTCOME_SUCCESS = "success"
OUTCOME_SUCCESS_NO_CHECKPOINT = "success_no_checkpoint"
OUTCOME_CRASH_ATTITUDE = "crash_attitude"
OUTCOME_CRASH_CONTACT = "crash_contact_off_target"
OUTCOME_CRASH_BELOW_PAD = "crash_below_pad"
OUTCOME_CRASH_OOB = "crash_oob"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_ONGOING = "ongoing"

_PAD_CONTACT_RADIUS = 0.30


def _inside_cone(x: float, y: float, z: float) -> bool:
    """Check if a point is inside the operational cone.

    The cone has its tip at (CONE_TIP_X, CONE_TIP_Y, CONE_TIP_Z) and
    opens in the +X direction with half-angle defined by CONE_COS_HALF_ANGLE.
    """
    dx = x - CONE_TIP_X
    dy = y - CONE_TIP_Y
    dz = z - CONE_TIP_Z

    # Must be in front of the tip (+X direction)
    if dx <= 0.0:
        return False

    # Distance from tip to point
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-6:
        return True  # at the tip

    # Distance along cone axis (+X)
    if dx > CONE_LENGTH:
        return False

    # Angle check: cos(angle) = dot(vec, axis) / |vec| = dx / dist
    cos_angle = dx / dist
    return cos_angle >= CONE_COS_HALF_ANGLE


def check_termination(
    x: float, y: float, z: float,
    roll: float, pitch: float,
    vx: float, vy: float, vz: float,
    z_tof: float, d_pad: float, step: int,
    success_vz_max: float = SUCCESS_VZ_MAX,
    success_vxy_max: float = SUCCESS_VXY_MAX,
    success_d_xy_max: float = SUCCESS_D_XY_MAX,
    max_steps: int = MAX_STEPS,
    hover_checkpoint_reached: bool = False,
) -> tuple[bool, bool, str, float, dict]:
    """Evaluate all episode-ending conditions."""

    # 1. Attitude crash
    if abs(roll) > CRASH_ROLL_MAX or abs(pitch) > CRASH_PITCH_MAX:
        return True, False, OUTCOME_CRASH_ATTITUDE, R_CRASH, {}

    # 2. Surface contact
    d_from_pad_center = math.hypot(x - PAD_WORLD_POS[0], y - PAD_WORLD_POS[1])
    is_near_pad = d_from_pad_center < _PAD_CONTACT_RADIUS
    surface_z = PAD_ELEVATION if is_near_pad else 0.0
    true_clearance = z - surface_z

    if true_clearance < TOF_CONTACT_THRESHOLD:
        v_xy = math.hypot(vx, vy)

        success = (
            d_pad < success_d_xy_max
            and abs(vz) < success_vz_max
            and v_xy < success_vxy_max
        )

        if success:
            vz_bonus = 2.0 * (1.0 - abs(vz) / success_vz_max)
            vxy_bonus = 2.0 * (1.0 - v_xy / success_vxy_max)
            if hover_checkpoint_reached:
                return (True, False, OUTCOME_SUCCESS,
                        R_SUCCESS + vz_bonus + vxy_bonus,
                        {"terminal/vz_bonus": vz_bonus,
                         "terminal/vxy_bonus": vxy_bonus})
            else:
                return (True, False, OUTCOME_SUCCESS_NO_CHECKPOINT,
                        R_SUCCESS_NO_CHECKPOINT + vz_bonus + vxy_bonus,
                        {"terminal/vz_bonus": vz_bonus,
                         "terminal/vxy_bonus": vxy_bonus})
        else:
            pos_miss = min(d_pad / success_d_xy_max, 1.0)
            vxy_miss = min(v_xy / success_vxy_max, 1.0)
            vz_miss = min(abs(vz) / success_vz_max, 1.0)
            miss = (pos_miss + vxy_miss + vz_miss) / 3.0
            graduated_penalty = -20.0 - 15.0 * miss
            return (True, False, OUTCOME_CRASH_CONTACT, graduated_penalty,
                    {"terminal/miss_pos": pos_miss, "terminal/miss_vxy": vxy_miss,
                     "terminal/miss_vz": vz_miss, "terminal/miss_avg": miss})

    # 3. Below-pad breach (desk area only)
    is_over_desk = (DESK_X_MIN <= x <= DESK_X_MAX
                    and DESK_Y_MIN <= y <= DESK_Y_MAX)
    if is_over_desk and z < PAD_ELEVATION - 0.15:
        return True, False, OUTCOME_CRASH_BELOW_PAD, R_CRASH, {}

    # 4. Floor and ceiling
    if z < Z_FLOOR:
        return True, False, OUTCOME_CRASH_OOB, R_CRASH, {}
    if z > Z_CEILING:
        return True, False, OUTCOME_CRASH_OOB, R_CRASH, {}

    # 5. Conical operational volume
    if not _inside_cone(x, y, z):
        return True, False, OUTCOME_CRASH_OOB, R_CRASH, {}

    # 6. Timeout
    if step >= max_steps:
        return False, True, OUTCOME_TIMEOUT, R_TIMEOUT, {}

    return False, False, OUTCOME_ONGOING, 0.0, {}