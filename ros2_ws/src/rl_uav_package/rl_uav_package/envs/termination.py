"""
Episode termination checks.

Pure function — takes pre-computed state, returns termination verdict.
All thresholds are imported from config/constants.py so nothing is hardcoded here.

Reference: Project Plan §8.9 (Episode Termination)
"""

import math
import numpy as np

from rl_uav_package.config.constants import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MAX,
    PAD_ELEVATION,
    TOF_CONTACT_THRESHOLD,
    SUCCESS_D_XY_MAX, SUCCESS_VZ_MAX, SUCCESS_VXY_MAX, SUCCESS_YAW_ERROR_MAX,
    CRASH_ROLL_MAX, CRASH_PITCH_MAX,
    MAX_STEPS,
    R_SUCCESS, R_CRASH, R_TIMEOUT,
)

# Outcome labels — used by the logging module to track termination statistics.
OUTCOME_SUCCESS = "success"
OUTCOME_CRASH_ATTITUDE = "crash_attitude"
OUTCOME_CRASH_CONTACT = "crash_contact_off_target"
OUTCOME_CRASH_BELOW_PAD = "crash_below_pad"
OUTCOME_CRASH_WALL = "crash_wall"
OUTCOME_CRASH_OOB = "crash_oob"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_ONGOING = "ongoing"


def check_termination(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    vx: float,
    vy: float,
    vz: float,
    z_tof: float,
    d_pad: float,
    yaw_error: float,
    step: int,
) -> tuple[bool, bool, str, float]:
    """Evaluate all episode-ending conditions.

    Parameters
    ----------
    x, y, z : float
        Drone position in world frame (m).
    roll, pitch : float
        Drone attitude (rad).
    vx, vy, vz : float
        Drone velocity (m/s), world frame.
    z_tof : float
        Cosine-corrected ToF range reading (m).
    d_pad : float
        Horizontal distance to landing pad center (m).
    yaw_error : float
        Signed angular error between drone heading and marker bearing (rad).
    step : int
        Current step count within the episode.

    Returns
    -------
    terminated : bool
        True if the episode ended due to a terminal condition (crash or success).
    truncated : bool
        True if the episode ended due to tim§e limit.
    outcome : str
        One of the OUTCOME_* labels, or OUTCOME_ONGOING if the episode continues.
    terminal_reward : float
        Bonus/penalty applied at termination. Zero if ongoing or timed out.
    """

    # ------------------------------------------------------------------
    # 1. Attitude crash — drone flipped beyond recovery
    # ------------------------------------------------------------------
    if abs(roll) > CRASH_ROLL_MAX or abs(pitch) > CRASH_PITCH_MAX:
        return True, False, OUTCOME_CRASH_ATTITUDE, R_CRASH

    # ------------------------------------------------------------------
    # 2. Surface contact — ToF below threshold
    #    This fires on ANY surface: pad, ground, or furniture.
    #    Must be checked before the below-pad condition, because a
    #    successful touchdown on the pad occurs at z ≈ PAD_ELEVATION
    #    and the ToF reads ~0 before z dips below PAD_ELEVATION.
    # ------------------------------------------------------------------
    if z_tof < TOF_CONTACT_THRESHOLD:
        v_xy = math.hypot(vx, vy)

        success = (
            d_pad < SUCCESS_D_XY_MAX
            and abs(vz) < SUCCESS_VZ_MAX
            and v_xy < SUCCESS_VXY_MAX
            and abs(yaw_error) < SUCCESS_YAW_ERROR_MAX
        )

        if success:
            return True, False, OUTCOME_SUCCESS, R_SUCCESS
        else:
            return True, False, OUTCOME_CRASH_CONTACT, R_CRASH

    # ------------------------------------------------------------------
    # 3. Below-pad breach — drone altitude below the landing surface
    #    No useful recovery exists from below the desk.
    # ------------------------------------------------------------------
    if z < PAD_ELEVATION:
        return True, False, OUTCOME_CRASH_BELOW_PAD, R_CRASH

    # ------------------------------------------------------------------
    # 4. Operational volume — wall breach and boundary violations
    # ------------------------------------------------------------------
    if x < X_MIN:
        return True, False, OUTCOME_CRASH_WALL, R_CRASH

    if x > X_MAX or y < Y_MIN or y > Y_MAX or z > Z_MAX:
        return True, False, OUTCOME_CRASH_OOB, R_CRASH

    # ------------------------------------------------------------------
    # 5. Timeout — episode exceeds maximum length
    # ------------------------------------------------------------------
    if step >= MAX_STEPS:
        return False, True, OUTCOME_TIMEOUT, R_TIMEOUT

    # ------------------------------------------------------------------
    # Episode continues.
    # ------------------------------------------------------------------
    return False, False, OUTCOME_ONGOING, 0.0