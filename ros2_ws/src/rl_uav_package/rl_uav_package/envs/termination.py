"""
Episode termination checks.

Pure function — takes pre-computed state, returns termination verdict.
All thresholds are imported from config/constants.py so nothing is hardcoded here.

Reference: Project Plan §8.9 (Episode Termination)
"""

import math

from rl_uav_package.config.constants import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MAX,
    PAD_ELEVATION, LANDING_PAD_FORWARD_OFFSET,
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

# Radius around the pad center within which surface contact is evaluated
# against the desk surface.  Slightly larger than the desk to give
# a generous contact zone while keeping the desk-edge termination bug fixed.
# Beyond this radius the drone is over the floor, not the desk.
_PAD_CONTACT_RADIUS = 0.25  # m  (pad radius 0.10 + 0.15 m buffer)


def check_termination(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    vx: float,
    vy: float,
    vz: float,
    z_tof: float,  # (Passed in but no longer used for physical collision)
    d_pad: float,
    yaw_error: float,
    step: int,
    success_vz_max: float = SUCCESS_VZ_MAX,
    success_vxy_max: float = SUCCESS_VXY_MAX,
    success_d_xy_max: float = SUCCESS_D_XY_MAX,
    success_yaw_error_max: float = SUCCESS_YAW_ERROR_MAX,
) -> tuple[bool, bool, str, float, dict]:
    """Evaluate all episode-ending conditions.

    Returns
    -------
    terminated : bool
    truncated : bool
    outcome : str
    terminal_reward : float
    terminal_breakdown : dict
        Per-component breakdown for TensorBoard logging (empty for most
        outcomes; populated for success and graduated-crash).
    """

    # ------------------------------------------------------------------
    # 1. Attitude crash — drone flipped beyond recovery
    # ------------------------------------------------------------------
    if abs(roll) > CRASH_ROLL_MAX or abs(pitch) > CRASH_PITCH_MAX:
        return True, False, OUTCOME_CRASH_ATTITUDE, R_CRASH, {}

    # ------------------------------------------------------------------
    # 2. Surface contact — evaluate based on physical truth, not sensor math.
    #
    #    Contact detection is scoped to a circular zone around the pad center.
    #    The full desk rectangle is intentionally NOT used here — the desk edge
    #    at x=0.60 is not a landing surface, and using the full rectangle caused
    #    the agent to converge on landing at the desk edge rather than the pad.
    #
    #    The z_tof sensor (in compute_derived) still uses the full desk footprint
    #    because the physical sensor would genuinely read the desk surface below.
    #    Surface contact detection and ToF simulation are separate concerns.
    # ------------------------------------------------------------------
    d_from_pad_center = math.hypot(x - LANDING_PAD_FORWARD_OFFSET, y)
    is_near_pad = d_from_pad_center < _PAD_CONTACT_RADIUS
    surface_z = PAD_ELEVATION if is_near_pad else 0.0
    true_clearance = z - surface_z

    if true_clearance < TOF_CONTACT_THRESHOLD:
        v_xy = math.hypot(vx, vy)

        success = (
            d_pad < success_d_xy_max
            and abs(vz) < success_vz_max
            and v_xy < success_vxy_max
            and abs(yaw_error) < success_yaw_error_max
        )

        if success:
            # Soft-landing bonus: up to 10 pts each for landing with
            # velocity well below the curriculum threshold.
            # 0 velocity → full 10 pts; threshold velocity → 0 pts.
            vz_bonus  = 30.0 * (1.0 - abs(vz) / success_vz_max)
            vxy_bonus = 30.0 * (1.0 - v_xy / success_vxy_max)
            terminal_breakdown = {
                "terminal/vz_bonus": vz_bonus,
                "terminal/vxy_bonus": vxy_bonus,
            }
            return True, False, OUTCOME_SUCCESS, R_SUCCESS + vz_bonus + vxy_bonus, terminal_breakdown
        else:
            # Graduated penalty based on how close ALL success criteria were.
            # Denominators are the curriculum thresholds so the penalty scales
            # appropriately as criteria tighten across stages.
            pos_miss = min(d_pad / success_d_xy_max, 1.0)
            vxy_miss = min(v_xy / success_vxy_max, 1.0)
            vz_miss = min(abs(vz) / success_vz_max, 1.0)
            yaw_miss = min(abs(yaw_error) / success_yaw_error_max, 1.0)

            # Average of all miss factors — 0.0 = nearly perfect, 1.0 = way off
            miss = (pos_miss + vxy_miss + vz_miss + yaw_miss) / 4.0

            # Base penalty is R_CRASH (-100). Near miss stays near -75.
            # Bad miss reaches -150.  Crashing is always worse than
            # timing out (R_TIMEOUT = -50).
            graduated_penalty = (0.75 * R_CRASH) - (75.0 * miss)

            terminal_breakdown = {
                "terminal/miss_pos": pos_miss,
                "terminal/miss_vxy": vxy_miss,
                "terminal/miss_vz": vz_miss,
                "terminal/miss_yaw": yaw_miss,
                "terminal/miss_avg": miss,
            }
            return True, False, OUTCOME_CRASH_CONTACT, graduated_penalty, terminal_breakdown

    # ------------------------------------------------------------------
    # 3. Below-pad breach — drone altitude below the landing surface.
    #
    #    Scoped to the desk footprint only.  Everywhere else in the room the
    #    operational floor is the ground plane (z ≈ 0).  The previous global
    #    check (z < PAD_ELEVATION anywhere) forced the drone to maintain desk
    #    height across the entire room, which fed directly into the desk-edge
    #    collision exploit in check 2.
    #
    #    With MulticopterMotorModel the drone pitches/rolls to translate,
    #    causing transient altitude dips during every horizontal correction.
    #    The 2 cm margin was designed for kinematic control — with real
    #    attitude dynamics the drone routinely dips 5–10 cm below the desk
    #    surface during normal maneuvering.  15 cm margin prevents these
    #    physics-induced dips from being mistaken for crashes.  The desk
    #    collision geometry in the SDF still prevents actual penetration.
    # ------------------------------------------------------------------
    is_over_desk = (0.0 <= x <= 0.60) and (-0.50 <= y <= 0.50)
    if is_over_desk and z < PAD_ELEVATION - 0.15:
        return True, False, OUTCOME_CRASH_BELOW_PAD, R_CRASH, {}
    if not is_over_desk and z < 0.03:
        return True, False, OUTCOME_CRASH_BELOW_PAD, R_CRASH, {}

    # ------------------------------------------------------------------
    # 4. Operational volume — wall breach and boundary violations
    # ------------------------------------------------------------------
    if x < X_MIN:
        return True, False, OUTCOME_CRASH_WALL, R_CRASH, {}

    if x > X_MAX or y < Y_MIN or y > Y_MAX or z > Z_MAX:
        return True, False, OUTCOME_CRASH_OOB, R_CRASH, {}

    # ------------------------------------------------------------------
    # 5. Timeout — episode exceeds maximum length
    # ------------------------------------------------------------------
    if step >= MAX_STEPS:
        return False, True, OUTCOME_TIMEOUT, R_TIMEOUT, {}

    # ------------------------------------------------------------------
    # Episode continues.
    # ------------------------------------------------------------------
    return False, False, OUTCOME_ONGOING, 0.0, {}