"""
Unit tests for envs/termination.py

Run with:  python -m pytest test_termination.py -v
"""

import math
import pytest

from rl_uav_package.envs.termination import (
    check_termination,
    OUTCOME_SUCCESS, OUTCOME_CRASH_ATTITUDE, OUTCOME_CRASH_CONTACT,
    OUTCOME_CRASH_BELOW_PAD, OUTCOME_CRASH_WALL, OUTCOME_CRASH_OOB,
    OUTCOME_TIMEOUT, OUTCOME_ONGOING,
)
from rl_uav_package.config.constants import (
    PAD_ELEVATION, MARKER_Z, MAX_STEPS,
    SUCCESS_D_XY_MAX, SUCCESS_VZ_MAX, SUCCESS_VXY_MAX, SUCCESS_YAW_ERROR_MAX,
    TOF_CONTACT_THRESHOLD,
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MAX,
    R_SUCCESS, R_CRASH, R_TIMEOUT,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _default_state(**overrides):
    """Return a safe mid-flight state dict, with any overrides applied."""
    state = dict(
        x=2.0, y=0.0, z=MARKER_Z + 0.3,        # well above pad, mid-range
        roll=0.0, pitch=0.0,
        vx=0.0, vy=0.0, vz=0.0,
        z_tof=MARKER_Z + 0.3,                    # over ground, altitude = z
        d_pad=1.7,                                # far from pad
        yaw_error=0.0,
        step=50,
    )
    state.update(overrides)
    return state


# ── Ongoing (no termination) ────────────────────────────────────────

class TestOngoing:
    def test_normal_flight(self):
        t, tr, outcome, r = check_termination(**_default_state())
        assert not t and not tr
        assert outcome == OUTCOME_ONGOING
        assert r == 0.0

    def test_step_one(self):
        """Very first step should not terminate."""
        t, tr, outcome, _ = check_termination(**_default_state(step=0))
        assert outcome == OUTCOME_ONGOING

    def test_just_under_max_steps(self):
        t, tr, outcome, _ = check_termination(**_default_state(step=MAX_STEPS - 1))
        assert outcome == OUTCOME_ONGOING


# ── Attitude crash ──────────────────────────────────────────────────

class TestAttitudeCrash:
    def test_roll_positive(self):
        t, _, outcome, r = check_termination(**_default_state(roll=math.radians(46)))
        assert t and outcome == OUTCOME_CRASH_ATTITUDE and r == R_CRASH

    def test_roll_negative(self):
        t, _, outcome, _ = check_termination(**_default_state(roll=math.radians(-46)))
        assert t and outcome == OUTCOME_CRASH_ATTITUDE

    def test_pitch_positive(self):
        t, _, outcome, _ = check_termination(**_default_state(pitch=math.radians(46)))
        assert t and outcome == OUTCOME_CRASH_ATTITUDE

    def test_pitch_negative(self):
        t, _, outcome, _ = check_termination(**_default_state(pitch=math.radians(-46)))
        assert t and outcome == OUTCOME_CRASH_ATTITUDE

    def test_exactly_at_limit_no_crash(self):
        """45° exactly should NOT crash (using > not >=)."""
        t, _, outcome, _ = check_termination(**_default_state(roll=math.radians(45)))
        assert outcome == OUTCOME_ONGOING

    def test_attitude_takes_priority_over_oob(self):
        """Flipped drone that is also out of bounds → attitude crash, not OOB."""
        t, _, outcome, _ = check_termination(**_default_state(
            roll=math.radians(50), x=10.0))
        assert outcome == OUTCOME_CRASH_ATTITUDE


# ── Surface contact ─────────────────────────────────────────────────

class TestSurfaceContact:
    """ToF < 0.10 m triggers contact. Outcome depends on success criteria."""

    def _landed_state(self, **overrides):
        """Drone sitting on the pad with perfect conditions."""
        state = dict(
            x=0.30, y=0.0, z=PAD_ELEVATION,      # right on the pad
            roll=0.0, pitch=0.0,
            vx=0.0, vy=0.0, vz=-0.1,             # gentle descent
            z_tof=0.02,                            # below threshold
            d_pad=0.0,                             # centered on pad
            yaw_error=0.0,
            step=100,
        )
        state.update(overrides)
        return state

    def test_perfect_landing(self):
        t, _, outcome, r = check_termination(**self._landed_state())
        assert t and outcome == OUTCOME_SUCCESS and r == R_SUCCESS

    def test_off_target(self):
        """On the ground but too far from pad center."""
        t, _, outcome, r = check_termination(**self._landed_state(
            d_pad=SUCCESS_D_XY_MAX + 0.01))
        assert t and outcome == OUTCOME_CRASH_CONTACT and r == R_CRASH

    def test_too_fast_vertically(self):
        t, _, outcome, _ = check_termination(**self._landed_state(
            vz=-(SUCCESS_VZ_MAX + 0.01)))
        assert outcome == OUTCOME_CRASH_CONTACT

    def test_too_fast_horizontally(self):
        t, _, outcome, _ = check_termination(**self._landed_state(
            vx=SUCCESS_VXY_MAX + 0.01))
        assert outcome == OUTCOME_CRASH_CONTACT

    def test_diagonal_horizontal_speed(self):
        """v_xy = hypot(vx, vy) must be under the cap."""
        # Each axis at 0.15 → hypot = 0.212 > 0.2
        t, _, outcome, _ = check_termination(**self._landed_state(vx=0.15, vy=0.15))
        assert outcome == OUTCOME_CRASH_CONTACT

    def test_bad_yaw(self):
        t, _, outcome, _ = check_termination(**self._landed_state(
            yaw_error=SUCCESS_YAW_ERROR_MAX + 0.01))
        assert outcome == OUTCOME_CRASH_CONTACT

    def test_at_exact_thresholds_is_success(self):
        """All values exactly at the limits (using strict <, not <=)."""
        t, _, outcome, _ = check_termination(**self._landed_state(
            d_pad=SUCCESS_D_XY_MAX - 0.001,
            vz=-(SUCCESS_VZ_MAX - 0.001),
            vx=SUCCESS_VXY_MAX - 0.001,
            yaw_error=SUCCESS_YAW_ERROR_MAX - 0.001,
        ))
        assert outcome == OUTCOME_SUCCESS

    def test_contact_on_furniture(self):
        """ToF triggered by a desk surface mid-approach — failed landing."""
        t, _, outcome, _ = check_termination(**self._landed_state(
            z=PAD_ELEVATION + 0.3, d_pad=2.0, z_tof=0.05))
        assert outcome == OUTCOME_CRASH_CONTACT


# ── Below-pad breach ────────────────────────────────────────────────

class TestBelowPad:
    def test_below_pad_over_ground(self):
        """Flying at z < PAD_ELEVATION with no surface contact."""
        t, _, outcome, _ = check_termination(**_default_state(
            z=PAD_ELEVATION - 0.01, z_tof=PAD_ELEVATION - 0.01))
        assert t and outcome == OUTCOME_CRASH_BELOW_PAD

    def test_exactly_at_pad_no_crash(self):
        """z == PAD_ELEVATION should NOT trigger below-pad (using <, not <=)."""
        t, _, outcome, _ = check_termination(**_default_state(
            z=PAD_ELEVATION, z_tof=PAD_ELEVATION))
        assert outcome == OUTCOME_ONGOING


# ── Boundary violations ────────────────────────────────────────────

class TestBoundaries:
    def test_wall_breach(self):
        t, _, outcome, _ = check_termination(**_default_state(x=X_MIN - 0.01))
        assert t and outcome == OUTCOME_CRASH_WALL

    def test_x_max(self):
        t, _, outcome, _ = check_termination(**_default_state(x=X_MAX + 0.01))
        assert t and outcome == OUTCOME_CRASH_OOB

    def test_y_min(self):
        t, _, outcome, _ = check_termination(**_default_state(y=Y_MIN - 0.01))
        assert t and outcome == OUTCOME_CRASH_OOB

    def test_y_max(self):
        t, _, outcome, _ = check_termination(**_default_state(y=Y_MAX + 0.01))
        assert t and outcome == OUTCOME_CRASH_OOB

    def test_z_max(self):
        t, _, outcome, _ = check_termination(**_default_state(z=Z_MAX + 0.01))
        assert t and outcome == OUTCOME_CRASH_OOB

    def test_all_corners_safe(self):
        """Drone at the inside edge of every boundary should be ongoing."""
        t, _, outcome, _ = check_termination(**_default_state(
            x=X_MIN + 0.01, y=0.0, z=PAD_ELEVATION + 0.01))
        assert outcome == OUTCOME_ONGOING

        t, _, outcome, _ = check_termination(**_default_state(
            x=X_MAX - 0.01, y=Y_MAX - 0.01, z=Z_MAX - 0.01))
        assert outcome == OUTCOME_ONGOING


# ── Timeout ─────────────────────────────────────────────────────────

class TestTimeout:
    def test_at_max_steps(self):
        t, tr, outcome, r = check_termination(**_default_state(step=MAX_STEPS))
        assert not t and tr
        assert outcome == OUTCOME_TIMEOUT
        assert r == R_TIMEOUT

    def test_well_past_max_steps(self):
        """Guard against edge case where step exceeds MAX_STEPS."""
        _, tr, outcome, _ = check_termination(**_default_state(step=MAX_STEPS + 100))
        assert tr and outcome == OUTCOME_TIMEOUT


# ── Priority ordering ──────────────────────────────────────────────

class TestPriority:
    def test_attitude_beats_surface_contact(self):
        """Flipped drone with ToF < threshold → attitude crash, not contact."""
        t, _, outcome, _ = check_termination(**_default_state(
            roll=math.radians(50), z_tof=0.05, d_pad=0.0, vz=0.0))
        assert outcome == OUTCOME_CRASH_ATTITUDE

    def test_contact_beats_below_pad(self):
        """Surface contact at z just below PAD_ELEVATION → contact, not below_pad."""
        t, _, outcome, _ = check_termination(**_default_state(
            z=PAD_ELEVATION - 0.01, z_tof=0.05, d_pad=2.0))
        assert outcome == OUTCOME_CRASH_CONTACT

    def test_below_pad_beats_timeout(self):
        t, _, outcome, _ = check_termination(**_default_state(
            z=PAD_ELEVATION - 0.01, z_tof=PAD_ELEVATION - 0.01, step=MAX_STEPS))
        assert outcome == OUTCOME_CRASH_BELOW_PAD
