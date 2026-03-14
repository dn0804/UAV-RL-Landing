"""
Unit tests for envs/rewards.py

Run with:  python -m pytest test_rewards.py -v
"""

import math
import numpy as np
import pytest

from rl_uav_package.envs.rewards import compute_reward, _sigmoid
from rl_uav_package.config.constants import (
    W_HORIZONTAL, W_DESCENT, W_YAW, W_JERK, W_TIME,
    DESCENT_GATE_H_MID, DESCENT_GATE_TAU, DESCENT_GATE_SIGMA,
    YAW_PENALTY_D_MIN,
    MARKER_Z,
    R_SUCCESS, R_CRASH,
)


# ── Helpers ──────────────────────────────────────────────────────────

ZERO_ACTION = np.zeros(4, dtype=np.float32)

def _base_call(**overrides):
    """Default call with zero deltas (hovering), overridden as needed."""
    kwargs = dict(
        d_pad=1.0, d_pad_prev=1.0,
        z=MARKER_Z + 0.5, z_prev=MARKER_Z + 0.5,
        h_above_marker=0.5,
        yaw_error=0.0, d_marker=1.0,
        action=ZERO_ACTION, prev_action=ZERO_ACTION,
    )
    kwargs.update(overrides)
    return compute_reward(**kwargs)


# ── Sigmoid helper ──────────────────────────────────────────────────

class TestSigmoid:
    def test_midpoint(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-6

    def test_large_positive(self):
        assert abs(_sigmoid(50.0) - 1.0) < 1e-6

    def test_large_negative(self):
        assert abs(_sigmoid(-50.0) - 0.0) < 1e-6

    def test_symmetry(self):
        assert abs(_sigmoid(2.0) + _sigmoid(-2.0) - 1.0) < 1e-6


# ── Horizontal centering  (§8.3) ───────────────────────────────────

class TestHorizontal:
    def test_approaching(self):
        """Distance decreasing → positive reward."""
        _, bd = _base_call(d_pad=1.0, d_pad_prev=1.05)
        assert bd["reward/horizontal"] == pytest.approx(W_HORIZONTAL * 0.05)

    def test_receding(self):
        """Distance increasing → negative reward."""
        _, bd = _base_call(d_pad=1.05, d_pad_prev=1.0)
        assert bd["reward/horizontal"] == pytest.approx(-W_HORIZONTAL * 0.05)

    def test_stationary(self):
        """No movement → zero horizontal reward."""
        _, bd = _base_call(d_pad=1.0, d_pad_prev=1.0)
        assert bd["reward/horizontal"] == pytest.approx(0.0)

    def test_typical_magnitude(self):
        """§8.3: At 0.5 m/s approach (0.05m/step at 10Hz) → ~0.5 per step."""
        _, bd = _base_call(d_pad=1.0, d_pad_prev=1.05)
        assert bd["reward/horizontal"] == pytest.approx(0.5, abs=0.01)


# ── Gated descent  (§8.4) ──────────────────────────────────────────

class TestDescent:
    def test_descending_high_altitude_free(self):
        """Well above h_mid → gate ≈ 0, centering_gate ≈ 1.
        Descent rewarded regardless of horizontal offset."""
        _, bd = _base_call(
            z=MARKER_Z + 2.0, z_prev=MARKER_Z + 2.03,
            h_above_marker=2.0,
            d_pad=3.0)  # far from pad
        assert bd["diag/gate_blend"] < 0.01       # gate inactive
        assert bd["diag/centering_gate"] > 0.99    # free descent
        assert bd["reward/descent"] > 0.0          # positive reward

    def test_descending_low_altitude_centered(self):
        """Well below h_mid, perfectly centered → gate ≈ 1, Gaussian ≈ 1.
        Full descent reward."""
        _, bd = _base_call(
            z=MARKER_Z + 0.1, z_prev=MARKER_Z + 0.13,
            h_above_marker=0.1,
            d_pad=0.0)  # perfectly centered
        assert bd["diag/gate_blend"] > 0.99
        assert bd["diag/centering_gate"] > 0.99
        expected = W_DESCENT * 0.03 * bd["diag/centering_gate"]
        assert bd["reward/descent"] == pytest.approx(expected, rel=0.01)

    def test_descending_low_altitude_off_center(self):
        """Well below h_mid, far from center → gate ≈ 1, Gaussian ≈ 0.
        Descent reward nearly zero (can't descend when off-center near pad)."""
        _, bd = _base_call(
            z=MARKER_Z + 0.1, z_prev=MARKER_Z + 0.13,
            h_above_marker=0.1,
            d_pad=0.5)  # far off center
        assert bd["diag/gate_blend"] > 0.99
        assert bd["diag/centering_gate"] < 0.01    # Gaussian kills it
        assert bd["reward/descent"] < 0.01          # nearly zero

    def test_ascending_always_penalized(self):
        """Ascending → negative reward, ungated by position."""
        _, bd_centered = _base_call(
            z=MARKER_Z + 0.53, z_prev=MARKER_Z + 0.5,
            h_above_marker=0.53,
            d_pad=0.0)  # centered
        _, bd_offset = _base_call(
            z=MARKER_Z + 0.53, z_prev=MARKER_Z + 0.5,
            h_above_marker=0.53,
            d_pad=3.0)  # far away

        # Both should be equally negative (ungated)
        assert bd_centered["reward/descent"] < 0.0
        assert bd_centered["reward/descent"] == pytest.approx(
            bd_offset["reward/descent"])

    def test_ascending_magnitude(self):
        """Ascending 0.03m → R = 20 × (−0.03) = −0.6."""
        _, bd = _base_call(
            z=MARKER_Z + 0.53, z_prev=MARKER_Z + 0.5,
            h_above_marker=0.53)
        assert bd["reward/descent"] == pytest.approx(W_DESCENT * (-0.03))

    def test_gate_at_sigmoid_midpoint(self):
        """h == h_mid → gate_blend ≈ 0.5."""
        _, bd = _base_call(h_above_marker=DESCENT_GATE_H_MID)
        assert bd["diag/gate_blend"] == pytest.approx(0.5, abs=0.01)


class TestGaussianGate:
    def test_at_center(self):
        """d_pad = 0 → Gaussian = 1."""
        gauss = math.exp(0)
        assert gauss == 1.0

    def test_at_pad_edge(self):
        """d_pad = σ = 0.10 → Gaussian = exp(−0.5) ≈ 0.607.
        Plan §8.4: 'reward at the zone edge is 61%'."""
        gauss = math.exp(-(0.10 ** 2) / (2 * DESCENT_GATE_SIGMA ** 2))
        assert gauss == pytest.approx(0.6065, abs=0.001)

    def test_at_double_sigma(self):
        """d_pad = 2σ = 0.20 → Gaussian = exp(−2) ≈ 0.135.
        Plan §8.4: 'about 14% at 0.20 m'."""
        gauss = math.exp(-(0.20 ** 2) / (2 * DESCENT_GATE_SIGMA ** 2))
        assert gauss == pytest.approx(0.1353, abs=0.001)

    def test_at_three_sigma(self):
        """d_pad = 3σ = 0.30 → Gaussian ≈ 0.011.
        Plan §8.4: 'effectively zero beyond 0.30 m'."""
        gauss = math.exp(-(0.30 ** 2) / (2 * DESCENT_GATE_SIGMA ** 2))
        assert gauss < 0.02


# ── Yaw-to-bearing penalty  (§8.5) ─────────────────────────────────

class TestYaw:
    def test_zero_error(self):
        """Perfect alignment → zero yaw penalty."""
        _, bd = _base_call(yaw_error=0.0)
        assert bd["reward/yaw"] == pytest.approx(0.0)

    def test_worst_case_close(self):
        """§8.5: At 0.3m, worst-case (π error) → −0.3."""
        _, bd = _base_call(yaw_error=math.pi, d_marker=0.3)
        assert bd["reward/yaw"] == pytest.approx(-W_YAW, abs=0.001)

    def test_worst_case_far(self):
        """§8.5: At 4.0m, worst-case → −0.0225."""
        _, bd = _base_call(yaw_error=math.pi, d_marker=4.0)
        expected = -W_YAW * 1.0 * (YAW_PENALTY_D_MIN / 4.0)
        assert bd["reward/yaw"] == pytest.approx(expected, abs=0.001)

    def test_closer_means_stronger(self):
        """Penalty magnitude increases as distance decreases."""
        _, bd_close = _base_call(yaw_error=0.5, d_marker=0.5)
        _, bd_far = _base_call(yaw_error=0.5, d_marker=3.0)
        assert abs(bd_close["reward/yaw"]) > abs(bd_far["reward/yaw"])

    def test_capped_below_d_min(self):
        """Distances below YAW_PENALTY_D_MIN should clamp (not explode)."""
        _, bd_at = _base_call(yaw_error=1.0, d_marker=YAW_PENALTY_D_MIN)
        _, bd_inside = _base_call(yaw_error=1.0, d_marker=0.05)
        assert bd_at["reward/yaw"] == pytest.approx(bd_inside["reward/yaw"])

    def test_negative_yaw_error(self):
        """Penalty uses abs(yaw_error), so sign doesn't matter."""
        _, bd_pos = _base_call(yaw_error=0.5)
        _, bd_neg = _base_call(yaw_error=-0.5)
        assert bd_pos["reward/yaw"] == pytest.approx(bd_neg["reward/yaw"])


# ── Jerk penalty  (§8.6) ───────────────────────────────────────────

class TestJerk:
    def test_no_change(self):
        """Identical consecutive actions → zero jerk."""
        _, bd = _base_call(action=ZERO_ACTION, prev_action=ZERO_ACTION)
        assert bd["reward/jerk"] == pytest.approx(0.0)

    def test_full_reversal_one_axis(self):
        """Single axis reversal −1 → +1: ‖Δa‖ = 2.0, penalty = −0.2."""
        a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, bd = _base_call(action=a, prev_action=b)
        assert bd["reward/jerk"] == pytest.approx(-W_JERK * 2.0)

    def test_full_reversal_all_axes(self):
        """§8.6: theoretical max ‖Δa‖ = √(4×4) = 4.0, penalty = −0.4."""
        a = np.ones(4, dtype=np.float32)
        b = -np.ones(4, dtype=np.float32)
        _, bd = _base_call(action=a, prev_action=b)
        assert bd["reward/jerk"] == pytest.approx(-W_JERK * 4.0)


# ── Time penalty  (§8.6) ───────────────────────────────────────────

class TestTime:
    def test_constant(self):
        """Time penalty is always −0.1 regardless of state."""
        _, bd = _base_call()
        assert bd["reward/time"] == pytest.approx(-W_TIME)


# ── Hovering scenario  (§8.7) ──────────────────────────────────────

class TestHovering:
    def test_hovering_earns_only_time_penalty(self):
        """§8.7: A hovering drone (all deltas zero, zero yaw error,
        same action) earns exactly −W_TIME per step. No positive
        shaping reward is possible from a stationary state."""
        total, bd = _base_call()
        assert bd["reward/horizontal"] == pytest.approx(0.0)
        assert bd["reward/descent"] == pytest.approx(0.0)
        assert bd["reward/yaw"] == pytest.approx(0.0)
        assert bd["reward/jerk"] == pytest.approx(0.0)
        assert total == pytest.approx(-W_TIME)


# ── Plan sanity check table  (§8.7) ────────────────────────────────

class TestPlanSanityCheck:
    """Verify the ordering from §8.7:
       Clean landing (+155) >> hovering (−30) >> immediate crash (−100).
    """

    def test_hovering_beats_crashing(self):
        """300 steps of hovering = −30, which is better than −100 crash."""
        cumulative_hover = -W_TIME * 300
        assert cumulative_hover > R_CRASH

    def test_landing_beats_hovering(self):
        """Rough estimate of a clean 150-step landing from 4m.
        Plan says ~+155. We verify it's positive and >> 0."""
        # Approximate: 4m approach at 0.05m/step for ~80 steps
        r_horiz = W_HORIZONTAL * 4.0       # total horizontal progress
        # ~1.0m descent at 0.03m/step for ~33 steps, gate ≈ 0.8 average
        r_desc = W_DESCENT * 1.0 * 0.8
        r_time = -W_TIME * 150
        r_success = R_SUCCESS
        rough_total = r_horiz + r_desc + r_time + r_success
        assert rough_total > 0
        assert rough_total > (-W_TIME * 300)  # better than hovering


# ── Breakdown dict completeness ─────────────────────────────────────

class TestBreakdown:
    def test_all_keys_present(self):
        _, bd = _base_call()
        expected_keys = {
            "reward/horizontal", "reward/descent", "reward/yaw",
            "reward/jerk", "reward/time", "reward/total",
            "diag/d_pad", "diag/h_above_marker",
            "diag/centering_gate", "diag/gate_blend", "diag/delta_z",
        }
        assert set(bd.keys()) == expected_keys

    def test_total_equals_sum(self):
        """The 'total' key must equal the sum of the five components."""
        total, bd = _base_call(
            d_pad=0.5, d_pad_prev=0.55,
            z=MARKER_Z + 0.5, z_prev=MARKER_Z + 0.53,
            h_above_marker=0.5,
            yaw_error=0.3, d_marker=1.0,
            action=np.array([0.5, 0.0, -0.3, 0.1], dtype=np.float32),
            prev_action=np.array([0.3, 0.1, -0.1, 0.0], dtype=np.float32),
        )
        components_sum = (
            bd["reward/horizontal"] + bd["reward/descent"] + bd["reward/yaw"]
            + bd["reward/jerk"] + bd["reward/time"]
        )
        assert total == pytest.approx(components_sum)
        assert bd["reward/total"] == pytest.approx(total)
