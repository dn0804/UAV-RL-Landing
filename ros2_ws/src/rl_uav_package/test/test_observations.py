"""
Unit tests for envs/observations.py

Run with:  python -m pytest test_observations.py -v
"""

import math
import numpy as np
import pytest

from rl_uav_package.envs.observations import ObservationBuilder
from rl_uav_package.config.constants import (
    OBS_DIM, ACTION_DIM, ACTION_HISTORY_STEPS,
    OBS_SCALE, OBS_CLIP, DROPOUT_TIMER_CAP,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _default_build(builder, **overrides):
    """Call build() with safe defaults, overriding as needed."""
    kwargs = dict(
        x=1.0, y=0.0, z=1.0, yaw=0.0,
        vx=0.0, vy=0.0, vz=0.0,
        z_tof_raw=1.0,
        roll=0.0, pitch=0.0,
        dropout_timer=0,
    )
    kwargs.update(overrides)
    return builder.build(**kwargs)


# ── Shape and dtype ─────────────────────────────────────────────────

class TestOutputFormat:
    def test_shape(self):
        b = ObservationBuilder()
        obs = _default_build(b)
        assert obs.shape == (OBS_DIM,)

    def test_dtype(self):
        b = ObservationBuilder()
        obs = _default_build(b)
        assert obs.dtype == np.float32

    def test_dim_constant_is_29(self):
        assert OBS_DIM == 29


# ── Channel ordering ────────────────────────────────────────────────

class TestChannelOrder:
    """Verify values land in the expected indices before normalization.
    We can check by using known inputs where normalization is predictable."""

    def test_sensor_channels(self):
        """Channels 0-8 are sensor data."""
        b = ObservationBuilder()
        obs = b.build(
            x=4.0, y=3.5, z=2.5, yaw=math.pi,
            vx=2.0, vy=2.0, vz=2.0,
            z_tof_raw=2.5,      # no tilt → no cosine correction
            roll=0.0, pitch=0.0,
            dropout_timer=30,
        )
        # Each value equals its scale factor → normalized to 1.0
        for i in range(9):
            assert obs[i] == pytest.approx(1.0, abs=0.001), (
                f"Channel {i}: expected 1.0, got {obs[i]}"
            )

    def test_action_history_channels(self):
        """Channels 9-28 are action history (scale factor = 1.0)."""
        b = ObservationBuilder()
        # Push a distinctive action
        action = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
        b.push_action(action)
        obs = _default_build(b)

        # Most recent action → channels 9, 10, 11, 12
        assert obs[9] == pytest.approx(0.5)
        assert obs[10] == pytest.approx(-0.5)
        assert obs[11] == pytest.approx(0.25)
        assert obs[12] == pytest.approx(-0.25)

    def test_action_history_newest_first(self):
        """The most recent action occupies channels 9-12, oldest at 25-28."""
        b = ObservationBuilder()

        old_action = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        new_action = np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32)

        # Push 5 old actions, then 1 new one
        for _ in range(ACTION_HISTORY_STEPS):
            b.push_action(old_action)
        b.push_action(new_action)  # this pushes out the oldest

        obs = _default_build(b)

        # Newest (new_action) at channels 9-12
        for i in range(4):
            assert obs[9 + i] == pytest.approx(0.9, abs=0.01)

        # Oldest remaining (old_action) at channels 25-28
        for i in range(4):
            assert obs[25 + i] == pytest.approx(0.1, abs=0.01)


# ── ToF cosine correction ──────────────────────────────────────────

class TestToFCorrection:
    def test_level_flight_no_correction(self):
        """Roll = pitch = 0 → cos(0) = 1 → no correction."""
        b = ObservationBuilder()
        obs = b.build(
            x=0, y=0, z=0, yaw=0,
            vx=0, vy=0, vz=0,
            z_tof_raw=1.0,
            roll=0.0, pitch=0.0,
            dropout_timer=0,
        )
        # Channel 7 = z_tof, normalized by 2.5
        assert obs[7] == pytest.approx(1.0 / 2.5)

    def test_pitch_reduces_tof(self):
        """Pitched drone → ToF reads longer than true vertical distance.
        Correction: z_tof = raw × cos(pitch) × cos(roll)."""
        b = ObservationBuilder()
        pitch = math.radians(20)
        raw_tof = 1.0
        expected_corrected = raw_tof * math.cos(pitch)  # roll = 0

        obs = b.build(
            x=0, y=0, z=0, yaw=0,
            vx=0, vy=0, vz=0,
            z_tof_raw=raw_tof,
            roll=0.0, pitch=pitch,
            dropout_timer=0,
        )
        assert obs[7] == pytest.approx(expected_corrected / 2.5)

    def test_roll_reduces_tof(self):
        b = ObservationBuilder()
        roll = math.radians(15)
        raw_tof = 0.5
        expected = raw_tof * math.cos(roll)

        obs = b.build(
            x=0, y=0, z=0, yaw=0,
            vx=0, vy=0, vz=0,
            z_tof_raw=raw_tof,
            roll=roll, pitch=0.0,
            dropout_timer=0,
        )
        assert obs[7] == pytest.approx(expected / 2.5)

    def test_combined_pitch_and_roll(self):
        b = ObservationBuilder()
        roll = math.radians(10)
        pitch = math.radians(20)
        raw_tof = 0.4
        expected = raw_tof * math.cos(pitch) * math.cos(roll)

        obs = b.build(
            x=0, y=0, z=0, yaw=0,
            vx=0, vy=0, vz=0,
            z_tof_raw=raw_tof,
            roll=roll, pitch=pitch,
            dropout_timer=0,
        )
        assert obs[7] == pytest.approx(expected / 2.5, abs=1e-5)


# ── Normalization ───────────────────────────────────────────────────

class TestNormalization:
    def test_values_at_scale_factor_become_one(self):
        """When raw value == scale factor, normalized value == 1.0."""
        b = ObservationBuilder()
        obs = b.build(
            x=4.0, y=3.5, z=2.5, yaw=math.pi,
            vx=2.0, vy=2.0, vz=2.0,
            z_tof_raw=2.5,
            roll=0.0, pitch=0.0,
            dropout_timer=30,
        )
        for i in range(9):
            assert obs[i] == pytest.approx(1.0, abs=0.001)

    def test_zero_stays_zero(self):
        """Zero input → zero output (normalization is division, not shift)."""
        b = ObservationBuilder()
        obs = b.build(
            x=0, y=0, z=0, yaw=0,
            vx=0, vy=0, vz=0,
            z_tof_raw=0,
            roll=0, pitch=0,
            dropout_timer=0,
        )
        for i in range(9):
            assert obs[i] == pytest.approx(0.0)


# ── Clipping ────────────────────────────────────────────────────────

class TestClipping:
    def test_extreme_value_clipped(self):
        """A physics-glitch value (50 m/s) should be clipped to ±1.5."""
        b = ObservationBuilder()
        obs = b.build(
            x=0, y=0, z=0, yaw=0,
            vx=50.0, vy=0, vz=0,   # absurd velocity
            z_tof_raw=0,
            roll=0, pitch=0,
            dropout_timer=0,
        )
        # vx at channel 4, normalized = 50/2 = 25 → clipped to 1.5
        assert obs[4] == pytest.approx(OBS_CLIP)

    def test_negative_extreme_clipped(self):
        b = ObservationBuilder()
        obs = b.build(
            x=0, y=-50.0, z=0, yaw=0,
            vx=0, vy=0, vz=0,
            z_tof_raw=0,
            roll=0, pitch=0,
            dropout_timer=0,
        )
        # y at channel 1, normalized = -50/3.5 = -14.3 → clipped to -1.5
        assert obs[1] == pytest.approx(-OBS_CLIP)

    def test_normal_values_not_clipped(self):
        """Values within [-1.5, 1.5] after normalization should pass through."""
        b = ObservationBuilder()
        obs = b.build(
            x=1.0, y=0.5, z=0.5, yaw=0.3,
            vx=0.5, vy=-0.3, vz=0.1,
            z_tof_raw=0.8,
            roll=0, pitch=0,
            dropout_timer=5,
        )
        # All these produce normalized values well within ±1.5
        for i in range(9):
            assert -OBS_CLIP <= obs[i] <= OBS_CLIP


# ── Dropout timer ───────────────────────────────────────────────────

class TestDropoutTimer:
    def test_zero_when_visible(self):
        b = ObservationBuilder()
        obs = _default_build(b, dropout_timer=0)
        assert obs[8] == pytest.approx(0.0)

    def test_capped_at_max(self):
        """Timer values above the cap should be clamped before normalization."""
        b = ObservationBuilder()
        obs1 = _default_build(b, dropout_timer=DROPOUT_TIMER_CAP)
        obs2 = _default_build(b, dropout_timer=DROPOUT_TIMER_CAP + 100)
        # Both should produce the same normalized value
        assert obs1[8] == pytest.approx(obs2[8])
        # And that value should be 30/30 = 1.0
        assert obs1[8] == pytest.approx(1.0)


# ── Action history buffer ───────────────────────────────────────────

class TestActionHistory:
    def test_initial_history_is_zeros(self):
        """After reset, all 20 action history channels should be zero."""
        b = ObservationBuilder()
        obs = _default_build(b)
        for i in range(9, OBS_DIM):
            assert obs[i] == pytest.approx(0.0)

    def test_push_fills_newest_slot(self):
        b = ObservationBuilder()
        b.push_action(np.array([0.7, 0.0, 0.0, 0.0]))
        obs = _default_build(b)
        # Newest action → channel 9
        assert obs[9] == pytest.approx(0.7)
        # Next slot (t-2) should still be zero
        assert obs[13] == pytest.approx(0.0)

    def test_five_pushes_fill_all_slots(self):
        b = ObservationBuilder()
        for i in range(1, ACTION_HISTORY_STEPS + 1):
            b.push_action(np.full(ACTION_DIM, float(i) * 0.1, dtype=np.float32))

        obs = _default_build(b)

        # Newest (pushed last, value 0.5) → channels 9-12
        assert obs[9] == pytest.approx(0.5)
        # Oldest (pushed first, value 0.1) → channels 25-28
        assert obs[25] == pytest.approx(0.1)

    def test_sixth_push_evicts_oldest(self):
        """Ring buffer has capacity 5. Sixth push drops the first."""
        b = ObservationBuilder()
        for i in range(1, ACTION_HISTORY_STEPS + 1):
            b.push_action(np.full(ACTION_DIM, float(i) * 0.1, dtype=np.float32))

        # Push a 6th → evicts the one with value 0.1
        b.push_action(np.full(ACTION_DIM, 0.6, dtype=np.float32))
        obs = _default_build(b)

        # Newest (0.6) at channels 9-12
        assert obs[9] == pytest.approx(0.6)
        # Oldest should now be 0.2 (the second push), at channels 25-28
        assert obs[25] == pytest.approx(0.2)

    def test_reset_clears_history(self):
        b = ObservationBuilder()
        b.push_action(np.ones(ACTION_DIM))
        b.reset()
        obs = _default_build(b)
        for i in range(9, OBS_DIM):
            assert obs[i] == pytest.approx(0.0)


# ── Integration: build does not mutate inputs ───────────────────────

class TestImmutability:
    def test_action_not_mutated_by_push(self):
        b = ObservationBuilder()
        action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        original = action.copy()
        b.push_action(action)
        np.testing.assert_array_equal(action, original)
