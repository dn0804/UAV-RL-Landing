"""
Unit tests for filters/ema.py

Run with:  python -m pytest test_ema.py -v
"""

import numpy as np
import pytest

from rl_uav_package.filters.ema import EMAFilter
from rl_uav_package.config.constants import EMA_ALPHA


class TestBasicOperation:
    def test_default_alpha(self):
        f = EMAFilter(n_channels=3)
        assert f.alpha == EMA_ALPHA

    def test_custom_alpha(self):
        f = EMAFilter(n_channels=1, alpha=0.7)
        assert f.alpha == 0.7

    def test_output_shape(self):
        f = EMAFilter(n_channels=3)
        out = f.update(np.array([1.0, 2.0, 3.0]))
        assert out.shape == (3,)
        assert out.dtype == np.float32

    def test_single_channel(self):
        f = EMAFilter(n_channels=1)
        out = f.update(np.array([5.0]))
        assert out.shape == (1,)


class TestReset:
    def test_reset_seeds_value(self):
        """After reset, the filter's value should equal the seed."""
        f = EMAFilter(n_channels=3)
        f.reset(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(f.value, [1.0, 2.0, 3.0])

    def test_reset_then_update_with_same_value(self):
        """Updating with the same value as the seed → no change."""
        f = EMAFilter(n_channels=2)
        f.reset(np.array([5.0, -3.0]))
        out = f.update(np.array([5.0, -3.0]))
        np.testing.assert_array_almost_equal(out, [5.0, -3.0])

    def test_reset_clears_prior_state(self):
        """Reset must overwrite any accumulated filter state."""
        f = EMAFilter(n_channels=1)
        # Drive the filter to a large value
        for _ in range(20):
            f.update(np.array([100.0]))
        # Reset to zero
        f.reset(np.array([0.0]))
        assert f.value[0] == pytest.approx(0.0)

    def test_no_reset_uses_first_reading(self):
        """Without an explicit reset, first update seeds the filter."""
        f = EMAFilter(n_channels=1)
        out = f.update(np.array([7.0]))
        assert out[0] == pytest.approx(7.0)


class TestEMAMath:
    def test_single_step(self):
        """v_new = α × raw + (1−α) × v_prev."""
        f = EMAFilter(n_channels=1, alpha=0.4)
        f.reset(np.array([10.0]))
        out = f.update(np.array([20.0]))
        expected = 0.4 * 20.0 + 0.6 * 10.0  # = 14.0
        assert out[0] == pytest.approx(expected)

    def test_two_steps(self):
        f = EMAFilter(n_channels=1, alpha=0.4)
        f.reset(np.array([0.0]))
        f.update(np.array([10.0]))     # → 4.0
        out = f.update(np.array([10.0]))  # → 0.4*10 + 0.6*4 = 6.4
        assert out[0] == pytest.approx(6.4)

    def test_convergence_to_constant(self):
        """Feeding a constant should converge the filter to that value."""
        f = EMAFilter(n_channels=1, alpha=0.4)
        f.reset(np.array([0.0]))
        for _ in range(50):
            out = f.update(np.array([1.0]))
        assert out[0] == pytest.approx(1.0, abs=1e-6)


class TestSpikeAttenuation:
    def test_single_spike_attenuated(self):
        """A single-step spike should be immediately damped."""
        f = EMAFilter(n_channels=1, alpha=0.4)
        f.reset(np.array([0.0]))

        # Steady state at 0, then one spike of 1.0, then back to 0
        f.update(np.array([0.0]))
        spike_response = f.update(np.array([1.0]))
        after_spike = f.update(np.array([0.0]))

        # Spike is attenuated to α = 0.4
        assert spike_response[0] == pytest.approx(0.4)
        # One step later it's further decayed
        assert after_spike[0] < spike_response[0]

    def test_spike_decays_within_3_steps(self):
        """Plan: 'a single-step 0.2 m/s spike is attenuated to 0.08 m/s
        immediately and vanishes by the third step.'"""
        f = EMAFilter(n_channels=1, alpha=0.4)
        f.reset(np.array([0.0]))

        # Spike
        out1 = f.update(np.array([0.2]))
        assert out1[0] == pytest.approx(0.08, abs=0.001)

        # Decay back toward 0
        out2 = f.update(np.array([0.0]))
        out3 = f.update(np.array([0.0]))

        # After 3 decay steps: 0.08 × (1−α)² = 0.0288
        # Over 85% of the original spike is gone.
        assert out3[0] < 0.03


class TestMultiChannel:
    def test_channels_independent(self):
        """Each channel should filter independently."""
        f = EMAFilter(n_channels=3, alpha=0.4)
        f.reset(np.array([0.0, 10.0, -5.0]))

        out = f.update(np.array([10.0, 10.0, -5.0]))

        # Channel 0: big jump → 0.4*10 + 0.6*0 = 4.0
        assert out[0] == pytest.approx(4.0)
        # Channel 1: no change → stays at 10.0
        assert out[1] == pytest.approx(10.0)
        # Channel 2: no change → stays at -5.0
        assert out[2] == pytest.approx(-5.0)


class TestValueProperty:
    def test_value_matches_last_update(self):
        f = EMAFilter(n_channels=2)
        f.reset(np.array([0.0, 0.0]))
        result = f.update(np.array([3.0, 4.0]))
        np.testing.assert_array_almost_equal(f.value, result)

    def test_value_dtype(self):
        f = EMAFilter(n_channels=1)
        f.reset(np.array([1.0]))
        assert f.value.dtype == np.float32
