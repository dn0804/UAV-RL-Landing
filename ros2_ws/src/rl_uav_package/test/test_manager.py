"""
Unit tests for curriculum/manager.py

Tests the curriculum manager's promotion logic, blend transitions,
Bernoulli stage sampling, and edge cases, using mocked dependencies.

Run with:  python -m pytest test_manager.py -v
"""

import math
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from rl_uav_package.curriculum.manager import (
    CurriculumManager,
    _BLEND_RATIOS,
)
from rl_uav_package.config.constants import (
    CURRICULUM_PROMOTION_THRESHOLD,
    CURRICULUM_WINDOW_SIZE,
    CURRICULUM_BLEND_EPISODES,
    CURRICULUM_BLEND_STEPS,
)
from rl_uav_package.envs.termination import OUTCOME_SUCCESS, OUTCOME_CRASH_OOB


# ── Helpers ──────────────────────────────────────────────────────────

def _make_manager(
    initial_stage=1,
    success_rate=0.0,
    seed=42,
    verbose=0,
):
    """Create a CurriculumManager with mocked spawner and logger."""
    spawner = MagicMock()
    spawner.stage = initial_stage

    logger = MagicMock()
    type(logger).success_rate = PropertyMock(return_value=success_rate)

    cm = CurriculumManager(
        spawner=spawner,
        training_logger=logger,
        seed=seed,
        verbose=verbose,
    )
    # SB3 sets these during training
    cm.locals = {"infos": []}
    cm.model = MagicMock()
    cm.logger = MagicMock()

    return cm, spawner, logger


def _push_episodes(cm, logger_mock, n, success_rate=None):
    """Simulate n episode completions through the callback.

    If success_rate is provided, update the mock's return value.
    """
    if success_rate is not None:
        type(logger_mock).success_rate = PropertyMock(return_value=success_rate)

    for _ in range(n):
        cm.locals = {"infos": [{"episode_outcome": OUTCOME_SUCCESS}]}
        cm._on_step()


# ── Blend ratios ────────────────────────────────────────────────────

class TestBlendRatios:
    def test_ratio_count(self):
        assert len(_BLEND_RATIOS) == CURRICULUM_BLEND_STEPS

    def test_ratios_ascending(self):
        for i in range(len(_BLEND_RATIOS) - 1):
            assert _BLEND_RATIOS[i] < _BLEND_RATIOS[i + 1]

    def test_first_ratio(self):
        """First blend step: 20% chance of next stage."""
        assert _BLEND_RATIOS[0] == pytest.approx(0.2)

    def test_last_ratio(self):
        """Last blend step: 100% chance of next stage."""
        assert _BLEND_RATIOS[-1] == pytest.approx(1.0)


# ── Initial state ───────────────────────────────────────────────────

class TestInitialState:
    def test_starts_at_given_stage(self):
        cm, _, _ = _make_manager(initial_stage=2)
        assert cm.current_stage == 2

    def test_not_blending_initially(self):
        cm, _, _ = _make_manager()
        assert not cm.is_blending
        assert cm.blend_ratio == 0.0

    def test_next_stage_equals_current_when_not_blending(self):
        cm, _, _ = _make_manager(initial_stage=1)
        assert cm.next_stage == 1


# ── Promotion trigger ──────────────────────────────────────────────

class TestPromotion:
    def test_no_promotion_below_threshold(self):
        cm, spawner, logger = _make_manager(success_rate=0.69)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE + 10)
        assert not cm.is_blending
        assert cm.current_stage == 1

    def test_promotion_at_threshold(self):
        cm, spawner, logger = _make_manager(success_rate=0.70)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE + 1)
        assert cm.is_blending

    def test_promotion_above_threshold(self):
        cm, spawner, logger = _make_manager(success_rate=0.85)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE + 1)
        assert cm.is_blending

    def test_no_promotion_before_window_fills(self):
        """Even with 100% success, wait for enough episodes."""
        cm, spawner, logger = _make_manager(success_rate=1.0)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE - 1)
        assert not cm.is_blending

    def test_promotion_exactly_at_window_size(self):
        cm, spawner, logger = _make_manager(success_rate=1.0)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE)
        assert cm.is_blending


# ── Blend progression ──────────────────────────────────────────────

class TestBlendProgression:
    def test_blend_step_advances(self):
        cm, spawner, logger = _make_manager(success_rate=0.80)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE + 1)  # triggers blend
        assert cm.is_blending
        assert cm._blend_step == 0

        eps_per_step = CURRICULUM_BLEND_EPISODES // CURRICULUM_BLEND_STEPS
        _push_episodes(cm, logger, eps_per_step)
        assert cm._blend_step == 1

    def test_full_blend_promotes(self):
        cm, spawner, logger = _make_manager(success_rate=0.80)
        # Blend triggers on episode WINDOW (the first that passes the check)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE)      # triggers blend
        assert cm.is_blending
        _push_episodes(cm, logger, CURRICULUM_BLEND_EPISODES)   # complete blend

        assert not cm.is_blending
        assert cm.current_stage == 2

    def test_blend_ratio_progresses(self):
        cm, spawner, logger = _make_manager(success_rate=0.80)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE)

        eps_per_step = CURRICULUM_BLEND_EPISODES // CURRICULUM_BLEND_STEPS

        for step_idx in range(CURRICULUM_BLEND_STEPS):
            expected_ratio = _BLEND_RATIOS[step_idx]
            assert cm.blend_ratio == pytest.approx(expected_ratio), (
                f"Step {step_idx}: expected {expected_ratio}, got {cm.blend_ratio}"
            )
            _push_episodes(cm, logger, eps_per_step)

        # After all steps complete, blend is done
        assert not cm.is_blending
        assert cm.blend_ratio == 0.0


# ── Multi-stage progression ────────────────────────────────────────

class TestMultiStage:
    def test_stage_1_to_3(self):
        """Full journey through all three stages."""
        cm, spawner, logger = _make_manager(success_rate=0.80)

        # Stage 1 → 2
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE)
        assert cm.is_blending and cm.current_stage == 1

        # Drop success rate during blend so stage 2→3 doesn't auto-trigger
        _push_episodes(cm, logger, CURRICULUM_BLEND_EPISODES, success_rate=0.30)
        assert cm.current_stage == 2 and not cm.is_blending

        # Raise success rate again for stage 2 → 3
        _push_episodes(cm, logger, 1, success_rate=0.80)  # trigger
        assert cm.is_blending and cm.current_stage == 2
        _push_episodes(cm, logger, CURRICULUM_BLEND_EPISODES)
        assert cm.current_stage == 3 and not cm.is_blending

    def test_no_promotion_past_max_stage(self):
        cm, spawner, logger = _make_manager(initial_stage=3, success_rate=1.0)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE + 100)
        assert cm.current_stage == 3
        assert not cm.is_blending


# ── Spawner interaction ────────────────────────────────────────────

class TestSpawnerInteraction:
    def test_spawner_set_to_current_when_not_blending(self):
        cm, spawner, logger = _make_manager(success_rate=0.50)
        _push_episodes(cm, logger, 5)
        # Every episode should have set the spawner to stage 1
        for call in spawner.set_stage.call_args_list:
            assert call.args[0] == 1

    def test_spawner_gets_mixed_stages_during_blend(self):
        """During blending, the spawner should receive both current and next stage."""
        cm, spawner, logger = _make_manager(success_rate=0.80, seed=42)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE + 1)  # start blend

        # Run enough episodes through the blend that we expect both stages
        spawner.set_stage.reset_mock()
        _push_episodes(cm, logger, 100)

        stages_set = [call.args[0] for call in spawner.set_stage.call_args_list]
        assert 1 in stages_set, "Current stage never sampled during blend"
        assert 2 in stages_set, "Next stage never sampled during blend"

    def test_spawner_locked_after_blend(self):
        cm, spawner, logger = _make_manager(success_rate=0.80)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE)
        # Drop success rate so completing blend doesn't trigger 2→3
        _push_episodes(cm, logger, CURRICULUM_BLEND_EPISODES, success_rate=0.30)

        # Now push more episodes at low success (no new promotion)
        spawner.set_stage.reset_mock()
        _push_episodes(cm, logger, 10, success_rate=0.3)

        for call in spawner.set_stage.call_args_list:
            assert call.args[0] == 2


# ── Bernoulli sampling statistics ──────────────────────────────────

class TestBernoulliSampling:
    def test_blend_ratio_matches_sampling(self):
        """Over many episodes, the fraction of next-stage samples should
        approximate the blend ratio."""
        cm, spawner, logger = _make_manager(success_rate=0.80, seed=0)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE + 1)

        # Sit in blend step 0 (ratio = 0.2) for many episodes
        spawner.set_stage.reset_mock()
        eps_per_step = CURRICULUM_BLEND_EPISODES // CURRICULUM_BLEND_STEPS
        n_samples = min(eps_per_step - 1, 50)  # don't overflow into next step
        _push_episodes(cm, logger, n_samples)

        stages = [call.args[0] for call in spawner.set_stage.call_args_list]
        frac_next = sum(1 for s in stages if s == 2) / len(stages)

        # With ratio 0.2 and 50 samples, we expect ~0.2 ± 0.1
        assert 0.05 < frac_next < 0.45, (
            f"Expected ~20% next-stage, got {frac_next:.1%}"
        )


# ── next_stage property ────────────────────────────────────────────

class TestNextStage:
    def test_during_blend(self):
        cm, _, logger = _make_manager(success_rate=0.80)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE + 1)
        assert cm.next_stage == 2

    def test_at_max_stage(self):
        cm, _, _ = _make_manager(initial_stage=3)
        assert cm.next_stage == 3

    def test_not_blending(self):
        cm, _, _ = _make_manager(initial_stage=2)
        assert cm.next_stage == 2


# ── Edge cases ─────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_episode_info_ignored(self):
        """Steps without episode_outcome should be silently ignored."""
        cm, spawner, _ = _make_manager()
        cm.locals = {"infos": [{"reward/total": 0.5}]}
        cm._on_step()
        assert cm._ep_count == 0

    def test_empty_infos(self):
        cm, _, _ = _make_manager()
        cm.locals = {"infos": []}
        cm._on_step()
        assert cm._ep_count == 0

    def test_blend_not_interrupted_by_low_success(self):
        """Once a blend starts, it runs to completion regardless of
        performance during the transition."""
        cm, _, logger = _make_manager(success_rate=0.80)
        _push_episodes(cm, logger, CURRICULUM_WINDOW_SIZE)
        assert cm.is_blending

        # Drop success rate during blend
        _push_episodes(cm, logger, CURRICULUM_BLEND_EPISODES, success_rate=0.10)

        # Blend should still complete
        assert not cm.is_blending
        assert cm.current_stage == 2
