"""
Curriculum manager for staged training.

An SB3 callback that monitors the training logger's success rate and
orchestrates blended transitions between curriculum stages.  It controls
the spawner's stage setting — drone_env doesn't know curriculum exists.

Stage progression:  1 (close) → 2 (medium) → 3 (full cone)

Blended transitions:
    When success rate crosses the promotion threshold, the transition
    does not happen instantly.  Over ~200 episodes, the spawn distribution
    blends from the current stage to the next:

        80/20 → 60/40 → 40/60 → 20/80 → 0/100

    Each blend step lasts BLEND_EPISODES / BLEND_STEPS episodes.
    A Bernoulli draw each episode decides which stage the spawner uses.
    The blend schedule is linear and does not depend on performance
    during the transition.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_uav_package.config.constants import (
    CURRICULUM_STAGES,
    CURRICULUM_PROMOTION_THRESHOLD,
    CURRICULUM_WINDOW_SIZE,
    CURRICULUM_BLEND_EPISODES,
    CURRICULUM_BLEND_STEPS,
)
from rl_uav_package.envs.spawner import Spawner
from rl_uav_package.utils.training_logger import TrainingLogger


# Blend ratios: probability of sampling from the NEXT stage.
# 5 steps: 0.20, 0.40, 0.60, 0.80, 1.00
_BLEND_RATIOS = [
    (i + 1) / CURRICULUM_BLEND_STEPS
    for i in range(CURRICULUM_BLEND_STEPS)
]
# → [0.2, 0.4, 0.6, 0.8, 1.0]


class CurriculumManager(BaseCallback):
    """Monitors training progress and advances curriculum stages.

    Parameters
    ----------
    spawner : Spawner
        The spawner instance used by drone_env.  The manager calls
        set_stage() on it to control spawn distributions.
    training_logger : TrainingLogger
        The logging callback.  The manager reads its success_rate
        property to decide when to promote.
    seed : int
        Random seed for the blend Bernoulli draws.
    verbose : int
        0 = silent, 1 = stage transitions logged to console.
    """

    def __init__(
        self,
        spawner: Spawner,
        training_logger: TrainingLogger,
        seed: int = 0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._spawner = spawner
        self._logger = training_logger
        self._rng = np.random.default_rng(seed)

        # Current confirmed stage (fully transitioned to)
        self._current_stage: int = spawner.stage
        self._max_stage: int = max(CURRICULUM_STAGES.keys())

        # Blend state
        self._blending: bool = False
        self._blend_step: int = 0          # which ratio step we're on (0-indexed)
        self._blend_ep_counter: int = 0    # episodes elapsed in current blend step
        self._blend_eps_per_step: int = (
            CURRICULUM_BLEND_EPISODES // CURRICULUM_BLEND_STEPS
        )

        # Track episodes seen by this callback
        self._ep_count: int = 0

    # ── Public accessors ─────────────────────────────────────────

    @property
    def current_stage(self) -> int:
        """The stage the manager considers 'home' (fully promoted to)."""
        return self._current_stage

    @property
    def next_stage(self) -> int:
        """The stage being blended toward.  Same as current if not blending."""
        if self._blending and self._current_stage < self._max_stage:
            return self._current_stage + 1
        return self._current_stage

    @property
    def is_blending(self) -> bool:
        return self._blending

    @property
    def blend_ratio(self) -> float:
        """Current probability of sampling from the next stage.
        0.0 when not blending, progresses through _BLEND_RATIOS during blend."""
        if not self._blending:
            return 0.0
        return _BLEND_RATIOS[self._blend_step]

    # ── SB3 callback interface ───────────────────────────────────

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode_outcome" not in info:
                continue

            self._ep_count += 1
            self._on_episode_end()

        return True

    # ── Internal logic ───────────────────────────────────────────

    def _on_episode_end(self) -> None:
        """Check promotion conditions and advance blend state."""

        # Already at max stage, nothing to do
        if self._current_stage >= self._max_stage and not self._blending:
            return

        if self._blending:
            self._advance_blend()
        else:
            self._check_promotion()

        # Set the spawner stage for the next episode
        self._set_next_episode_stage()

    def _check_promotion(self) -> None:
        """Check if success rate warrants starting a blend transition."""
        # Need enough episodes for a meaningful signal
        if self._ep_count < CURRICULUM_WINDOW_SIZE:
            return

        # Already at max stage
        if self._current_stage >= self._max_stage:
            return

        if self._logger.success_rate >= CURRICULUM_PROMOTION_THRESHOLD:
            self._blending = True
            self._blend_step = 0
            self._blend_ep_counter = 0

            if self.verbose >= 1:
                next_s = self._current_stage + 1
                print(
                    f"\n[CURRICULUM] Stage {self._current_stage} → "
                    f"{next_s} transition started "
                    f"(success_rate={self._logger.success_rate:.2%}, "
                    f"ep={self._ep_count})"
                )

    def _advance_blend(self) -> None:
        """Advance the blend counter.  Move to next ratio step or finish."""
        self._blend_ep_counter += 1

        if self._blend_ep_counter >= self._blend_eps_per_step:
            # Move to next blend step
            self._blend_ep_counter = 0
            self._blend_step += 1

            if self._blend_step >= CURRICULUM_BLEND_STEPS:
                # Blend complete — promote to next stage
                self._current_stage += 1
                self._blending = False
                self._blend_step = 0

                if self.verbose >= 1:
                    print(
                        f"\n[CURRICULUM] Stage {self._current_stage} "
                        f"fully active (ep={self._ep_count})"
                    )
            else:
                if self.verbose >= 1:
                    ratio = _BLEND_RATIOS[self._blend_step]
                    print(
                        f"  [CURRICULUM] Blend step {self._blend_step + 1}/"
                        f"{CURRICULUM_BLEND_STEPS}: "
                        f"{(1 - ratio):.0%}/{ratio:.0%} "
                        f"(current/next)"
                    )

    def _set_next_episode_stage(self) -> None:
        """Decide which stage the spawner should use for the next episode."""
        if not self._blending:
            # Steady state — use current stage
            self._spawner.set_stage(self._current_stage)
            return

        # Blending — Bernoulli draw
        next_stage = self._current_stage + 1
        ratio = _BLEND_RATIOS[self._blend_step]

        if self._rng.random() < ratio:
            self._spawner.set_stage(next_stage)
        else:
            self._spawner.set_stage(self._current_stage)