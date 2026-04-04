"""
Curriculum manager for staged training.

SB3 callback that monitors success rate and orchestrates blended
transitions.  Records curriculum/stage to TensorBoard.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_uav_package.config.constants import (
    CURRICULUM_STAGES, CURRICULUM_PROMOTION_THRESHOLD,
    CURRICULUM_WINDOW_SIZE, CURRICULUM_BLEND_EPISODES,
    CURRICULUM_BLEND_STEPS,
)
from rl_uav_package.envs.spawner import Spawner
from rl_uav_package.utils.training_logger import TrainingLogger

_BLEND_RATIOS = [(i + 1) / CURRICULUM_BLEND_STEPS
                  for i in range(CURRICULUM_BLEND_STEPS)]


class CurriculumManager(BaseCallback):
    def __init__(self, spawner, training_logger, seed=0, verbose=1):
        super().__init__(verbose)
        self._spawner = spawner
        self._logger = training_logger
        self._rng = np.random.default_rng(seed)
        self._current_stage = spawner.stage
        self._max_stage = max(CURRICULUM_STAGES.keys())
        self._blending = False
        self._blend_step = 0
        self._blend_ep_counter = 0
        self._blend_eps_per_step = CURRICULUM_BLEND_EPISODES // CURRICULUM_BLEND_STEPS
        self._ep_count = 0
        self._cooldown_remaining = 0

    @property
    def current_stage(self):
        return self._current_stage

    @property
    def next_stage(self):
        if self._blending and self._current_stage < self._max_stage:
            return self._current_stage + 1
        return self._current_stage

    @property
    def is_blending(self):
        return self._blending

    @property
    def blend_ratio(self):
        return _BLEND_RATIOS[self._blend_step] if self._blending else 0.0

    def _on_step(self):
        # Record current stage to TensorBoard
        if self.logger is not None:
            self.logger.record("curriculum/stage", self._current_stage)

        for info in self.locals.get("infos", []):
            if "episode_outcome" in info:
                self._ep_count += 1
                self._on_episode_end()
        return True

    def _on_episode_end(self):
        if self._current_stage >= self._max_stage and not self._blending:
            return
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        if self._blending:
            self._advance_blend()
        else:
            self._check_promotion()
        self._set_next_episode_stage()

    def _check_promotion(self):
        if (self._ep_count < CURRICULUM_WINDOW_SIZE
                or self._current_stage >= self._max_stage
                or self._cooldown_remaining > 0):
            return
        stage_cfg = CURRICULUM_STAGES.get(self._current_stage, {})
        threshold = stage_cfg.get("promotion_threshold",
                                  CURRICULUM_PROMOTION_THRESHOLD)
        if self._logger.success_rate >= threshold:
            self._blending = True
            self._blend_step = 0
            self._blend_ep_counter = 0
            if self.verbose >= 1:
                ns = self._current_stage + 1
                print(f"\n[CURRICULUM] {self._current_stage} → {ns} started "
                      f"(sr={self._logger.success_rate:.2%}, "
                      f"threshold={threshold:.0%}, ep={self._ep_count})")

    def _advance_blend(self):
        self._blend_ep_counter += 1
        if self._blend_ep_counter >= self._blend_eps_per_step:
            self._blend_ep_counter = 0
            self._blend_step += 1
            if self._blend_step >= CURRICULUM_BLEND_STEPS:
                self._current_stage += 1
                self._blending = False
                self._blend_step = 0
                self._cooldown_remaining = CURRICULUM_WINDOW_SIZE
                if self.verbose >= 1:
                    cfg = CURRICULUM_STAGES.get(self._current_stage, {})
                    print(f"\n[CURRICULUM] Stage {self._current_stage} "
                          f"({cfg.get('name', '?')}) fully active "
                          f"(ep={self._ep_count})")
            elif self.verbose >= 1:
                r = _BLEND_RATIOS[self._blend_step]
                print(f"  [CURRICULUM] Blend {self._blend_step + 1}/"
                      f"{CURRICULUM_BLEND_STEPS}: {1-r:.0%}/{r:.0%}")

    def _set_next_episode_stage(self):
        if not self._blending:
            self._spawner.set_stage(self._current_stage)
            return
        ns = self._current_stage + 1
        r = _BLEND_RATIOS[self._blend_step]
        self._spawner.set_stage(ns if self._rng.random() < r else self._current_stage)