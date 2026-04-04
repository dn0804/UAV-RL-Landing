"""
Curriculum manager for MPI parallel training.

Standalone version of CurriculumManager that doesn't inherit from
SB3's BaseCallback.  Rank 0 owns an instance and calls
process_outcomes() each update.  The broadcast state tuple is
sent to all workers so they can set their Spawner correctly.

Also contains apply_curriculum_to_spawner() which workers call
during rollout collection on episode resets.
"""

import numpy as np

from rl_uav_package.config.constants import (
    CURRICULUM_STAGES, CURRICULUM_PROMOTION_THRESHOLD,
    CURRICULUM_WINDOW_SIZE, CURRICULUM_BLEND_EPISODES,
    CURRICULUM_BLEND_STEPS,
)

_BLEND_RATIOS = [(i + 1) / CURRICULUM_BLEND_STEPS
                 for i in range(CURRICULUM_BLEND_STEPS)]


class MPICurriculumManager:
    """Curriculum manager for MPI training (Rank 0 only).

    Mirrors the logic of curriculum/manager.py CurriculumManager
    but operates on batched outcome lists rather than per-step
    SB3 callback infos.

    The broadcast state is (stage, is_blending, blend_step) —
    enough for workers to set their Spawner correctly.
    """

    def __init__(self, initial_stage=0, seed=0, verbose=1):
        self.stage = initial_stage
        self._max_stage = max(CURRICULUM_STAGES.keys())
        self._blending = False
        self._blend_step = 0
        self._blend_ep_counter = 0
        self._blend_eps_per_step = (CURRICULUM_BLEND_EPISODES
                                    // CURRICULUM_BLEND_STEPS)
        self._ep_count = 0
        self._cooldown_remaining = 0
        self._verbose = verbose

    @property
    def blend_ratio(self):
        if self._blending:
            return _BLEND_RATIOS[self._blend_step]
        return 0.0

    def get_broadcast_state(self):
        """Minimal state tuple for MPI broadcast to workers.

        Returns
        -------
        tuple
            (stage, is_blending, blend_step)
        """
        return (self.stage, self._blending, self._blend_step)

    def process_outcomes(self, outcomes, success_rate):
        """Process a batch of episode outcomes.  Rank 0 only.

        Parameters
        ----------
        outcomes : list[dict]
            Flat list of outcome dicts from all workers.
        success_rate : float
            Current sliding-window success rate.
        """
        for _ in outcomes:
            self._ep_count += 1
            if self.stage >= self._max_stage and not self._blending:
                continue
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                continue
            if self._blending:
                self._advance_blend()
            else:
                self._check_promotion(success_rate)

    def _check_promotion(self, success_rate):
        if (self._ep_count < CURRICULUM_WINDOW_SIZE
                or self.stage >= self._max_stage
                or self._cooldown_remaining > 0):
            return
        stage_cfg = CURRICULUM_STAGES.get(self.stage, {})
        threshold = stage_cfg.get("promotion_threshold",
                                  CURRICULUM_PROMOTION_THRESHOLD)
        if success_rate >= threshold:
            self._blending = True
            self._blend_step = 0
            self._blend_ep_counter = 0
            if self._verbose >= 1:
                ns = self.stage + 1
                print(f"\n[CURRICULUM] {self.stage} → {ns} "
                      f"blend started (sr={success_rate:.2%}, "
                      f"threshold={threshold:.0%}, ep={self._ep_count})")

    def _advance_blend(self):
        self._blend_ep_counter += 1
        if self._blend_ep_counter >= self._blend_eps_per_step:
            self._blend_ep_counter = 0
            self._blend_step += 1
            if self._blend_step >= CURRICULUM_BLEND_STEPS:
                # Promotion complete
                self.stage += 1
                self._blending = False
                self._blend_step = 0
                self._cooldown_remaining = CURRICULUM_WINDOW_SIZE
                if self._verbose >= 1:
                    cfg = CURRICULUM_STAGES.get(self.stage, {})
                    print(f"\n[CURRICULUM] Stage {self.stage} "
                          f"({cfg.get('name', '?')}) fully active "
                          f"(ep={self._ep_count})")
            elif self._verbose >= 1:
                r = _BLEND_RATIOS[self._blend_step]
                print(f"  [CURRICULUM] blend step "
                      f"{self._blend_step + 1}/{CURRICULUM_BLEND_STEPS}: "
                      f"{1-r:.0%}/{r:.0%}")


def apply_curriculum_to_spawner(spawner, curriculum_state, rng):
    """Set the spawner stage based on broadcast curriculum state.

    Called on each episode reset during rollout collection.
    Each worker makes its own random blend decision.

    Parameters
    ----------
    spawner : Spawner
        The worker's local spawner.
    curriculum_state : tuple
        (stage, is_blending, blend_step) from Rank 0.
    rng : np.random.Generator
        Worker-local RNG for blend randomization.
    """
    stage, is_blending, blend_step = curriculum_state
    if not is_blending:
        spawner.set_stage(stage)
    else:
        ratio = _BLEND_RATIOS[blend_step]
        next_stage = stage + 1
        if rng.random() < ratio:
            spawner.set_stage(next_stage)
        else:
            spawner.set_stage(stage)