"""
Training metrics logger for TensorBoard.

Tracks the checkpoint pipeline: approach → hover checkpoint → descend → land.
"""

from collections import deque
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_uav_package.envs.termination import (
    OUTCOME_SUCCESS, OUTCOME_SUCCESS_NO_CHECKPOINT,
    OUTCOME_CRASH_ATTITUDE, OUTCOME_CRASH_CONTACT,
    OUTCOME_CRASH_BELOW_PAD, OUTCOME_CRASH_OOB,
    OUTCOME_TIMEOUT,
)

# Only full-pipeline landings (checkpoint + land) count toward
# curriculum promotion.
_PROMOTION_OUTCOMES = {OUTCOME_SUCCESS}

ALL_OUTCOMES = [
    OUTCOME_SUCCESS, OUTCOME_SUCCESS_NO_CHECKPOINT,
    OUTCOME_CRASH_ATTITUDE, OUTCOME_CRASH_CONTACT,
    OUTCOME_CRASH_BELOW_PAD, OUTCOME_CRASH_OOB,
    OUTCOME_TIMEOUT,
]

REWARD_KEYS = [
    "reward/horizontal", "reward/z_align", "reward/yaw",
    "reward/centering", "reward/vel_xy", "reward/vel_z",
    "reward/vel_xy_uni", "reward/velocity", "reward/jerk",
    "reward/time", "reward/hover_bonus", "reward/dropout_freeze",
    "reward/descent_committed", "reward/xy_hold", "reward/total",
]

DIAG_KEYS = [
    "diag/d_pad", "diag/delta_z", "diag/pixel_dist", "diag/v_xy",
    "diag/proximity", "diag/z_dist_to_marker", "diag/z_dist_to_target",
    "diag/phase",
]

TERMINAL_KEYS = [
    "terminal/vz_bonus", "terminal/vxy_bonus",
    "terminal/miss_pos", "terminal/miss_vxy", "terminal/miss_vz",
    "terminal/miss_avg",
]


class TrainingLogger(BaseCallback):
    def __init__(self, window_size=200, log_reward_freq=1, verbose=0):
        super().__init__(verbose)
        self._window_size = window_size
        self._log_reward_freq = log_reward_freq
        self._outcomes: deque[str] = deque(maxlen=window_size)
        self._ep_reward_sums: dict[str, float] = {}
        self._ep_diag_sums: dict[str, float] = {}
        self._ep_steps = 0
        self._episode_count = 0
        self._total_successes = 0
        self._rollout_start_time = None
        self._rollout_end_time = None
        self._last_collect_time = None
        self._last_train_time = None

    def _on_training_start(self):
        if self.logger is None:
            return
        for key, val in {
            "hparam/gamma": self.model.gamma,
            "hparam/n_steps": self.model.n_steps,
            "hparam/batch_size": self.model.batch_size,
            "hparam/n_epochs": self.model.n_epochs,
            "hparam/ent_coef": self.model.ent_coef,
        }.items():
            if isinstance(val, (int, float)):
                self.logger.record(key, val)

    def _on_rollout_start(self):
        now = time.time()
        # If we have a previous rollout_end, the gap is the train time
        if self._rollout_end_time is not None:
            self._last_train_time = now - self._rollout_end_time
        self._rollout_start_time = now

    def _on_rollout_end(self):
        now = time.time()
        if self._rollout_start_time is not None:
            self._last_collect_time = now - self._rollout_start_time
        self._rollout_end_time = now

        # Log timing metrics
        if self.logger is not None:
            if self._last_collect_time is not None:
                n_steps = self.model.n_steps
                collect_fps = int(n_steps / max(self._last_collect_time, 1e-6))
                self.logger.record("time/collect_fps", collect_fps)
                self.logger.record("time/collect_seconds",
                                   round(self._last_collect_time, 2))
            if self._last_train_time is not None:
                self.logger.record("time/train_seconds",
                                   round(self._last_train_time, 2))

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            self._ep_steps += 1
            for key in REWARD_KEYS:
                if key in info:
                    self._ep_reward_sums[key] = self._ep_reward_sums.get(key, 0.0) + info[key]
            for key in DIAG_KEYS:
                if key in info:
                    self._ep_diag_sums[key] = self._ep_diag_sums.get(key, 0.0) + info[key]
            if "episode_outcome" in info:
                self._on_episode_end(info)
        return True

    def _on_episode_end(self, info):
        self._episode_count += 1
        outcome = info["episode_outcome"]
        self._outcomes.append(outcome)
        if outcome in _PROMOTION_OUTCOMES:
            self._total_successes += 1

        n_success = sum(1 for o in self._outcomes if o in _PROMOTION_OUTCOMES)
        success_rate = n_success / len(self._outcomes)

        self.logger.record("episode/success_rate", success_rate)
        self.logger.record("episode/count", self._episode_count)
        self.logger.record("episode/length", self._ep_steps)

        window_len = len(self._outcomes)
        for ot in ALL_OUTCOMES:
            self.logger.record(f"outcomes/{ot}",
                               sum(1 for o in self._outcomes if o == ot) / window_len)

        for key in ["final_d_pad", "final_vz", "final_vxy"]:
            if key in info:
                self.logger.record(f"episode/{key}", info[key])

        if "hover_checkpoint_reached" in info:
            self.logger.record("episode/checkpoint_reached",
                               int(info["hover_checkpoint_reached"]))
        if "descent_committed" in info:
            self.logger.record("episode/descent_committed",
                               int(info["descent_committed"]))
        if "hover_dwell_count" in info:
            self.logger.record("episode/hover_dwell_count",
                               info["hover_dwell_count"])

        for key in TERMINAL_KEYS:
            if key in info:
                self.logger.record(key, info[key])

        if self._episode_count % self._log_reward_freq == 0:
            for key in REWARD_KEYS:
                self.logger.record(f"episode_{key}",
                                   self._ep_reward_sums.get(key, 0.0))
            if self._ep_steps > 0:
                for key in DIAG_KEYS:
                    self.logger.record(f"episode_avg_{key}",
                                       self._ep_diag_sums.get(key, 0.0) / self._ep_steps)

        if self.verbose >= 1:
            ckpt = "✓" if info.get("hover_checkpoint_reached") else "·"
            print(f"  Ep {self._episode_count:>5d} | {outcome:<28s} | "
                  f"steps={self._ep_steps:>3d} | ckpt={ckpt} | "
                  f"sr={success_rate:.2%}")

        self._ep_reward_sums.clear()
        self._ep_diag_sums.clear()
        self._ep_steps = 0

    @property
    def episode_count(self):
        return self._episode_count

    @property
    def success_rate(self):
        if not self._outcomes:
            return 0.0
        return sum(1 for o in self._outcomes if o in _PROMOTION_OUTCOMES) / len(self._outcomes)

    @property
    def recent_outcomes(self):
        return list(self._outcomes)

    def get_outcome_distribution(self):
        if not self._outcomes:
            return {o: 0.0 for o in ALL_OUTCOMES}
        n = len(self._outcomes)
        return {o: sum(1 for x in self._outcomes if x == o) / n for o in ALL_OUTCOMES}