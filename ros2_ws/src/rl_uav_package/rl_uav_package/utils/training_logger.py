"""
Training metrics logger for TensorBoard.

An SB3 BaseCallback that captures the info dict from every step and
episode termination, aggregates statistics, and writes them to
TensorBoard.  Replaces the lightweight EpisodeLogCallback in
train_ppo.py with richer tracking.

Logged metrics:

    reward/*            Per-step reward components (rolled up per episode)
    episode/*           Per-episode outcomes, success rate, duration
    episode/final_*     Terminal state metrics (d_pad, vz, vxy)
    outcomes/*          Outcome type counts over a rolling window
"""

from collections import deque
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_uav_package.envs.termination import (
    OUTCOME_SUCCESS,
    OUTCOME_CRASH_ATTITUDE,
    OUTCOME_CRASH_CONTACT,
    OUTCOME_CRASH_BELOW_PAD,
    OUTCOME_CRASH_WALL,
    OUTCOME_CRASH_OOB,
    OUTCOME_TIMEOUT,
)

ALL_OUTCOMES = [
    OUTCOME_SUCCESS,
    OUTCOME_CRASH_ATTITUDE,
    OUTCOME_CRASH_CONTACT,
    OUTCOME_CRASH_BELOW_PAD,
    OUTCOME_CRASH_WALL,
    OUTCOME_CRASH_OOB,
    OUTCOME_TIMEOUT,
]

# Reward component keys emitted by rewards.py
REWARD_KEYS = [
    "reward/horizontal",
    "reward/descent",
    "reward/yaw",
    "reward/jerk",
    "reward/time",
    "reward/total",
]

# Diagnostic keys from rewards.py
DIAG_KEYS = [
    "diag/centering_gate",
    "diag/gate_blend",
    "diag/delta_z",
]


class TrainingLogger(BaseCallback):
    """Rich TensorBoard logging callback for PPO training.

    Parameters
    ----------
    window_size : int
        Rolling window for success rate and outcome distribution.
    log_reward_freq : int
        Log per-episode reward breakdowns every N episodes.
        Set to 1 for full granularity (slightly more TensorBoard data).
    verbose : int
        Verbosity level.  0 = silent, 1 = episode summaries to console.
    """

    def __init__(
        self,
        window_size: int = 200,
        log_reward_freq: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._window_size = window_size
        self._log_reward_freq = log_reward_freq

        # Rolling outcome window
        self._outcomes: deque[str] = deque(maxlen=window_size)

        # Per-episode accumulators (reset each episode)
        self._ep_reward_sums: dict[str, float] = {}
        self._ep_diag_sums: dict[str, float] = {}
        self._ep_steps: int = 0

        # Counters
        self._episode_count: int = 0
        self._total_successes: int = 0

    def _on_training_start(self) -> None:
        """Log hyperparameters as text for easy reference in TensorBoard."""
        if self.logger is None:
            return
        # SB3's logger.record writes scalars; hparams are logged once
        hparams = {
            "hparam/learning_rate": self.model.learning_rate
            if isinstance(self.model.learning_rate, float)
            else "scheduled",
            "hparam/gamma": self.model.gamma,
            "hparam/n_steps": self.model.n_steps,
            "hparam/batch_size": self.model.batch_size,
            "hparam/n_epochs": self.model.n_epochs,
            "hparam/ent_coef": self.model.ent_coef,
        }
        for key, val in hparams.items():
            if isinstance(val, (int, float)):
                self.logger.record(key, val)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            self._ep_steps += 1

            # Accumulate per-step reward components
            for key in REWARD_KEYS:
                if key in info:
                    self._ep_reward_sums[key] = (
                        self._ep_reward_sums.get(key, 0.0) + info[key]
                    )

            # Accumulate diagnostics (we'll average them)
            for key in DIAG_KEYS:
                if key in info:
                    self._ep_diag_sums[key] = (
                        self._ep_diag_sums.get(key, 0.0) + info[key]
                    )

            # Episode ended
            if "episode_outcome" in info:
                self._on_episode_end(info)

        return True

    def _on_episode_end(self, info: dict) -> None:
        """Process a completed episode."""
        self._episode_count += 1
        outcome = info["episode_outcome"]

        # Track outcome
        self._outcomes.append(outcome)
        if outcome == OUTCOME_SUCCESS:
            self._total_successes += 1

        # ── Success rate (rolling window) ────────────────────
        n_success = sum(1 for o in self._outcomes if o == OUTCOME_SUCCESS)
        success_rate = n_success / len(self._outcomes)
        self.logger.record("episode/success_rate", success_rate)
        self.logger.record("episode/count", self._episode_count)
        self.logger.record("episode/length", self._ep_steps)

        # ── Outcome distribution (rolling window fractions) ──
        window_len = len(self._outcomes)
        for outcome_type in ALL_OUTCOMES:
            count = sum(1 for o in self._outcomes if o == outcome_type)
            fraction = count / window_len
            # Clean name: "crash_attitude" → "outcomes/crash_attitude"
            self.logger.record(f"outcomes/{outcome_type}", fraction)

        # ── Terminal state metrics ───────────────────────────
        if "final_d_pad" in info:
            self.logger.record("episode/final_d_pad", info["final_d_pad"])
        if "final_vz" in info:
            self.logger.record("episode/final_vz", info["final_vz"])
        if "final_vxy" in info:
            self.logger.record("episode/final_vxy", info["final_vxy"])

        # ── Per-episode reward breakdown ─────────────────────
        if self._episode_count % self._log_reward_freq == 0:
            for key in REWARD_KEYS:
                total = self._ep_reward_sums.get(key, 0.0)
                self.logger.record(f"episode_{key}", total)

            # Average diagnostics over episode length
            if self._ep_steps > 0:
                for key in DIAG_KEYS:
                    avg = self._ep_diag_sums.get(key, 0.0) / self._ep_steps
                    self.logger.record(f"episode_avg_{key}", avg)

        # ── Console output ───────────────────────────────────
        if self.verbose >= 1:
            print(
                f"  Ep {self._episode_count:>5d} | "
                f"{outcome:<28s} | "
                f"steps={self._ep_steps:>3d} | "
                f"success_rate={success_rate:.2%}"
            )

        # Reset accumulators
        self._ep_reward_sums.clear()
        self._ep_diag_sums.clear()
        self._ep_steps = 0

    # ── Public accessors for curriculum manager ──────────────

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def success_rate(self) -> float:
        """Rolling success rate over the window.  Returns 0 if no episodes."""
        if not self._outcomes:
            return 0.0
        return sum(1 for o in self._outcomes if o == OUTCOME_SUCCESS) / len(self._outcomes)

    @property
    def recent_outcomes(self) -> list[str]:
        """Copy of the rolling outcome window."""
        return list(self._outcomes)

    def get_outcome_distribution(self) -> dict[str, float]:
        """Fraction of each outcome type in the rolling window."""
        if not self._outcomes:
            return {o: 0.0 for o in ALL_OUTCOMES}
        n = len(self._outcomes)
        return {o: sum(1 for x in self._outcomes if x == o) / n for o in ALL_OUTCOMES}
