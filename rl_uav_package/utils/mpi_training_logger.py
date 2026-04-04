"""
MPI training metrics logger for TensorBoard.

Standalone replacement for TrainingLogger that doesn't inherit from
SB3 callbacks.  Tracks the same metrics as the serial version:
reward component breakdowns, diagnostic averages, terminal bonuses,
outcome distributions, and success rate.

Used only by Rank 0 — workers don't log.
"""

from collections import deque

from rl_uav_package.envs.termination import (
    OUTCOME_SUCCESS, OUTCOME_SUCCESS_NO_CHECKPOINT,
    OUTCOME_CRASH_ATTITUDE, OUTCOME_CRASH_CONTACT,
    OUTCOME_CRASH_BELOW_PAD, OUTCOME_CRASH_OOB,
    OUTCOME_TIMEOUT,
)
from rl_uav_package.utils.training_logger import (
    REWARD_KEYS, DIAG_KEYS, TERMINAL_KEYS, ALL_OUTCOMES,
    _PROMOTION_OUTCOMES,
)

# ═══════════════════════════════════════════════════════════════
# Success Rate Tracker
# ═══════════════════════════════════════════════════════════════


class SuccessTracker:
    """Sliding-window success rate tracker for MPI training.

    Mirrors the success_rate property of the serial TrainingLogger
    but operates on batched outcome dicts rather than per-step infos.
    """

    def __init__(self, window_size=200):
        self._window = deque(maxlen=window_size)
        self._episode_count = 0

    def add_outcomes(self, outcomes):
        """Add a batch of outcome dicts from all workers.

        Parameters
        ----------
        outcomes : list[dict]
            Each dict must have an "outcome" key with the episode
            outcome string.
        """
        for o in outcomes:
            self._window.append(o["outcome"] in _PROMOTION_OUTCOMES)
            self._episode_count += 1

    @property
    def success_rate(self):
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def count(self):
        return self._episode_count

    @property
    def window_size(self):
        return len(self._window)


# ═══════════════════════════════════════════════════════════════
# TensorBoard Logging
# ═══════════════════════════════════════════════════════════════


def log_mpi_metrics(model, all_outcomes, success_tracker, curriculum,
                    update, n_workers, n_steps, wall_time,
                    collect_time=None, train_time=None):
    """Log training metrics to TensorBoard via SB3's logger.

    Mirrors the serial TrainingLogger output format so TensorBoard
    dashboards are consistent between serial and MPI runs.

    Parameters
    ----------
    model : SafePPO
        The PPO model (Rank 0's copy with logger attached).
    all_outcomes : list[dict]
        Flattened episode outcomes from all workers for this update.
    success_tracker : SuccessTracker
        Sliding-window success rate tracker.
    curriculum : MPICurriculumManager
        Curriculum manager for stage/blend logging.
    update : int
        Current update index (0-based).
    n_workers : int
        Number of MPI worker ranks.
    n_steps : int
        Steps per worker per update.
    wall_time : float
        Seconds since training started.
    collect_time : float, optional
        Seconds spent on rollout collection + MPI gather this update.
    train_time : float, optional
        Seconds spent on buffer assembly + gradient update this update.
    """
    logger = model.logger
    total_steps = (update + 1) * n_steps * n_workers

    # ── Timing ────────────────────────────────────────────
    fps = total_steps / max(wall_time, 1e-6)
    logger.record("time/fps", int(fps))
    logger.record("time/total_timesteps", total_steps)
    logger.record("time/time_elapsed", int(wall_time))
    logger.record("time/iterations", update + 1)

    if collect_time is not None:
        steps_this_update = n_steps * n_workers
        collect_fps = int(steps_this_update / max(collect_time, 1e-6))
        logger.record("time/collect_fps", collect_fps)
        logger.record("time/collect_seconds", round(collect_time, 2))
    if train_time is not None:
        logger.record("time/train_seconds", round(train_time, 2))

    # ── Episode metrics ───────────────────────────────────
    if all_outcomes:
        outcome_types = [o["outcome"] for o in all_outcomes]
        n_eps = len(outcome_types)

        logger.record("episode/count", success_tracker.count)
        logger.record("episode/success_rate",
                       success_tracker.success_rate)
        logger.record("episode/episodes_this_update", n_eps)

        # Most recent episode details (matches serial logger)
        last = all_outcomes[-1]
        logger.record("episode/length", last.get("length", 0))
        for key in ["final_d_pad", "final_vz", "final_vxy"]:
            if key in last:
                logger.record(f"episode/{key}", last[key])
        if "hover_checkpoint_reached" in last:
            logger.record("episode/checkpoint_reached",
                          int(last["hover_checkpoint_reached"]))
        if "descent_committed" in last:
            logger.record("episode/descent_committed",
                          int(last["descent_committed"]))
        if "hover_dwell_count" in last:
            logger.record("episode/hover_dwell_count",
                          last["hover_dwell_count"])

        # ── Outcome distribution (over window) ────────────
        window_len = success_tracker.window_size
        if window_len > 0:
            for ot in ALL_OUTCOMES:
                logger.record(f"outcomes/{ot}",
                              sum(1 for o in all_outcomes
                                  if o["outcome"] == ot) / n_eps)

        # ── Terminal keys (from most recent episode) ──────
        for tkey in TERMINAL_KEYS:
            if tkey in last:
                logger.record(tkey, last[tkey])

        # ── Reward component breakdowns ───────────────────
        # Average across all completed episodes this update
        reward_totals = {}
        diag_totals = {}
        total_diag_steps = 0
        n_with_rewards = 0

        for o in all_outcomes:
            rsums = o.get("reward_sums", {})
            if rsums:
                n_with_rewards += 1
                for key, val in rsums.items():
                    reward_totals[key] = (
                        reward_totals.get(key, 0.0) + val)
            dsums = o.get("diag_sums", {})
            dsteps = o.get("diag_steps", 0)
            if dsums and dsteps > 0:
                total_diag_steps += dsteps
                for key, val in dsums.items():
                    diag_totals[key] = (
                        diag_totals.get(key, 0.0) + val)

        # Log average reward sums per episode
        if n_with_rewards > 0:
            for key in REWARD_KEYS:
                val = reward_totals.get(key, 0.0) / n_with_rewards
                # Match serial format: "episode_reward/horizontal"
                logger.record(f"episode_{key}", val)

        # Log average diagnostic values per step
        if total_diag_steps > 0:
            for key in DIAG_KEYS:
                val = diag_totals.get(key, 0.0) / total_diag_steps
                # Match serial format: "episode_avg_diag/d_pad"
                logger.record(f"episode_avg_{key}", val)

    # ── Curriculum ────────────────────────────────────────
    logger.record("curriculum/stage", curriculum.stage)
    logger.record("curriculum/blend_ratio", curriculum.blend_ratio)

    # ── Dump all recorded values ──────────────────────────
    logger.dump(step=total_steps)