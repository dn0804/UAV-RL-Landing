"""
Rollout collection and buffer assembly for parallel PPO training.

collect_rollout() replaces SB3's collect_rollouts() — each MPI worker
calls it independently on its own ghost drone.

fill_rollout_buffer() assembles gathered rollouts from all workers
into SB3's RolloutBuffer for gradient computation on Rank 0.
"""

import numpy as np
import torch as th

from rl_uav_package.config.constants import OBS_DIM, ACTION_DIM
from rl_uav_package.utils.training_logger import REWARD_KEYS, DIAG_KEYS
from rl_uav_package.curriculum.mpi_manager import apply_curriculum_to_spawner


def collect_rollout(env, model, n_steps, last_obs, last_episode_start,
                    spawner, curriculum_state, rng):
    """Collect n_steps transitions from a single environment.

    This replaces SB3's collect_rollouts().  Each worker calls this
    independently on its own ghost drone.

    Parameters
    ----------
    env : DroneEnv
        Raw (unwrapped) environment.
    model : SafePPO
        Local model with current policy weights.
    n_steps : int
        Number of steps to collect.
    last_obs : np.ndarray
        Observation carried over from previous rollout (or env.reset).
    last_episode_start : bool
        Whether last_obs is the first step of a new episode.
    spawner : Spawner
        Worker's local spawner for curriculum-aware resets.
    curriculum_state : tuple
        (stage, is_blending, blend_step) for spawner decisions.
    rng : np.random.Generator
        Worker-local RNG.

    Returns
    -------
    rollout : dict
        Numpy arrays of collected transitions.
    episode_outcomes : list[dict]
        Outcome info for each completed episode, including
        per-episode reward/diagnostic accumulations.
    last_obs : np.ndarray
        Observation to carry into next rollout.
    last_episode_start : bool
        Whether last_obs starts a new episode.
    """
    observations = np.zeros((n_steps, OBS_DIM), dtype=np.float32)
    actions = np.zeros((n_steps, ACTION_DIM), dtype=np.float32)
    rewards = np.zeros(n_steps, dtype=np.float32)
    episode_starts = np.zeros(n_steps, dtype=np.float32)
    values = np.zeros(n_steps, dtype=np.float32)
    log_probs = np.zeros(n_steps, dtype=np.float32)
    episode_outcomes = []

    # Per-episode accumulators for detailed logging
    ep_reward_sums = {}
    ep_diag_sums = {}
    ep_steps = 0

    obs = last_obs
    episode_start = last_episode_start
    model.policy.set_training_mode(False)

    for step in range(n_steps):
        # Store current state
        observations[step] = obs
        episode_starts[step] = float(episode_start)

        # ── Policy inference (no gradient) ────────────────
        with th.no_grad():
            obs_t = th.as_tensor(
                obs[np.newaxis], dtype=th.float32, device=model.device)
            action_t, value_t, log_prob_t = model.policy(obs_t)

        action_np = action_t.cpu().numpy().squeeze(0)
        values[step] = value_t.cpu().item()
        log_probs[step] = log_prob_t.cpu().item()

        # Clip to action space bounds
        clipped = np.clip(action_np,
                          env.action_space.low, env.action_space.high)
        actions[step] = clipped

        # ── Environment step ──────────────────────────────
        new_obs, reward, terminated, truncated, info = env.step(clipped)
        rewards[step] = reward
        ep_steps += 1

        # Accumulate per-step reward and diagnostic values
        for key in REWARD_KEYS:
            if key in info:
                ep_reward_sums[key] = (
                    ep_reward_sums.get(key, 0.0) + info[key])
        for key in DIAG_KEYS:
            if key in info:
                ep_diag_sums[key] = (
                    ep_diag_sums.get(key, 0.0) + info[key])

        episode_start = False
        if terminated or truncated:
            # Build detailed outcome dict
            if "episode_outcome" in info:
                outcome = {
                    "outcome": info["episode_outcome"],
                    "final_d_pad": info.get("final_d_pad", 0.0),
                    "final_vz": info.get("final_vz", 0.0),
                    "final_vxy": info.get("final_vxy", 0.0),
                    "length": ep_steps,
                    "hover_checkpoint_reached": info.get(
                        "hover_checkpoint_reached", False),
                    "descent_committed": info.get(
                        "descent_committed", False),
                    "hover_dwell_count": info.get(
                        "hover_dwell_count", 0),
                    "reward_sums": dict(ep_reward_sums),
                    "diag_sums": dict(ep_diag_sums),
                    "diag_steps": ep_steps,
                }
                # Include terminal keys if present
                for tkey in ["terminal/vz_bonus", "terminal/vxy_bonus",
                             "terminal/miss_pos", "terminal/miss_vxy",
                             "terminal/miss_vz", "terminal/miss_avg"]:
                    if tkey in info:
                        outcome[tkey] = info[tkey]
                episode_outcomes.append(outcome)

            # Reset accumulators
            ep_reward_sums.clear()
            ep_diag_sums.clear()
            ep_steps = 0

            # Reset with curriculum-aware stage
            apply_curriculum_to_spawner(spawner, curriculum_state, rng)
            new_obs, _ = env.reset()
            episode_start = True

        obs = new_obs

    # ── Bootstrap value for GAE ───────────────────────────
    with th.no_grad():
        obs_t = th.as_tensor(
            obs[np.newaxis], dtype=th.float32, device=model.device)
        last_value = model.policy.predict_values(obs_t).cpu().item()

    rollout = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "episode_starts": episode_starts,
        "values": values,
        "log_probs": log_probs,
        "last_values": last_value,
        "last_dones": float(episode_start),
    }
    return rollout, episode_outcomes, obs, episode_start


def fill_rollout_buffer(buffer, all_rollouts, n_workers, n_steps):
    """Assemble gathered rollouts into SB3's RolloutBuffer.

    Each worker's data occupies one column in the n_envs dimension.
    SB3's GAE computation handles per-env episode boundaries correctly.

    Parameters
    ----------
    buffer : RolloutBuffer
        Pre-allocated buffer with n_envs=n_workers.
    all_rollouts : list[dict]
        One rollout dict per worker from comm.gather().
    n_workers : int
        Number of MPI ranks / ghost drones.
    n_steps : int
        Steps per worker per update.
    """
    buffer.reset()

    # Direct array assignment — each worker is one "env" column
    for i, r in enumerate(all_rollouts):
        buffer.observations[:, i, :] = r["observations"]
        buffer.actions[:, i, :] = r["actions"]
        buffer.rewards[:, i] = r["rewards"]
        buffer.episode_starts[:, i] = r["episode_starts"]
        buffer.values[:, i] = r["values"]
        buffer.log_probs[:, i] = r["log_probs"]

    buffer.pos = n_steps
    buffer.full = True

    # Compute GAE with per-worker bootstrap values
    last_values = th.tensor(
        [r["last_values"] for r in all_rollouts], dtype=th.float32)
    last_dones = np.array(
        [r["last_dones"] for r in all_rollouts], dtype=np.float32)
    buffer.compute_returns_and_advantage(last_values, last_dones)
