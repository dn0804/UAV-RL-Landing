"""
MPI-parallel PPO training for autonomous UAV landing.

Synchronized parallel rollout collection with ghost drones in a
single Gazebo instance.  Each MPI rank controls one ghost drone.

Architecture:
    Rank 0 (Learner):  Owns the PPO model.  Runs gradient updates.
    Rank 1..N (Workers): Each controls one drone.  Collects rollouts.
    All ranks participate in rollout collection.

Usage (via run.sh):
    train --multi 4 --timesteps 2000000
"""

import argparse
import os
import sys
import time

import numpy as np
import torch as th
from mpi4py import MPI

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_uav_package.config.constants import (
    PPO_CONFIG, NET_ARCH, ACTIVATION_FN, LOG_STD_INIT, LOG_STD_MIN,
    CURRICULUM_STAGES, CURRICULUM_WINDOW_SIZE,
)
from rl_uav_package.envs.drone_env import DroneEnv
from rl_uav_package.train_ppo import SafePPO, linear_schedule, ACTIVATION_MAP
from rl_uav_package.curriculum.mpi_manager import MPICurriculumManager
from rl_uav_package.utils.mpi_comms import broadcast_weights, broadcast_curriculum
from rl_uav_package.utils.mpi_training_logger import SuccessTracker, log_mpi_metrics
from rl_uav_package.utils.rollout import collect_rollout, fill_rollout_buffer


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="MPI-parallel PPO training for UAV landing.")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--stage", type=int, default=0,
                   choices=list(CURRICULUM_STAGES.keys()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--models-dir", type=str, default="models/ppo_landing")
    p.add_argument("--log-dir", type=str, default="logs/")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--skip-env-check", action="store_true")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--n-workers", type=int, required=True,
                   help="Must match mpirun -n (set by run.sh)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ── MPI initialization ────────────────────────────────
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != args.n_workers:
        if rank == 0:
            print(f"[ERROR] MPI world size ({size}) != "
                  f"--n-workers ({args.n_workers})")
        return 1

    n_workers = size
    n_steps = PPO_CONFIG["n_steps"]

    if rank == 0:
        os.makedirs(args.models_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        if args.run_name is None:
            args.run_name = (f"MPI{n_workers}_s{args.stage}_"
                             f"{time.strftime('%Y%m%d_%H%M%S')}")
        cfg = CURRICULUM_STAGES[args.stage]
        print(f"[INFO] Run: {args.run_name} | "
              f"Stage {args.stage} ({cfg['name']})")
        print(f"[INFO] Workers: {n_workers} | "
              f"Steps/worker: {n_steps:,} | "
              f"Buffer/update: {n_steps * n_workers:,}")
        print(f"[INFO] Timesteps: {args.total_timesteps:,} | "
              f"Seed: {args.seed}")

    # ── Environment ───────────────────────────────────────
    env = DroneEnv(
        model_name=f"tello_{rank}",
        image_topic=f"/camera_{rank}/image_raw",
        initial_stage=args.stage,
        seed=args.seed + rank,
    )

    # DummyVecEnv needed for SB3 model construction only
    vec_env = DummyVecEnv([lambda: env])

    if rank == 0:
        print(f"[INFO] All {n_workers} environments created.")

    # ── Model ─────────────────────────────────────────────
    activation_cls = ACTIVATION_MAP.get(ACTIVATION_FN)
    policy_kwargs = dict(
        net_arch=NET_ARCH,
        activation_fn=activation_cls,
        log_std_init=LOG_STD_INIT,
    )

    if args.resume:
        if rank == 0:
            print(f"[INFO] Resuming from {args.resume}")
        model = SafePPO.load(
            args.resume, env=vec_env,
            tensorboard_log=args.log_dir,
            learning_rate=linear_schedule(PPO_CONFIG["learning_rate"]))
        model.ent_coef = PPO_CONFIG["ent_coef"]
        model.log_std_min = LOG_STD_MIN
        with th.no_grad():
            model.policy.log_std.clamp_(min=LOG_STD_MIN)
    else:
        model = SafePPO(
            "MlpPolicy", vec_env,
            learning_rate=linear_schedule(PPO_CONFIG["learning_rate"]),
            gamma=PPO_CONFIG["gamma"],
            gae_lambda=PPO_CONFIG["gae_lambda"],
            clip_range=PPO_CONFIG["clip_range"],
            n_epochs=PPO_CONFIG["n_epochs"],
            batch_size=PPO_CONFIG["batch_size"],
            n_steps=n_steps,
            ent_coef=PPO_CONFIG["ent_coef"],
            vf_coef=PPO_CONFIG["vf_coef"],
            max_grad_norm=PPO_CONFIG["max_grad_norm"],
            log_std_min=LOG_STD_MIN,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=args.seed + rank,
            tensorboard_log=args.log_dir if rank == 0 else None,
        )

    # Set up TensorBoard logger on Rank 0
    if rank == 0:
        tb_path = os.path.join(args.log_dir, args.run_name)
        new_logger = configure_logger(tb_path, ["stdout", "tensorboard"])
        model.set_logger(new_logger)

        # Replace rollout buffer with one sized for all workers
        model.rollout_buffer = RolloutBuffer(
            buffer_size=n_steps,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=model.device,
            n_envs=n_workers,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
        )

        print(f"[INFO] Params: "
              f"{sum(p.numel() for p in model.policy.parameters()):,}")

    # ── Curriculum + tracking (Rank 0) ────────────────────
    curriculum = MPICurriculumManager(
        initial_stage=args.stage, seed=args.seed, verbose=1)
    success_tracker = SuccessTracker(window_size=CURRICULUM_WINDOW_SIZE)

    # Worker-local RNG for curriculum blend randomization
    worker_rng = np.random.default_rng(args.seed + rank + 1000)

    # ── Scheduling ────────────────────────────────────────
    steps_per_update = n_steps * n_workers
    total_updates = args.total_timesteps // steps_per_update
    checkpoint_interval = max(1, args.checkpoint_freq // steps_per_update)

    # Set total timesteps for learning rate schedule
    model._total_timesteps = args.total_timesteps

    if rank == 0:
        print(f"[INFO] Total updates: {total_updates:,} | "
              f"Checkpoint every {checkpoint_interval} updates")
        print(f"[INFO] Starting training loop...")
        print("=" * 60)

    # ── Initial reset ─────────────────────────────────────
    obs, _ = env.reset()
    episode_start = True

    # Initial curriculum state
    if rank == 0:
        curriculum_state = curriculum.get_broadcast_state()
    else:
        curriculum_state = None
    curriculum_state = broadcast_curriculum(curriculum_state, comm)

    # ── Main loop ─────────────────────────────────────────
    wall_start = time.time()

    try:
        for update in range(total_updates):
            # ── 1. Broadcast weights ──────────────────────
            broadcast_weights(model, comm, rank)

            # ── 2. Collect rollout ────────────────────────
            collect_start = time.time()
            rollout, episode_outcomes, obs, episode_start = \
                collect_rollout(
                    env=env,
                    model=model,
                    n_steps=n_steps,
                    last_obs=obs,
                    last_episode_start=episode_start,
                    spawner=env.spawner,
                    curriculum_state=curriculum_state,
                    rng=worker_rng,
                )

            # ── 3. Gather rollouts to Rank 0 ─────────────
            all_rollouts = comm.gather(rollout, root=0)
            all_outcomes_nested = comm.gather(episode_outcomes, root=0)
            collect_time = time.time() - collect_start

            # ── Steps 4-9: Rank 0 only ───────────────────
            if rank == 0:
                # Flatten outcome lists from all workers
                all_outcomes = []
                for worker_outcomes in all_outcomes_nested:
                    all_outcomes.extend(worker_outcomes)

                # ── 4. Fill buffer + GAE ──────────────────
                train_start = time.time()
                fill_rollout_buffer(
                    model.rollout_buffer, all_rollouts,
                    n_workers, n_steps)

                # ── 5. Update progress for LR schedule ────
                timestep = (update + 1) * steps_per_update
                model.num_timesteps = timestep
                model._current_progress_remaining = (
                    1.0 - timestep / args.total_timesteps)

                # ── 6. PPO gradient update ────────────────
                model.train()
                train_time = time.time() - train_start

                # ── 7. Track outcomes + curriculum ────────
                if all_outcomes:
                    success_tracker.add_outcomes(all_outcomes)

                if not args.no_curriculum:
                    curriculum.process_outcomes(
                        all_outcomes, success_tracker.success_rate)

                # ── 8. Log + checkpoint ───────────────────
                wall_elapsed = time.time() - wall_start
                log_mpi_metrics(
                    model, all_outcomes, success_tracker, curriculum,
                    update, n_workers, n_steps, wall_elapsed,
                    collect_time=collect_time,
                    train_time=train_time)

                if (update + 1) % checkpoint_interval == 0:
                    step_count = (update + 1) * steps_per_update
                    path = os.path.join(
                        args.models_dir,
                        f"ppo_checkpoint_{step_count}_steps")
                    model.save(path)
                    model.save(os.path.join(args.models_dir, "latest"))

            # ── 9. Broadcast curriculum state ─────────────
            if rank == 0:
                curriculum_state = curriculum.get_broadcast_state()
            else:
                curriculum_state = None
            curriculum_state = broadcast_curriculum(
                curriculum_state, comm)

    except KeyboardInterrupt:
        if rank == 0:
            print("\n[INFO] Interrupted — saving model...")
            model.save(os.path.join(args.models_dir, "interrupted"))
            print("[INFO] Saved interrupted.zip")

    # ── Cleanup ───────────────────────────────────────────
    if rank == 0:
        model.save(os.path.join(args.models_dir, "latest"))
        wall_total = time.time() - wall_start
        total_steps = total_updates * steps_per_update
        print("=" * 60)
        print(f"[INFO] Training complete.")
        print(f"[INFO] Total steps: {total_steps:,} | "
              f"Wall time: {wall_total:.0f}s | "
              f"FPS: {total_steps / max(wall_total, 1):.0f}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())