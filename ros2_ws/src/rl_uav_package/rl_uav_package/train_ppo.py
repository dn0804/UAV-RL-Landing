"""
PPO training entry point for autonomous UAV landing.

Usage:
    python3 -m rl_uav_package.train_ppo --total-timesteps 500000 --stage 0
    python3 -m rl_uav_package.train_ppo --resume models/ppo_landing/latest.zip
"""

import argparse, os, sys, time
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_uav_package.config.constants import (
    PPO_CONFIG, NET_ARCH, ACTIVATION_FN, LOG_STD_INIT, LOG_STD_MIN,
    CURRICULUM_STAGES,
)
from rl_uav_package.envs.drone_env import DroneEnv
from rl_uav_package.utils.training_logger import TrainingLogger
from rl_uav_package.curriculum.manager import CurriculumManager


class SafePPO(PPO):
    def __init__(self, *args, log_std_min=LOG_STD_MIN, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_std_min = log_std_min
        self._nan_revert_count = 0

    def train(self):
        saved = {k: v.clone() for k, v in self.policy.state_dict().items()}
        super().train()
        if any(th.isnan(p).any() for p in self.policy.parameters()):
            self.policy.load_state_dict(saved)
            self._nan_revert_count += 1
            print(f"[NaN WATCHDOG] Reverted (#{self._nan_revert_count})")
        with th.no_grad():
            self.policy.log_std.clamp_(min=self.log_std_min)


def linear_schedule(initial_lr):
    def schedule(progress_remaining):
        return initial_lr * (0.1 + 0.7 * progress_remaining)
    return schedule


ACTIVATION_MAP = {"Tanh": th.nn.Tanh, "ReLU": th.nn.ReLU}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train PPO for UAV landing.")
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
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.run_name is None:
        args.run_name = f"PPO_s{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}"

    cfg = CURRICULUM_STAGES[args.stage]
    print(f"[INFO] Run: {args.run_name} | Stage {args.stage} ({cfg['name']})")
    print(f"[INFO] Spawn: {cfg['d_min']}-{cfg['d_max']}m | "
          f"Landing: d<{cfg['success_d_xy_max']}m vz<{cfg['success_vz_max']}m/s")
    print(f"[INFO] Timesteps: {args.total_timesteps:,} | Seed: {args.seed}")

    raw_env = DroneEnv(initial_stage=args.stage, seed=args.seed)

    if not args.skip_env_check:
        try:
            check_env(raw_env, warn=True)
        except Exception as e:
            print(f"[ERROR] Env check failed: {e}")
            raw_env.close()
            return 1

    env = DummyVecEnv([lambda: raw_env])

    activation_cls = ACTIVATION_MAP.get(ACTIVATION_FN)
    if activation_cls is None:
        env.close()
        return 1

    policy_kwargs = dict(net_arch=NET_ARCH, activation_fn=activation_cls,
                         log_std_init=LOG_STD_INIT)

    if args.resume:
        print(f"[INFO] Resuming from {args.resume}")
        model = SafePPO.load(args.resume, env=env,
                             tensorboard_log=args.log_dir,
                             learning_rate=linear_schedule(PPO_CONFIG["learning_rate"]))
        model.ent_coef = PPO_CONFIG["ent_coef"]
        model.target_kl = None
        model.log_std_min = LOG_STD_MIN
        model.lr_schedule = model.learning_rate
        with th.no_grad():
            model.policy.log_std.clamp_(min=LOG_STD_MIN)
    else:
        model = SafePPO(
            "MlpPolicy", env,
            learning_rate=linear_schedule(PPO_CONFIG["learning_rate"]),
            gamma=PPO_CONFIG["gamma"], gae_lambda=PPO_CONFIG["gae_lambda"],
            clip_range=PPO_CONFIG["clip_range"], n_epochs=PPO_CONFIG["n_epochs"],
            batch_size=PPO_CONFIG["batch_size"], n_steps=PPO_CONFIG["n_steps"],
            ent_coef=PPO_CONFIG["ent_coef"], vf_coef=PPO_CONFIG["vf_coef"],
            max_grad_norm=PPO_CONFIG["max_grad_norm"], log_std_min=LOG_STD_MIN,
            policy_kwargs=policy_kwargs, verbose=1, seed=args.seed,
            tensorboard_log=args.log_dir)

    print(f"[INFO] Params: {sum(p.numel() for p in model.policy.parameters()):,}")

    logger = TrainingLogger(window_size=200, verbose=0)
    callbacks = [
        logger,
        CheckpointCallback(save_freq=args.checkpoint_freq,
                           save_path=args.models_dir,
                           name_prefix="ppo_checkpoint"),
    ]

    if not args.no_curriculum:
        callbacks.append(CurriculumManager(
            spawner=raw_env.spawner, training_logger=logger,
            seed=args.seed, verbose=1))
        print("[INFO] Curriculum enabled.")

    print(f"[INFO] Training for {args.total_timesteps:,} steps...")
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                    tb_log_name=args.run_name,
                    reset_num_timesteps=(args.resume is None))
        model.save(os.path.join(args.models_dir, "latest"))
    except KeyboardInterrupt:
        model.save(os.path.join(args.models_dir, "interrupted"))
        print("\n[INFO] Interrupted, model saved.")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())