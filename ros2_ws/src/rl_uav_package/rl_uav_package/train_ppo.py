"""
PPO training entry point for autonomous UAV landing.

Usage:
    # Stage 1 baseline, 500k steps
    ros2 run rl_uav_package train_ppo

    # Or directly:
    python3 -m rl_uav_package.train_ppo --total-timesteps 500000 --stage 1

    # Resume from checkpoint:       
    python3 -m rl_uav_package.train_ppo --resume models/ppo_landing/latest.zip
"""

import argparse
import os
import sys
import time

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_uav_package.config.constants import (
    PPO_CONFIG, NET_ARCH, ACTIVATION_FN, LOG_STD_INIT, LOG_STD_MIN
)
from rl_uav_package.envs.drone_env import DroneEnv
from rl_uav_package.utils.training_logger import TrainingLogger
from rl_uav_package.curriculum.manager import CurriculumManager

# ── SafePPO: NaN watchdog + std floor ───────────────────────────────

class SafePPO(PPO):
    """PPO with built-in NaN recovery and entropy floor.

    Overrides train() to:
      1. Save a copy of weights before SGD
      2. Run the normal PPO update
      3. Check for NaN in any parameter — if found, revert to saved weights
      4. Clamp log_std to LOG_STD_MIN to prevent entropy collapse

    This is serialization-safe (no closures capturing external state),
    unlike the monkey-patching approach which broke checkpoint saving.
    """

    def __init__(self, *args, log_std_min: float = LOG_STD_MIN, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_std_min = log_std_min
        self._nan_revert_count = 0

    def train(self) -> None:
        # Save weights before SGD
        saved = {k: v.clone() for k, v in self.policy.state_dict().items()}

        # Normal PPO update
        super().train()

        # Check for NaN corruption
        has_nan = any(th.isnan(p).any() for p in self.policy.parameters())
        if has_nan:
            self.policy.load_state_dict(saved)
            self._nan_revert_count += 1
            print(f"[NaN WATCHDOG] Reverted weights "
                  f"(occurrence #{self._nan_revert_count})")

        # Enforce std floor
        with th.no_grad():
            self.policy.log_std.clamp_(min=self.log_std_min)

# ── Linear learning rate schedule ────────────────────────────────────

def linear_schedule(initial_lr: float):
    """Return a callable that decays the learning rate linearly to 0.3.

    SB3 calls this function with progress_remaining ∈ [1.0, 0.3],
    where 1.0 is the start of training and 0.3 is the end.
    """
    def schedule(progress_remaining: float) -> float:
        return initial_lr * (0.1 + 0.7 * progress_remaining)
    return schedule


# ── Activation function resolution ──────────────────────────────────

ACTIVATION_MAP = {
    "Tanh": th.nn.Tanh,
    "ReLU": th.nn.ReLU,
}


# ── Main ────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train PPO agent for UAV precision landing.",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=500_000,
        help="Total training timesteps (default: 500k for Stage 1 baseline).",
    )
    parser.add_argument(
        "--stage", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7],
        help="Curriculum stage to train in (default: 0).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a saved model .zip to resume training from.",
    )
    parser.add_argument(
        "--models-dir", type=str, default="models/ppo_landing",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="TensorBoard run name. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--skip-env-check", action="store_true",
        help="Skip the SB3 environment checker (faster startup).",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=50_000,
        help="Save a checkpoint every N timesteps.",
    )
    parser.add_argument(
        "--no-curriculum", action="store_true",
        help="Disable curriculum progression.  Train on a single stage only.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ── Run name ─────────────────────────────────────────────────
    if args.run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.run_name = f"PPO_stage{args.stage}_{timestamp}"

    print(f"[INFO] Run: {args.run_name}")
    print(f"[INFO] Stage: {args.stage}")
    print(f"[INFO] Timesteps: {args.total_timesteps:,}")
    print(f"[INFO] Seed: {args.seed}")

    # ── Environment ──────────────────────────────────────────────
    print("[INFO] Creating environment...")
    raw_env = DroneEnv(initial_stage=args.stage, seed=args.seed)

    if not args.skip_env_check:
        print("[INFO] Running SB3 environment checker...")
        try:
            check_env(raw_env, warn=True)
            print("[INFO] Environment check passed.")
        except Exception as e:
            print(f"[ERROR] Environment check failed: {e}")
            raw_env.close()
            return 1

    # DummyVecEnv (single process) — SubprocVecEnv would duplicate
    # ROS 2 nodes with identical names, causing crashes.
    env = DummyVecEnv([lambda: raw_env])

    # ── Resolve activation function ──────────────────────────────
    activation_cls = ACTIVATION_MAP.get(ACTIVATION_FN)
    if activation_cls is None:
        print(f"[ERROR] Unknown activation: {ACTIVATION_FN}")
        env.close()
        return 1

    # ── Policy kwargs ────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=NET_ARCH,
        activation_fn=activation_cls,
        log_std_init=LOG_STD_INIT,
    )

    # ── Create or load model ─────────────────────────────────────
    if args.resume:
        print(f"[INFO] Resuming from {args.resume}")
        model = SafePPO.load(
            args.resume,
            env=env,
            tensorboard_log=args.log_dir,
            learning_rate=linear_schedule(PPO_CONFIG["learning_rate"]),
        )
        model.ent_coef = PPO_CONFIG["ent_coef"]  # use config value
        model.target_kl = None                     # no KL early stopping
        model.log_std_min = LOG_STD_MIN            # std floor for SafePPO
        # Override LR schedule for remaining training
        model.lr_schedule = model.learning_rate
        # Clamp log_std if it collapsed during previous training
        with th.no_grad():
            old_std = model.policy.log_std.exp().mean().item()
            model.policy.log_std.clamp_(min=LOG_STD_MIN)
            new_std = model.policy.log_std.exp().mean().item()
            if old_std != new_std:
                print(f"[INFO] log_std clamped: std {old_std:.3f} → {new_std:.3f}")
    else:
        print("[INFO] Initializing new PPO agent...")
        model = SafePPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule(PPO_CONFIG["learning_rate"]),
            gamma=PPO_CONFIG["gamma"],
            gae_lambda=PPO_CONFIG["gae_lambda"],
            clip_range=PPO_CONFIG["clip_range"],
            n_epochs=PPO_CONFIG["n_epochs"],
            batch_size=PPO_CONFIG["batch_size"],
            n_steps=PPO_CONFIG["n_steps"],
            ent_coef=PPO_CONFIG["ent_coef"],
            vf_coef=PPO_CONFIG["vf_coef"],
            max_grad_norm=PPO_CONFIG["max_grad_norm"],
            log_std_min=LOG_STD_MIN,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            tensorboard_log=args.log_dir,
        )

    # Print param count for verification
    total_params = sum(
        p.numel() for p in model.policy.parameters()
    )
    print(f"[INFO] Policy parameters: {total_params:,}")

    # ── Callbacks ────────────────────────────────────────────────
    training_logger = TrainingLogger(window_size=200, verbose=0)

    callbacks = [
        training_logger,
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=args.models_dir,
            name_prefix="ppo_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
    ]

    if not args.no_curriculum:
        curriculum = CurriculumManager(
            spawner=raw_env.spawner,
            training_logger=training_logger,
            seed=args.seed,
            verbose=1,
        )
        callbacks.append(curriculum)
        print("[INFO] Curriculum manager enabled.")
    else:
        print(f"[INFO] Curriculum disabled. Training on stage {args.stage} only.")

    # ── Train ────────────────────────────────────────────────────
    print(f"[INFO] Starting training for {args.total_timesteps:,} timesteps...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=args.run_name,
            reset_num_timesteps=(args.resume is None),
        )

        # Save final model
        final_path = os.path.join(args.models_dir, "latest")
        model.save(final_path)
        print(f"[INFO] Final model saved to {final_path}.zip")

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Saving...")
        interrupted_path = os.path.join(args.models_dir, "interrupted")
        model.save(interrupted_path)
        print(f"[INFO] Interrupted model saved to {interrupted_path}.zip")

    finally:
        print("[INFO] Shutting down...")
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())