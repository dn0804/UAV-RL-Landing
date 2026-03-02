import os
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Adjust this import based on your exact ROS 2 package structure
from rl_uav_package.envs.drone_env import DroneEnv

def main():
    # 1. Define paths for saving models and logs
    models_dir = "models/ppo_landing"
    log_dir = "logs/"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("[INFO] Initializing Drone Environment...")
    
    # 2. Instantiate the environment
    # The DroneEnv handles its own rclpy.init() and background thread
    raw_env = DroneEnv()

    # 3. Sanity Check
    # SB3 provides a checker to ensure the custom environment follows Gymnasium API rules.
    # We run this before training to catch shape/dtype mismatches early.
    print("[INFO] Running SB3 Environment Checker...")
    try:
        check_env(raw_env, warn=True)
        print("[INFO] Environment check passed!")
    except Exception as e:
        print(f"[ERROR] Environment check failed: {e}")
        raw_env.close()
        return

    # 4. Vectorize the environment
    # Stable-Baselines3 requires environments to be vectorized. 
    # We MUST use DummyVecEnv (single thread) rather than SubprocVecEnv (multiprocessing) 
    # because duplicating ROS 2 nodes with identical names/topics in parallel processes will crash.
    env = DummyVecEnv([lambda: raw_env])

    # 5. Initialize the PPO Agent
    # MlpPolicy: Uses a standard Multi-Layer Perceptron (no CNNs, since we have a flat state vector)
    # verbose=1: Prints training metrics to the console
    print("[INFO] Initializing PPO Agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
    )

    # 6. Execute the Integration Test Training Loop
    # 10,000 steps is very short (about 16 minutes of simulated flight at 10Hz)
    # but it is enough to prove the physics engine, ROS bridge, and RL agent are talking.
    total_timesteps = 10_000
    
    print(f"[INFO] Starting training for {total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=total_timesteps, tb_log_name="PPO_integration_test")
        
        # 7. Save the test model
        model_path = os.path.join(models_dir, "ppo_test_model")
        model.save(model_path)
        print(f"[INFO] Model saved to {model_path}.zip")

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user. Saving current model...")
        model_path = os.path.join(models_dir, "ppo_interrupted_model")
        model.save(model_path)
        
    finally:
        # 8. Clean up ROS 2 nodes and threads safely
        print("[INFO] Shutting down environment...")
        env.close()

if __name__ == "__main__":
    main()