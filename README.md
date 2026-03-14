# Autonomous UAV Landing Using Reinforcement Learning

A PPO-trained neural network agent that learns autonomous precision landing on a DJI Tello drone through simulated trial and error. The agent practices thousands of landing attempts inside a Gazebo physics simulator, then transfers the learned policy to real hardware.

**Stack:** ROS 2 Humble · Gazebo Garden · Stable-Baselines3 (PPO) · OpenCV · Docker

## Project Overview

The goal is to land a drone on a wall-mounted target using only its forward-facing camera and onboard sensors. The agent receives a 29-dimensional observation vector (position, velocity, ToF range, dropout timer, and 5-step action history) and outputs 4 continuous velocity commands. Training uses a 5-component progress-based reward function with a Gaussian-gated descent mechanism, a 3-stage spatial curriculum with blended transitions, and domain randomization for sim-to-real transfer.

See `UAV_RL_Landing_Project_Plan.md` for the full technical design.

## Architecture

```
rl_uav_package/
├── config/constants.py           # Every tunable number in one place
├── envs/
│   ├── drone_env.py              # Gymnasium wrapper (ROS 2 + Gazebo orchestrator)
│   ├── rewards.py                # 5-component progress-based reward function
│   ├── termination.py            # Crash / success / timeout conditions
│   ├── observations.py           # 29-dim normalized observation builder
│   └── spawner.py                # Rejection-sampled spawn positions
├── filters/ema.py                # EMA velocity filter (filter-in-the-loop)
├── vision/aruco_tracker.py       # Coordinate transforms, dropout timer, solvePnP
├── curriculum/manager.py         # Blended 3-stage curriculum transitions
├── utils/training_logger.py      # TensorBoard metrics and episode tracking
├── domain_randomization/         # (Future: wind, sensor noise, battery sag, etc.)
├── train_ppo.py                  # Training entry point
├── launch/
│   ├── tello_world.sdf           # Gazebo world (wall, desk, marker, Tello)
│   └── tello_sim.launch.py       # ROS 2 launch file
└── config/camera_config.yaml     # Camera intrinsics (sim and real)
```

## Prerequisites

- Docker and Docker Compose
- Git
- NVIDIA GPU + drivers (optional, for Gazebo GUI rendering)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bstahman/UAV-RL-Landing.git
cd UAV-RL-Landing
```

### 2. Build and Start the Container

**Option A — Docker Compose (headless training):**

```bash
sudo bash scripts/build_container.sh
docker exec -it uav_rl_container bash
```

**Option B — VS Code Dev Container (recommended for development):**

Open the repo in VS Code and select "Reopen in Container" when prompted. The `.devcontainer/devcontainer.json` is pre-configured with GPU access and X11 display forwarding.

> Your local repository is mounted into the container. Code edits on the host are reflected immediately.

### 3. Build the ROS 2 Workspace

From inside the container:

```bash
cd ros2_ws
colcon build --packages-select rl_uav_package --symlink-install
```

**When do you need to rebuild?** Because of `--symlink-install`, edits to Python files take effect immediately. You only need to rebuild after changing `setup.py`, `package.xml`, launch files, or adding new sub-packages.

## Usage

### Training

```bash
# Stage 1 baseline (500k steps, ~40 min)
bash scripts/run.sh --no-curriculum --total-timesteps 500000 --skip-env-check

# Full curriculum training (Stage 1 → 2 → 3)
bash scripts/run.sh --total-timesteps 2000000

# With Gazebo GUI visible (requires GPU + X11)
bash scripts/run.sh --gui --no-curriculum --total-timesteps 500000 --skip-env-check

# Resume from checkpoint
bash scripts/run.sh --resume models/ppo_landing/latest.zip --total-timesteps 1000000
```

All flags after `--gui` are forwarded to `train_ppo.py`. Run `python3 -m rl_uav_package.train_ppo --help` for the full list.

### Monitoring

TensorBoard starts automatically with every training run:

```
http://localhost:6006
```

Key metrics to watch:
- `episode/success_rate` — primary convergence signal
- `episode_reward/*` — per-component reward breakdown
- `outcomes/*` — crash type distribution (attitude, OOB, below pad, etc.)
- `episode/final_d_pad` — how close the agent gets at termination

### Running Tests

```bash
cd ros2_ws/src/rl_uav_package
python3 -m pytest test/ -v
```

The test suite covers rewards, termination, observations, EMA filter, spawner, ArUco tracker, drone_env pure functions, training logger, and curriculum manager — all without requiring a running simulator.

## Training Pipeline

The agent trains through a 3-stage spatial curriculum:

| Stage | Spawn Range | Approach Angle | What the Agent Learns |
|-------|-------------|----------------|----------------------|
| 1 — Close | 0.3–0.8 m | ±10° | Basic alignment and descent |
| 2 — Medium | 0.5–1.5 m | ±25° | Longer approaches, sensor noise tolerance |
| 3 — Full | 0.3–4.0 m | ±60° | Full operational envelope |

Stage transitions are triggered at 70% success rate over a rolling window and blended over ~200 episodes to prevent value function collapse.

## Coordinate System

```
Wall at x = 0, marker at (0, 0, 0.90)
Landing pad center at (0.30, 0, 0.75)
+x = into room (away from wall)
+y = left (facing wall)
+z = up
Drone spawns at positive x, facing the wall
```

## Project Status

- [x] Simulation infrastructure (Gazebo + ROS 2 bridge)
- [x] Gymnasium environment wrapper with modular architecture
- [x] 5-component reward function with Gaussian-gated descent
- [x] 3-stage curriculum with blended transitions
- [x] TensorBoard logging with reward breakdowns and outcome tracking
- [x] 178 unit tests (all pure computation, no simulator required)
- [ ] Baseline agent convergence (Stage 1, no DR)
- [ ] Vision pipeline integration (ArUco tracking via camera)
- [ ] MPI-parallelized training on HPC cluster
- [ ] Domain randomization (wind, sensor noise, battery sag, furniture)
- [ ] Real hardware deployment on DJI Tello