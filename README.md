# Autonomous UAV Landing Using Reinforcement Learning

A PPO-trained neural network agent that learns autonomous precision landing on a DJI Tello drone through simulated trial and error. The agent practices thousands of landing attempts inside a Gazebo physics simulator, then transfers the learned policy to real hardware.

**Stack:** Gazebo Garden · Stable-Baselines3 (PPO) · OpenCV · Docker

## Project Overview

The goal is to land a drone on a wall-mounted target using only its forward-facing camera and onboard sensors. The agent receives a 33-dimensional observation vector (relative position, velocity, ToF range, dropout timer, and 5-step action history) and outputs 4 continuous velocity commands. Training uses a multi-component progress-based reward function with a checkpoint-gated descent mechanism, an 8-stage spatial curriculum with blended transitions, and domain randomization for sim-to-real transfer.

## Architecture

```
rl_uav_package/
├── config/constants.py           # Every tunable number in one place
├── envs/
│   ├── drone_env.py              # Gymnasium wrapper (gz-transport + Gazebo)
│   ├── rewards.py                # Multi-component progress-based reward
│   ├── termination.py            # Crash / success / timeout conditions
│   ├── observations.py           # 33-dim normalized observation builder
│   └── spawner.py                # Rejection-sampled spawn positions
├── filters/ema.py                # EMA velocity filter (filter-in-the-loop)
├── vision/aruco_tracker.py       # Coordinate transforms, dropout timer, solvePnP
├── curriculum/manager.py         # Blended 8-stage curriculum transitions
├── domain_randomization/         # (Future: wind, sensor noise, battery sag, etc.)
├── utils/training_logger.py      # TensorBoard metrics and episode tracking
└── train_ppo.py                  # Training entry point

sim/
├── tello_world.sdf               # Gazebo world (desk, marker, Tello)
├── aruco_marker.png               # ArUco marker texture
└── models/                        # Gazebo model assets (meshes, SDF)

gz_transport_py/
├── CMakeLists.txt                 # Build config for pybind11 wrapper
└── gz_transport_py.cpp            # Minimal Python bindings for gz-transport12
```

## Prerequisites

- Docker and Docker Compose
- Git
- NVIDIA GPU + drivers (for Gazebo rendering)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bstahman/UAV-RL-Landing.git
cd UAV-RL-Landing
```

### 2. Build and Start the Container

```bash
sudo bash scripts/build_container.sh
docker exec -it uav_rl_container bash
```

### 3. Build gz-transport Python Bindings

From inside the container:

```bash
bash scripts/build_gz_transport.sh
```

### 4. Install the Python Package

```bash
pip install -e .
```

## Usage

### Training

```bash
# Stage 0 baseline (500k steps)
train --total-timesteps 500000 --skip-env-check

# Full curriculum training
train --total-timesteps 4000000

# With Gazebo GUI visible (requires GPU + X11)
train --gui --total-timesteps 500000 --skip-env-check

# Resume from checkpoint
train --resume latest --timesteps 1000000
train --resume 350000 --timesteps 2000000
```

Run `python3 -m rl_uav_package.train_ppo --help` for the full list of options.

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

## Training Pipeline

The agent trains through an 8-stage spatial curriculum:

| Stage | Name | Spawn Range | Angle | What the Agent Learns |
|-------|------|-------------|-------|----------------------|
| 0 | close_easy | 0.5–1.5 m | ±15° | Basic alignment and descent |
| 1–3 | close_moderate → precise | 0.5–1.5 m | ±15–25° | Tightening precision |
| 4–5 | medium | 0.5–2.5 m | ±30–40° | Longer approaches |
| 6–7 | far | 0.5–4.0 m | ±45–60° | Full operational envelope |

Stage transitions are triggered at 75% success rate over a rolling 300-episode window and blended over 1000 episodes to prevent value function collapse.

## Sim-to-Real Communication

The training environment communicates directly with Gazebo via gz-transport (Gazebo's native transport layer) through custom pybind11 bindings, bypassing the need for ROS or any middleware bridge. This reduces latency and eliminates a significant CPU bottleneck in the training loop.

## Project Status

- [x] Simulation infrastructure (Gazebo + gz-transport direct)
- [x] Gymnasium environment wrapper with modular architecture
- [x] Multi-component reward function with checkpoint-gated descent
- [x] 8-stage curriculum with blended transitions
- [x] TensorBoard logging with reward breakdowns and outcome tracking
- [x] Custom pybind11 bindings for gz-transport12 (no ROS bridge)
- [ ] MPI-parallelized training (multi-drone single-sim)
- [ ] Domain randomization (wind, sensor noise, battery sag, furniture)
- [ ] Real hardware deployment on DJI Tello
