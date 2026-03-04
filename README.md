## Installation & Setup
This project runs inside a containerized Ubuntu 22.04 environment with ROS 2 Humble, Gazebo Garden, and a simulated DJI Tello stack via `ros_gz_bridge`. All dependencies are bundled in the Docker image to ensure consistency across operating systems and hardware.

### Prerequisites
* Docker and Docker Compose
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/bstahman/UAV-RL-Landing.git
cd UAV-RL-Landing
```

### 2. Build and Start the Container

**On Linux:**

```bash
sudo bash scripts/build_container.sh
```

**On Windows (PowerShell Admin):**

```powershell
.\scripts\start_uav.ps1
```

### 3. Enter the Workspace
Once the container is running in the background, open a terminal inside it:

```bash
docker exec -it uav_rl_container bash
```

> **Note:** Your local repository folder is mounted to `/workspace` inside the container. Any code you edit on your host machine will instantly update inside the container.

### 4. Build the ROS 2 Workspace
Before running the simulation for the first time, or after making structural changes to the package, you must compile the ROS 2 workspace.

From inside the container, navigate to the workspace and build:

```bash
cd /workspace/ros2_ws
colcon build --packages-select rl_uav_package --symlink-install
```

**When do you need to rebuild?** Because we use the `--symlink-install` flag, you do not need to rebuild if you are only editing Python scripts (like `drone_env.py` or `train_ppo.py`). However, you must run the build command again if you modify:

* `package.xml` (Adding or removing dependencies)
* `setup.py`
* Any `.launch.py` files (like `tello_sim.launch.py`)
* Any Gazebo `.sdf` world files

### 5. Run the Simulation Stack
To launch Gazebo, spawn the Tello drone, and start the Reinforcement Learning training loop, run the startup script from the root of the workspace:

```bash
bash scripts/run.sh
```
