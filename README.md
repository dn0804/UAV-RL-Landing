## Installation & Setup
This project runs inside a containerized Ubuntu 22.04 environment with ROS 2 Humble, Gazebo Garden, and PX4 SITL (v1.14). All dependencies are bundled in the Docker image.

### Prerequisites
* Docker and Git

### 1. Clone the Repository
```
git clone https://github.com/bstahman/UAV-RL-Landing.git
cd UAV-RL-Landing
```

### 2. Build and Start the Container
On Linux:
```
sudo bash scripts/build_container.sh
```

On Windows (PowerShell Admin):
```
.\scripts\start_uav.ps1
```

### 3. Enter the Workspace
Once the container is running in the background, open a terminal inside it:
```
docker exec -it uav_rl_container bash
```

Note: Your local repository folder is mounted to `/workspace` inside the container. Any code you edit on your host machine will instantly update inside the container.

### 4. Run the Simulation Stack
To run the stack, run the startup script `run.sh` from within the docker container:
```
bash scripts/run.sh
```
