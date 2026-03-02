## Installation & Setup
This project runs inside a containerized Ubuntu 22.04 environment with ROS 2 Humble, Gazebo Garden, and PX4 SITL (v1.14). All dependencies are bundled in the Docker image to ensure consistency across operating systems.

### Prerequisites
* Docker and Docker Compose
* Git

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

### 4. Verify the Simulation Stack
To confirm the flight controller and 3D physics engine are working, run the following inside the container:
Bash
```
cd /PX4-Autopilot
make px4_sitl gz_x500
```

If successful, Gazebo will open displaying an x500 quadrotor, and your terminal will show the `pxh>` flight controller prompt. Type `commander takeoff` in the terminal to execute a test flight.
