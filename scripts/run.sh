#!/bin/bash

echo "--- Cleaning up old processes ---"
pkill -f tensorboard
pkill -f gz  # Kills Gazebo Garden processes

echo "--- Starting TensorBoard (Silent Background) ---"
cd /workspace/ros2_ws && tensorboard --logdir src/rl_uav_package/logs/ --bind_all > /dev/null 2>&1 &

echo "--- Starting Simulator & Bridge ---"
cd /workspace/ros2_ws
source install/setup.bash
# Run our new launch file in the background
ros2 launch rl_uav_package tello_sim.launch.py &

echo "--- Waiting 8 seconds for Gazebo to boot and download the drone ---"
sleep 8

echo "--- Starting RL Agent ---"
cd /workspace/ros2_ws
source install/setup.bash
# Run python in the foreground so Ctrl+C gracefully stops training
python3 src/rl_uav_package/rl_uav_package/train_ppo.py

# Cleanup on exit
echo -e "\n--- Shutting down background processes ---"
kill $(jobs -p) 2>/dev/null
wait 2>/dev/null
echo "Cleanup complete!"