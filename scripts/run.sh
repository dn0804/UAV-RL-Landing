#!/bin/bash

echo "--- Cleaning up old processes ---"
pkill -f MicroXRCEAgent
pkill -f tensorboard
pkill -f px4

echo "--- Starting DDS Agent & TensorBoard (Silent Background) ---"
MicroXRCEAgent udp4 -p 8888 > /dev/null 2>&1 &
cd /workspace/ros2_ws && tensorboard --logdir src/rl_uav_package/logs/ --bind_all > /dev/null 2>&1 &

echo "--- Starting PX4 & Gazebo ---"
cd /PX4-Autopilot
# We run PX4 in the background but allow its text to print to this terminal
make px4_sitl gz_x500 &

echo "--- Waiting 12 seconds for PX4 to boot ---"
sleep 12

echo "--- Starting RL Agent ---"
cd /workspace/ros2_ws
source install/setup.bash
# Run python in the foreground so Ctrl+C gracefully stops training and saves the model
python3 src/rl_uav_package/rl_uav_package/train_ppo.py

# When the python script finishes (or you hit Ctrl+C), the script reaches this point
echo -e "\n--- Shutting down simulator and background processes ---"
kill $(jobs -p) 2>/dev/null
wait 2>/dev/null
echo "Cleanup complete!"