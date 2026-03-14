#!/bin/bash
set -e

# ──────────────────────────────────────────────────────────────
# UAV RL Landing — Training launcher
#
# Usage:
#   ./scripts/run.sh                          # Stage 1, 500k steps
#   ./scripts/run.sh --gui                    # same, with Gazebo viewer
#   ./scripts/run.sh --stage 2 --total-timesteps 1000000
#   ./scripts/run.sh --resume models/ppo_landing/latest.zip
#   ./scripts/run.sh --skip-env-check         # faster restart
#
# The --gui flag is consumed by this script.
# All other arguments are forwarded to train_ppo.py.
# ──────────────────────────────────────────────────────────────

# ── Parse --gui flag (consume it, pass everything else through) ──

GUI=false
TRAIN_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--gui" ]; then
        GUI=true
    else
        TRAIN_ARGS+=("$arg")
    fi
done

# Derive workspace path relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WS="$REPO_ROOT/ros2_ws"
GAZEBO_BOOT_WAIT=8

# ── Cleanup old processes ────────────────────────────────────

echo "--- Cleaning up old processes ---"
pkill -f tensorboard 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f parameter_bridge 2>/dev/null || true
sleep 1

# ── Source workspace ─────────────────────────────────────────

cd "$WS"
source install/setup.bash

# ── TensorBoard ──────────────────────────────────────────────

echo "--- Starting TensorBoard ---"
tensorboard --logdir logs/ --bind_all > /dev/null 2>&1 &
TB_PID=$!

# ── Gazebo + ROS bridge ─────────────────────────────────────

echo "--- Starting Simulator & Bridge ---"
ros2 launch rl_uav_package tello_sim.launch.py &
SIM_PID=$!

echo "--- Waiting ${GAZEBO_BOOT_WAIT}s for Gazebo to boot ---"
sleep "$GAZEBO_BOOT_WAIT"

# ── Gazebo GUI (optional) ───────────────────────────────────

GUI_PID=""
if [ "$GUI" = true ]; then
    echo "--- Launching Gazebo GUI ---"
    gz sim -g &
    GUI_PID=$!
fi

# ── Training ─────────────────────────────────────────────────

echo "--- Starting RL Agent ---"
echo "    Args: ${TRAIN_ARGS[*]}"

# Run in foreground so Ctrl+C triggers KeyboardInterrupt in Python,
# which saves an interrupted checkpoint before exiting.
python3 -m rl_uav_package.train_ppo "${TRAIN_ARGS[@]}"
TRAIN_EXIT=$?

# ── Shutdown ─────────────────────────────────────────────────

echo ""
echo "--- Shutting down background processes ---"
kill $SIM_PID 2>/dev/null || true
kill $TB_PID 2>/dev/null || true
[ -n "$GUI_PID" ] && kill $GUI_PID 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f parameter_bridge 2>/dev/null || true
wait 2>/dev/null
echo "Cleanup complete!"
exit $TRAIN_EXIT