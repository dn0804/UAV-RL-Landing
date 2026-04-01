#!/bin/bash
set -e

# ──────────────────────────────────────────────────────────────
# UAV RL Landing — Training launcher
#
# Usage:
#   train                                     # Fresh Stage 0, 500k steps
#   train --gui                               # Same, with Gazebo viewer
#   train --stage 2 --timesteps 1000000       # Fresh from stage 2
#   train --resume 650000 --timesteps 5000000 # Resume from checkpoint 650k
#   train --resume latest                     # Resume from latest.zip
#   train --resume interrupted                # Resume from interrupted.zip
#   train --skip-env-check                    # Faster startup
#
# The --gui and --timesteps flags are consumed by this script.
# --resume accepts: a step number, "latest", "interrupted", or a full path.
# All other arguments are forwarded to train_ppo.py.
# ──────────────────────────────────────────────────────────────

# Derive workspace path relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WS="$REPO_ROOT/ros2_ws"
MODELS_DIR="$WS/models/ppo_landing"
GAZEBO_BOOT_WAIT=8

# ── Parse arguments ─────────────────────────────────────────

GUI=false
TRAIN_ARGS=()

args=("$@")
idx=0
while [ $idx -lt ${#args[@]} ]; do
    arg="${args[$idx]}"
    next="${args[$((idx+1))]:-}"

    case "$arg" in
        --gui)
            GUI=true
            ;;
        --timesteps)
            # Alias for --total-timesteps
            TRAIN_ARGS+=("--total-timesteps" "$next")
            idx=$((idx+1))
            ;;
        --resume)
            # Resolve shorthand checkpoint references
            resolved=""
            if [ "$next" = "latest" ]; then
                resolved="$MODELS_DIR/latest"
            elif [ "$next" = "interrupted" ]; then
                resolved="$MODELS_DIR/interrupted"
            elif [[ "$next" =~ ^[0-9]+$ ]]; then
                # Numeric — find matching checkpoint
                pattern="$MODELS_DIR/ppo_checkpoint_${next}_steps"
                if [ -f "${pattern}.zip" ]; then
                    resolved="$pattern"
                else
                    # Try dppo checkpoint format
                    pattern2="$MODELS_DIR/dppo_${next}"
                    if [ -f "${pattern2}.zip" ]; then
                        resolved="$pattern2"
                    else
                        echo "[ERROR] No checkpoint found for step ${next}"
                        echo "        Tried: ${pattern}.zip"
                        echo "        Tried: ${pattern2}.zip"
                        echo "        Available checkpoints:"
                        ls "$MODELS_DIR"/*.zip 2>/dev/null | sed 's/.*\//          /' || echo "          (none)"
                        exit 1
                    fi
                fi
            elif [ -f "${next}.zip" ] || [ -f "$next" ]; then
                # Full path provided
                resolved="$next"
            elif [ -f "$MODELS_DIR/${next}.zip" ] || [ -f "$MODELS_DIR/$next" ]; then
                # Name without directory
                resolved="$MODELS_DIR/$next"
            else
                echo "[ERROR] Cannot resolve checkpoint: $next"
                echo "        Available checkpoints:"
                ls "$MODELS_DIR"/*.zip 2>/dev/null | sed 's/.*\//          /' || echo "          (none)"
                exit 1
            fi
            echo "    Resolved checkpoint: ${resolved}.zip"
            TRAIN_ARGS+=("--resume" "$resolved")
            idx=$((idx+1))
            ;;
        *)
            TRAIN_ARGS+=("$arg")
            ;;
    esac
    idx=$((idx+1))
done

# ── Cleanup old processes ────────────────────────────────────

echo "--- Cleaning up old processes ---"
pkill -9 -f tensorboard 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f parameter_bridge 2>/dev/null || true
sleep 1
rm -rf /tmp/gz-* /tmp/gazebo-* 2>/dev/null || true

# ── Rebuild package ─────────────────────────────────────────

echo "--- Rebuilding rl_uav_package ---"
cd "$WS"
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
colcon build --packages-select rl_uav_package 2>&1 | tail -2
echo "    Build complete."

# ── Source workspace ─────────────────────────────────────────

source install/setup.bash

# ── TensorBoard ──────────────────────────────────────────────

echo "--- Starting TensorBoard ---"
tensorboard --logdir logs/ --bind_all > /dev/null 2>&1 &
TB_PID=$!

# ── Gazebo + ROS bridge ─────────────────────────────────────

echo "--- Starting Simulator & Bridge ---"

# Tell Gazebo exactly where to find your new local models!
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$WS/src/rl_uav_package/models

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
    
    # Wait for the GUI process to initialize its service endpoints
    echo "    Waiting 3s for GUI services to start..."
    sleep 3
    
    echo "    Locking camera to 'tello'"
    
    # 1. Initiate Follow Mode (target: tello)
    gz service -s /gui/follow \
        --reqtype gz.msgs.StringMsg \
        --reptype gz.msgs.Boolean \
        --timeout 2000 \
        --req 'data: "tello"' > /dev/null 2>&1 || true
        
    # 2. Lock the fixed 3rd-person offset
    gz service -s /gui/follow/offset \
        --reqtype gz.msgs.Vector3d \
        --reptype gz.msgs.Boolean \
        --timeout 2000 \
        --req 'x: -1.5, y: 0, z: 0.5' > /dev/null 2>&1 || true
fi

# ── Training ─────────────────────────────────────────────────

echo "--- Starting RL Agent ---"
echo "    Args: ${TRAIN_ARGS[*]}"

python3 -m rl_uav_package.train_ppo "${TRAIN_ARGS[@]}"
TRAIN_EXIT=$?

# ── Shutdown ─────────────────────────────────────────────────

echo ""
echo "--- Shutting down background processes ---"
kill $SIM_PID 2>/dev/null || true
kill $TB_PID 2>/dev/null || true
[ -n "$GUI_PID" ] && kill $GUI_PID 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f parameter_bridge 2>/dev/null || true
wait 2>/dev/null
echo "Cleanup complete!"
exit $TRAIN_EXIT