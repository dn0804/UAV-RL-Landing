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

MODELS_DIR="$REPO_ROOT/models/ppo_landing"
LOGS_DIR="$REPO_ROOT/logs"
SIM_DIR="$REPO_ROOT/sim"
WORLD_FILE="$SIM_DIR/tello_world.sdf"
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
            TRAIN_ARGS+=("--total-timesteps" "$next")
            idx=$((idx+1))
            ;;
        --resume)
            resolved=""
            if [ "$next" = "latest" ]; then
                resolved="$MODELS_DIR/latest"
            elif [ "$next" = "interrupted" ]; then
                resolved="$MODELS_DIR/interrupted"
            elif [[ "$next" =~ ^[0-9]+$ ]]; then
                pattern="$MODELS_DIR/ppo_checkpoint_${next}_steps"
                if [ -f "${pattern}.zip" ]; then
                    resolved="$pattern"
                else
                    echo "[ERROR] No checkpoint found for step ${next}"
                    echo "        Available checkpoints:"
                    ls "$MODELS_DIR"/*.zip 2>/dev/null | sed 's/.*\//          /' || echo "          (none)"
                    exit 1
                fi
            elif [ -f "${next}.zip" ] || [ -f "$next" ]; then
                resolved="$next"
            elif [ -f "$MODELS_DIR/${next}.zip" ] || [ -f "$MODELS_DIR/$next" ]; then
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
sleep 1
rm -rf /tmp/gz-* /tmp/gazebo-* 2>/dev/null || true

# ── Set up Gazebo resource paths ─────────────────────────────

export GZ_SIM_RESOURCE_PATH="${SIM_DIR}/models:${SIM_DIR}:${GZ_SIM_RESOURCE_PATH:-}"

# ── TensorBoard ──────────────────────────────────────────────

echo "--- Starting TensorBoard ---"
mkdir -p "$LOGS_DIR"
tensorboard --logdir "$LOGS_DIR" --bind_all > /dev/null 2>&1 &
TB_PID=$!

# ── Gazebo (headless) ───────────────────────────────────────

echo "--- Starting Gazebo ---"
gz sim -s -r "$WORLD_FILE" > /dev/null 2>&1 &
SIM_PID=$!

echo "--- Waiting ${GAZEBO_BOOT_WAIT}s for Gazebo to boot ---"
sleep "$GAZEBO_BOOT_WAIT"

# ── Gazebo GUI (optional) ───────────────────────────────────

GUI_PID=""
if [ "$GUI" = true ]; then
    echo "--- Launching Gazebo GUI ---"
    gz sim -g &
    GUI_PID=$!

    echo "    Waiting 3s for GUI services to start..."
    sleep 3

    echo "    Locking camera to 'tello'"
    gz service -s /gui/follow \
        --reqtype gz.msgs.StringMsg \
        --reptype gz.msgs.Boolean \
        --timeout 2000 \
        --req 'data: "tello"' > /dev/null 2>&1 || true

    gz service -s /gui/follow/offset \
        --reqtype gz.msgs.Vector3d \
        --reptype gz.msgs.Boolean \
        --timeout 2000 \
        --req 'x: -1.0, y: 0, z: 0.5' > /dev/null 2>&1 || true
fi

# ── Training ─────────────────────────────────────────────────

echo "--- Starting RL Agent ---"
echo "    Args: ${TRAIN_ARGS[*]}"

cd "$REPO_ROOT"
python3 -m rl_uav_package.train_ppo \
    --models-dir "$MODELS_DIR" \
    --log-dir "$LOGS_DIR" \
    "${TRAIN_ARGS[@]}"
TRAIN_EXIT=$?

# ── Shutdown ─────────────────────────────────────────────────

echo ""
echo "--- Shutting down ---"
kill $SIM_PID 2>/dev/null || true
kill $TB_PID 2>/dev/null || true
[ -n "$GUI_PID" ] && kill $GUI_PID 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
wait 2>/dev/null
echo "Done."
exit $TRAIN_EXIT
