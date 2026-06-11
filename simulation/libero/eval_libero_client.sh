#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${DREAMZERO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}

CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-"$REPO_ROOT/checkpoints/dreamzero_libero_wan22_lora"}
LIBERO_ROOT=${LIBERO_ROOT:-"/inspire/hdd/project/realtimedecisionmaking/chentao-25011/surd/codes/LIBERO"}
SERVER_HOST=${SERVER_HOST:-"localhost"}
SERVER_PORT=${SERVER_PORT:-8000}
BENCHMARK_NAME=${BENCHMARK_NAME:-"libero_spatial"}
TASK_IDS=${TASK_IDS:-}
N_EVAL=${N_EVAL:-20}
MAX_STEPS=${MAX_STEPS:-500}
OPEN_LOOP_HORIZON=${OPEN_LOOP_HORIZON:-8}
CAMERA_HEIGHT=${CAMERA_HEIGHT:-160}
CAMERA_WIDTH=${CAMERA_WIDTH:-320}
TASK_ORDER_INDEX=${TASK_ORDER_INDEX:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/runs/libero_spatial_eval"}
SAVE_VIDEO=${SAVE_VIDEO:-false}
SAVE_VIDEO_PRED=${SAVE_VIDEO_PRED:-false}
VIDEO_EPISODES_PER_TASK=${VIDEO_EPISODES_PER_TASK:-1}
VIDEO_FPS=${VIDEO_FPS:-20}
DEBUG_OPEN_LOOP=${DEBUG_OPEN_LOOP:-false}
RESET_SERVER_EACH_REQUEST=${RESET_SERVER_EACH_REQUEST:-false}
# Joint-position control / gripper / settle knobs (see eval_libero_client.py).
CONTROLLER=${CONTROLLER:-"JOINT_POSITION"}
JOINT_CONTROL_MODE=${JOINT_CONTROL_MODE:-"delta"}
JOINT_DELTA_BOUND=${JOINT_DELTA_BOUND:-1.0}
JOINT_KP=${JOINT_KP:-}
GRIPPER_THRESHOLD=${GRIPPER_THRESHOLD:-0.02}
GRIPPER_INVERT=${GRIPPER_INVERT:-false}
NUM_SETTLE_STEPS=${NUM_SETTLE_STEPS:-10}

if [ -z "$CHECKPOINT_PATH" ]; then
    LATEST_CHECKPOINT=$(find "$TRAIN_OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        CHECKPOINT_PATH="$LATEST_CHECKPOINT"
    fi
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: CHECKPOINT_PATH is not set and no checkpoint-* directory was found under $TRAIN_OUTPUT_DIR"
    exit 1
fi
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: checkpoint directory not found at $CHECKPOINT_PATH"
    exit 1
fi
if [ ! -d "$LIBERO_ROOT" ]; then
    echo "ERROR: LIBERO_ROOT not found at $LIBERO_ROOT"
    exit 1
fi

CLIENT_ARGS=(
    simulation/libero/eval_libero_client.py
    --libero-root "$LIBERO_ROOT"
    --host "$SERVER_HOST"
    --port "$SERVER_PORT"
    --benchmark-name "$BENCHMARK_NAME"
    --task-order-index "$TASK_ORDER_INDEX"
    --n-eval "$N_EVAL"
    --max-steps "$MAX_STEPS"
    --open-loop-horizon "$OPEN_LOOP_HORIZON"
    --camera-height "$CAMERA_HEIGHT"
    --camera-width "$CAMERA_WIDTH"
    --checkpoint-path "$CHECKPOINT_PATH"
    --output-dir "$OUTPUT_DIR"
    --video-episodes-per-task "$VIDEO_EPISODES_PER_TASK"
    --video-fps "$VIDEO_FPS"
    --controller "$CONTROLLER"
    --joint-control-mode "$JOINT_CONTROL_MODE"
    --joint-delta-bound "$JOINT_DELTA_BOUND"
    --gripper-threshold "$GRIPPER_THRESHOLD"
    --num-settle-steps "$NUM_SETTLE_STEPS"
)

if [ -n "$JOINT_KP" ]; then
    CLIENT_ARGS+=(--joint-kp "$JOINT_KP")
fi
if [ "$GRIPPER_INVERT" = "true" ]; then
    CLIENT_ARGS+=(--gripper-invert)
fi

if [ "$SAVE_VIDEO" = "true" ]; then
    CLIENT_ARGS+=(--save-video)
fi
if [ "$SAVE_VIDEO_PRED" = "true" ]; then
    CLIENT_ARGS+=(--save-video-pred)
fi
if [ "$DEBUG_OPEN_LOOP" = "true" ]; then
    CLIENT_ARGS+=(--debug-open-loop)
fi
if [ "$RESET_SERVER_EACH_REQUEST" = "true" ]; then
    CLIENT_ARGS+=(--reset-server-each-request)
fi
if [ -n "$TASK_IDS" ]; then
    # shellcheck disable=SC2206
    TASK_ID_ARRAY=($TASK_IDS)
    CLIENT_ARGS+=(--task-ids "${TASK_ID_ARRAY[@]}")
fi
if [ "$#" -gt 0 ]; then
    CLIENT_ARGS+=("$@")
fi

cd "$REPO_ROOT"
echo "Starting LIBERO eval client benchmark=${BENCHMARK_NAME} server=${SERVER_HOST}:${SERVER_PORT} output=${OUTPUT_DIR}"
python "${CLIENT_ARGS[@]}"
