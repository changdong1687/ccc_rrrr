#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${DREAMZERO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}

MODEL_PATH=${MODEL_PATH:-}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-"$REPO_ROOT/checkpoints/dreamzero_libero_wan22_lora_smoke_gbs16/checkpoint-9500"}
BIND_HOST=${BIND_HOST:-"0.0.0.0"}
BIND_PORT=${BIND_PORT:-8000}
DEVICE=${DEVICE:-"cuda:0"}
DEBUG_OPEN_LOOP=${DEBUG_OPEN_LOOP:-false}
NUM_GPUS=${NUM_GPUS:-1}
MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-}
NUM_FRAME_PER_BLOCK=${NUM_FRAME_PER_BLOCK:-2}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-43200}

if [ -z "$MODEL_PATH" ]; then
    LATEST_CHECKPOINT=$(find "$TRAIN_OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        MODEL_PATH="$LATEST_CHECKPOINT"
    fi
fi

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH is not set and no checkpoint-* directory was found under $TRAIN_OUTPUT_DIR"
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model checkpoint directory not found at $MODEL_PATH"
    exit 1
fi

SERVER_ARGS=(
    simulation/libero/eval_libero_server.py
    --model-path "$MODEL_PATH"
    --host "$BIND_HOST"
    --port "$BIND_PORT"
    --device "$DEVICE"
    --timeout-seconds "$TIMEOUT_SECONDS"
)

if [ "$DEBUG_OPEN_LOOP" = "true" ]; then
    SERVER_ARGS+=(--debug-open-loop)
fi
if [ -n "$MAX_CHUNK_SIZE" ]; then
    SERVER_ARGS+=(--max-chunk-size "$MAX_CHUNK_SIZE")
fi
if [ -n "$NUM_FRAME_PER_BLOCK" ]; then
    SERVER_ARGS+=(--num-frame-per-block "$NUM_FRAME_PER_BLOCK")
fi

cd "$REPO_ROOT"
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Starting multi-GPU LIBERO policy server on ${BIND_HOST}:${BIND_PORT} with $NUM_GPUS GPUs"
    torchrun --nproc_per_node "$NUM_GPUS" --standalone "${SERVER_ARGS[@]}"
else
    echo "Starting single-GPU LIBERO policy server on ${DEVICE} at ${BIND_HOST}:${BIND_PORT}"
    python "${SERVER_ARGS[@]}"
fi
