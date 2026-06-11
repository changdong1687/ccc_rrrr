#!/bin/bash
# Overfit-debug training: train on a tiny single-task (or single-episode) LIBERO
# dataset to check whether the model can learn the correct joint actions.
#
# This does NOT modify any existing code. It only sets env vars that the existing
# simulation/libero/train_libero_wan22.sh already honors, then calls it.
#
# Steps:
#   1. Build a tiny dataset (pick ONE):
#      a) one full task (~50 demos), using the existing converter:
#         python simulation/libero/convert_libero_data.py \
#           --input /path/to/LIBERO/.../<one_task>.hdf5 \
#           --output ./data/libero_debug_onetask --force
#      b) a single episode (strongest overfit), using the subsetter:
#         python simulation/libero/debug/make_debug_dataset.py \
#           --src ./data/libero_spatial --dst ./data/libero_debug_onetask \
#           --num-episodes 1 --force
#   2. Run this script.
#
# Tunables (env vars):
#   LIBERO_DATA_ROOT   tiny dataset dir (default ./data/libero_debug_onetask)
#   OUTPUT_DIR         checkpoint dir   (default ./checkpoints/dreamzero_libero_overfit_debug)
#   MAX_STEPS          training steps   (default 5000)
#   NUM_GPUS           (default 1)
#   PER_DEVICE_BS      (default 1)
#   GRADIENT_ACCUM     (default 1)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${DREAMZERO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}

export LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"$REPO_ROOT/data/libero_debug_onetask"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/checkpoints/dreamzero_libero_overfit_debug"}
export MAX_STEPS=${MAX_STEPS:-5000}
export NUM_GPUS=${NUM_GPUS:-1}
export PER_DEVICE_BS=${PER_DEVICE_BS:-1}
export GRADIENT_ACCUM=${GRADIENT_ACCUM:-1}

if [ ! -d "$LIBERO_DATA_ROOT" ]; then
    echo "ERROR: tiny dataset not found at $LIBERO_DATA_ROOT"
    echo "Build it first (see the header of this script)."
    exit 1
fi

echo "[overfit-train] data=$LIBERO_DATA_ROOT output=$OUTPUT_DIR max_steps=$MAX_STEPS gpus=$NUM_GPUS"
exec bash "$REPO_ROOT/simulation/libero/train_libero_wan22.sh"
