#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$REPO_ROOT/.venv/bin/python"

if [ -x "$DEFAULT_PYTHON" ]; then
    PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/nuscenes/recondrive.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/recondrive_stage2.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/work_dirs/recondrive_stage2_eval_output}"
DATA_PATH="${DATA_PATH:-}"
VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-}"

DEVICE="${DEVICE:-0}"
EVAL_RESOLUTION="${EVAL_RESOLUTION:-280x518}"
FRAME_SKIP="${FRAME_SKIP:-6}"
MAX_SAMPLES_PER_SCENE="${MAX_SAMPLES_PER_SCENE:-}"

# SAVE_NOVEL_RENDERS for left and right moving
NOVEL_VIEW_DISTANCES="${NOVEL_VIEW_DISTANCES:-1.0,2.0,3.0}"
SAVE_NOVEL_RENDERS="${SAVE_NOVEL_RENDERS:-false}"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 1
fi

if [ ! -e "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

inference_args=(
    -m scripts.inference
    --cfg_path="$CONFIG_PATH"
    --restore_ckpt="$CHECKPOINT_PATH"
    --output_dir="$OUTPUT_DIR"
    --device="$DEVICE"
    --frame_skip="$FRAME_SKIP"
    --novel_distances="$NOVEL_VIEW_DISTANCES"
    --eval_resolution="$EVAL_RESOLUTION"
)

if [ -n "$DATA_PATH" ]; then
    inference_args+=(--data_path="$DATA_PATH")
fi

if [ -n "$VGGT_CHECKPOINT" ]; then
    inference_args+=(--vggt_checkpoint="$VGGT_CHECKPOINT")
fi

if [ -n "$SAM2_CHECKPOINT" ]; then
    inference_args+=(--sam2_checkpoint="$SAM2_CHECKPOINT")
fi

if [ -n "$MAX_SAMPLES_PER_SCENE" ]; then
    inference_args+=(--max_samples_per_scene="$MAX_SAMPLES_PER_SCENE")
fi

if [ "$SAVE_NOVEL_RENDERS" = "false" ]; then
    inference_args+=(--no_renders)
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" "${inference_args[@]}"
