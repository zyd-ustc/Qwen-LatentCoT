#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"/data1/weights/Qwen-Image-Edit"}
DATA_PATH=${DATA_PATH:-"/data0/data/1turn-reflect/intermediate /data0/data/cot_triplet/intermediate"}
OUT_DIR=${OUT_DIR:-"/data2/checkpoints/stage1_1"}
# Qwen2.5-VL vision inputs are incompatible with torch DataParallel in this setup.
# Default to single visible GPU unless user explicitly overrides CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
QWEN_IMAGE_EDIT_ROOT=${QWEN_IMAGE_EDIT_ROOT:-""}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-""}
# Support either whitespace-separated or colon-separated lists.
DATA_PATH_CLEAN=${DATA_PATH//:/ }
read -r -a DATA_PATHS <<< "$DATA_PATH_CLEAN"
DS_ARGS=()
if [[ -n "$DEEPSPEED_CONFIG" ]]; then
  DS_ARGS=(--deepspeed "$DEEPSPEED_CONFIG")
fi

python -m qwen_latent_cot.cli train \
  --stage stage1-1 \
  --model-path "$MODEL_PATH" \
  --qwen-image-edit-root "$QWEN_IMAGE_EDIT_ROOT" \
  "${DS_ARGS[@]}" \
  --data-path "${DATA_PATHS[@]}" \
  --output-dir "$OUT_DIR" \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --epochs 2 \
  --learning-rate 1e-5 \
  --latent-size 8
