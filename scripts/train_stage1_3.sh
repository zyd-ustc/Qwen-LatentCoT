#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"/path/to/stage1_2"}
DATA_PATH=${DATA_PATH:-"./data"}
TEACHER_LATENTS=${TEACHER_LATENTS:-"./artifacts/teacher_latents"}
OUT_DIR=${OUT_DIR:-"./checkpoints/stage1_3"}
QWEN_IMAGE_EDIT_ROOT=${QWEN_IMAGE_EDIT_ROOT:-""}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-""}
DATA_PATH_CLEAN=${DATA_PATH//:/ }
read -r -a DATA_PATHS <<< "$DATA_PATH_CLEAN"
DS_ARGS=()
if [[ -n "$DEEPSPEED_CONFIG" ]]; then
  DS_ARGS=(--deepspeed "$DEEPSPEED_CONFIG")
fi

python -m qwen_latent_cot.cli train \
  --stage stage1-3 \
  --model-path "$MODEL_PATH" \
  --qwen-image-edit-root "$QWEN_IMAGE_EDIT_ROOT" \
  "${DS_ARGS[@]}" \
  --data-path "${DATA_PATHS[@]}" \
  --output-dir "$OUT_DIR" \
  --teacher-latent-dir "$TEACHER_LATENTS" \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --epochs 3 \
  --learning-rate 1e-5 \
  --latent-size 8 \
  --alignment-layer all_layers \
  --alignment-weight 1.0 \
  --mask-question-image \
  --observation-tokens-cannot-see-question-image
