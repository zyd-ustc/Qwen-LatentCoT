#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"./checkpoints/stage1_3"}
DATA_PATH=${DATA_PATH:-"./data"}
TEACHER_LATENTS=${TEACHER_LATENTS:-"./artifacts/teacher_latents"}
OUT_DIR=${OUT_DIR:-"./checkpoints/stage1_4"}
QWEN_IMAGE_EDIT_ROOT=${QWEN_IMAGE_EDIT_ROOT:-""}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-""}

BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
LATENT_SIZE=${LATENT_SIZE:-8}
ALIGNMENT_LAYER=${ALIGNMENT_LAYER:-all_layers}
ALIGNMENT_WEIGHT=${ALIGNMENT_WEIGHT:-1.0}
STRUCTURE_CE_WEIGHT=${STRUCTURE_CE_WEIGHT:-1.0}
CE_EMPHASIZE_FACTOR=${CE_EMPHASIZE_FACTOR:-1.0}
IMAGE_RESIZE=${IMAGE_RESIZE:-global}
DATA_PATH_CLEAN=${DATA_PATH//:/ }

read -r -a DATA_PATHS <<< "$DATA_PATH_CLEAN"

DS_ARGS=()
if [[ -n "$DEEPSPEED_CONFIG" ]]; then
  DS_ARGS=(--deepspeed "$DEEPSPEED_CONFIG")
fi

python -m qwen_latent_cot.cli train \
  --stage stage1-4 \
  --model-path "$MODEL_PATH" \
  --qwen-image-edit-root "$QWEN_IMAGE_EDIT_ROOT" \
  "${DS_ARGS[@]}" \
  --data-path "${DATA_PATHS[@]}" \
  --output-dir "$OUT_DIR" \
  --teacher-latent-dir "$TEACHER_LATENTS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --latent-size "$LATENT_SIZE" \
  --image-resize "$IMAGE_RESIZE" \
  --alignment-layer "$ALIGNMENT_LAYER" \
  --alignment-weight "$ALIGNMENT_WEIGHT" \
  --stage1-4-structure-ce-weight "$STRUCTURE_CE_WEIGHT" \
  --ce-emphasize-factor "$CE_EMPHASIZE_FACTOR"
