#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"/path/to/stage1_1"}
DATA_PATH=${DATA_PATH:-"./data"}
OUT_DIR=${OUT_DIR:-"./artifacts/teacher_reps"}
QWEN_IMAGE_EDIT_ROOT=${QWEN_IMAGE_EDIT_ROOT:-""}
# Support either whitespace-separated or colon-separated lists.
DATA_PATH_CLEAN=${DATA_PATH//:/ }
read -r -a DATA_PATHS <<< "$DATA_PATH_CLEAN"

python -m qwen_latent_cot.cli precompute-rep \
  --model-path "$MODEL_PATH" \
  --qwen-image-edit-root "$QWEN_IMAGE_EDIT_ROOT" \
  --data-path "${DATA_PATHS[@]}" \
  --output-dir "$OUT_DIR" \
  --batch-size 1 \
  --latent-size 8 \
  --output-hidden-states
