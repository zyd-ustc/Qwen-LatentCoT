#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"/path/to/stage1_2"}
DATA_PATH=${DATA_PATH:-"/path/to/train.jsonl"}
TEACHER_LATENTS=${TEACHER_LATENTS:-"./artifacts/teacher_latents"}
OUT_DIR=${OUT_DIR:-"./checkpoints/stage1_3"}

python -m qwen_latent_cot.cli train \
  --stage stage1-3 \
  --model-path "$MODEL_PATH" \
  --data-path "$DATA_PATH" \
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
