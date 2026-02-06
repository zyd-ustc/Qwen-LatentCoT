#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"/path/to/stage1_1"}
DATA_PATH=${DATA_PATH:-"/path/to/train.jsonl"}
TEACHER_REPS=${TEACHER_REPS:-"./artifacts/teacher_reps"}
OUT_DIR=${OUT_DIR:-"./checkpoints/stage1_2"}

python -m qwen_latent_cot.cli train \
  --stage stage1-2 \
  --model-path "$MODEL_PATH" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUT_DIR" \
  --teacher-reps-dir "$TEACHER_REPS" \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --epochs 3 \
  --learning-rate 1e-5 \
  --latent-size 8 \
  --alignment-layer all_layers \
  --alignment-weight 1.0 \
  --mask-question-image \
  --observation-tokens-cannot-see-question-image
