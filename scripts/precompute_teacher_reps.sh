#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"/path/to/stage1_1"}
DATA_PATH=${DATA_PATH:-"./data"}
OUT_DIR=${OUT_DIR:-"./artifacts/teacher_reps"}

python -m qwen_latent_cot.cli precompute-rep \
  --model-path "$MODEL_PATH" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUT_DIR" \
  --batch-size 1 \
  --latent-size 8 \
  --output-hidden-states
