#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"qwen_latent_cot/models/Qwen-Image-Edit"}
DATA_PATH=${DATA_PATH:-"./data"}
OUT_DIR=${OUT_DIR:-"./checkpoints/stage1_1"}

python -m qwen_latent_cot.cli train \
  --stage stage1-1 \
  --model-path "$MODEL_PATH" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUT_DIR" \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --epochs 2 \
  --learning-rate 1e-5 \
  --latent-size 8
