#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:-"A cinematic portrait of a panda astronaut walking on Mars"}
OUT_DIR=${2:-"./outputs/infer_two_turn"}

# ---------- backend 选择 ----------
# mock（无权重验证）: --backend mock
# 本地权重推理:       --backend local --qwen-image-model <path>
BACKEND=${BACKEND:-"local"}
QWEN_IMAGE_MODEL=${QWEN_IMAGE_MODEL:-"qwen_latent_cot/models/Qwen-Image-Edit"}

if [ "$BACKEND" = "local" ]; then
  python -m qwen_latent_cot.cli infer \
    --prompt "$PROMPT" \
    --output-dir "$OUT_DIR" \
    --backend local \
    --qwen-image-model "$QWEN_IMAGE_MODEL" \
    --init-image zeros \
    --reflector heuristic \
    --num-inference-steps 50 \
    --guidance-scale 4.0 \
    --aspect-ratio 1:1
else
  python -m qwen_latent_cot.cli infer \
    --prompt "$PROMPT" \
    --output-dir "$OUT_DIR" \
    --backend mock \
    --reflector heuristic \
    --num-inference-steps 40 \
    --guidance-scale 4.0
fi
