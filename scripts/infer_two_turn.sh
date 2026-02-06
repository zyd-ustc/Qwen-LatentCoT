#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:-"A cinematic portrait of a panda astronaut walking on Mars"}
OUT_DIR=${2:-"./outputs/infer_two_turn"}

python -m qwen_latent_cot.cli infer \
  --prompt "$PROMPT" \
  --output-dir "$OUT_DIR" \
  --backend mock \
  --reflector heuristic \
  --num-inference-steps 40 \
  --guidance-scale 4.0
