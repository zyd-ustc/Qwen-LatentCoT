#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-"./checkpoints/stage1_4"}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-""}
PROMPT=${PROMPT:-"Turn the scene into a rainy cyberpunk street at night."}
IMAGE=${IMAGE:-""}
OUT_DIR=${OUT_DIR:-"./outputs/infer_cort"}
DTYPE=${DTYPE:-bfloat16}
LATENT_SIZE=${LATENT_SIZE:-8}
MAX_CORT_TURNS=${MAX_CORT_TURNS:-3}
MAX_REFLECTION_TOKENS=${MAX_REFLECTION_TOKENS:-128}
MIN_REFLECTION_TOKENS=${MIN_REFLECTION_TOKENS:-1}
RUN_INFER=${RUN_INFER:-1}

if [[ "$RUN_INFER" == "1" ]]; then
  CMD=(
    python -m qwen_latent_cot.cli infer-cort
    --model-path "$MODEL_PATH"
    --prompt "$PROMPT"
    --output-dir "$OUT_DIR"
    --dtype "$DTYPE"
    --latent-size "$LATENT_SIZE"
    --max-cort-turns "$MAX_CORT_TURNS"
    --max-reflection-tokens "$MAX_REFLECTION_TOKENS"
    --min-reflection-tokens "$MIN_REFLECTION_TOKENS"
  )
  if [[ -n "$BASE_MODEL_PATH" ]]; then
    CMD+=(--base-model-path "$BASE_MODEL_PATH")
  fi
  if [[ -n "$IMAGE" ]]; then
    CMD+=(--image "$IMAGE")
  fi
  "${CMD[@]}"
fi

python scripts/validate_infer_cort.py \
  --output-dir "$OUT_DIR" \
  --latent-size "$LATENT_SIZE"
