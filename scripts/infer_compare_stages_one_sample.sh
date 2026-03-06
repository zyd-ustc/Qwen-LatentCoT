#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-"/data0/data/cort_kturn/intermediate"}
OUT_DIR=${OUT_DIR:-"./outputs/compare_one_sample"}
STAGES=${STAGES:-"untrained stage1-1 stage1-2 stage1-3 stage2"}

QWEN_IMAGE_BASE_MODEL=${QWEN_IMAGE_BASE_MODEL:-"/data1/weights/Qwen-Image"}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-"/data2/checkpoints/stage2_e2e"}

REFLECTOR_BASE_MODEL=${REFLECTOR_BASE_MODEL:-"/data1/weights/Qwen-Image-Edit"}
STAGE1_1_CHECKPOINT=${STAGE1_1_CHECKPOINT:-"/data2/checkpoints/stage1_1"}
STAGE1_2_CHECKPOINT=${STAGE1_2_CHECKPOINT:-"/data2/checkpoints/stage1_2"}
STAGE1_3_CHECKPOINT=${STAGE1_3_CHECKPOINT:-"/data2/checkpoints/stage1_3"}

UNTRAINED_REFLECTOR_MODEL=${UNTRAINED_REFLECTOR_MODEL:-"$REFLECTOR_BASE_MODEL"}
STAGE2_REFLECTOR_CHECKPOINT=${STAGE2_REFLECTOR_CHECKPOINT:-"$STAGE1_3_CHECKPOINT"}
SAMPLE_SEED=${SAMPLE_SEED:-}

CMD=(
  python -m qwen_latent_cot.cli infer-compare-stages
  --data-path "$DATA_PATH"
  --output-dir "$OUT_DIR"
  --stages ${STAGES}
  --qwen-image-base-model "$QWEN_IMAGE_BASE_MODEL"
  --stage2-checkpoint "$STAGE2_CHECKPOINT"
  --reflector-base-model "$REFLECTOR_BASE_MODEL"
  --stage1-1-checkpoint "$STAGE1_1_CHECKPOINT"
  --stage1-2-checkpoint "$STAGE1_2_CHECKPOINT"
  --stage1-3-checkpoint "$STAGE1_3_CHECKPOINT"
  --untrained-reflector-model "$UNTRAINED_REFLECTOR_MODEL"
  --stage2-reflector-checkpoint "$STAGE2_REFLECTOR_CHECKPOINT"
  --gen-seed 42
  --num-inference-steps 50
  --guidance-scale 4.0
  --aspect-ratio 1:1
  --dtype bfloat16
  --init-image zeros
)

if [ -n "$SAMPLE_SEED" ]; then
  CMD+=(--sample-seed "$SAMPLE_SEED")
fi

"${CMD[@]}"
