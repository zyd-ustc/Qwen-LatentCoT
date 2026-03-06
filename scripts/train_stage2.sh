#!/usr/bin/env bash
set -euo pipefail

QWEN_IMAGE_MODEL_PATH=${QWEN_IMAGE_MODEL_PATH:-"/data1/weights/Qwen-Image"}
DATA_PATH=${DATA_PATH:-"/data0/data/cort_kturn/intermediate"}
OUT_DIR=${OUT_DIR:-"/data2/checkpoints/stage2_e2e"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"/home/ydzhi/Qwen-LatentCoT/configs/stage2/deepspeed_zero2_bf16.json"}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

DATA_PATH_CLEAN=${DATA_PATH//:/ }
read -r -a DATA_PATHS <<< "$DATA_PATH_CLEAN"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" torchrun --nproc_per_node="$(echo "$CUDA_DEVICES" | awk -F',' '{print NF}')" -m qwen_latent_cot.cli train-stage2 \
  --qwen-image-model-path "$QWEN_IMAGE_MODEL_PATH" \
  --data-path "${DATA_PATHS[@]}" \
  --output-dir "$OUT_DIR" \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --epochs 1 \
  --learning-rate 1e-5 \
  --image-size 512 \
  --latent-token-repeat 8 \
  --diffusion-weight 1.0 \
  --recon-weight 0.1 \
  --deepspeed "$DEEPSPEED_CONFIG"
