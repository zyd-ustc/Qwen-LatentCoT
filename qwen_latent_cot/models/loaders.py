"""Model loading helpers."""

from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


def _resolve_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[key]


def load_qwen2_5_vl(
    model_path: str,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
) -> tuple[AutoProcessor, Qwen2_5_VLForConditionalGeneration]:
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    config = Qwen2_5_VLConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    config.use_cache = False

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=_resolve_dtype(dtype),
        trust_remote_code=trust_remote_code,
    )

    return processor, model
