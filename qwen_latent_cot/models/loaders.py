"""Model loading helpers."""

from __future__ import annotations

import os

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
    base_model_path: str | None = None,
) -> tuple[AutoProcessor, Qwen2_5_VLForConditionalGeneration]:
    base_model_path = (base_model_path or "").strip() or None

    # Heuristic: a Trainer checkpoint produced by this repo usually contains `model.safetensors`
    # but does NOT contain HF `config.json` / `generation_config.json` / `text_encoder/`.
    ckpt_weights = os.path.join(model_path, "model.safetensors")
    is_trainer_ckpt = (
        os.path.isfile(ckpt_weights)
        and (not os.path.isdir(os.path.join(model_path, "text_encoder")))
        and (not os.path.isfile(os.path.join(model_path, "config.json")))
    )

    processor_path = model_path
    if os.path.isdir(os.path.join(model_path, "processor")):
        processor_path = os.path.join(model_path, "processor")
    elif is_trainer_ckpt and not os.path.isfile(os.path.join(model_path, "processor_config.json")) and base_model_path:
        # Some checkpoints may not carry processor files.
        processor_path = (
            os.path.join(base_model_path, "processor")
            if os.path.isdir(os.path.join(base_model_path, "processor"))
            else base_model_path
        )

    processor = AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    text_encoder_path = model_path
    if os.path.isdir(os.path.join(model_path, "text_encoder")):
        text_encoder_path = os.path.join(model_path, "text_encoder")
    elif is_trainer_ckpt:
        if not base_model_path:
            raise ValueError(
                "Detected a stage checkpoint directory without `text_encoder/` or `config.json`. "
                "Please pass `--qwen-image-edit-root` (or `base_model_path`) pointing to the base Qwen-Image-Edit weights "
                "so we can load a correct config."
            )
        text_encoder_path = (
            os.path.join(base_model_path, "text_encoder")
            if os.path.isdir(os.path.join(base_model_path, "text_encoder"))
            else base_model_path
        )

    config = Qwen2_5_VLConfig.from_pretrained(text_encoder_path, trust_remote_code=trust_remote_code)
    config.use_cache = False

    if not is_trainer_ckpt:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            text_encoder_path,
            config=config,
            torch_dtype=_resolve_dtype(dtype),
            trust_remote_code=trust_remote_code,
        )
        return processor, model

    # Load checkpoint weights saved from the wrapper (keys are usually prefixed by "model.").
    try:
        from safetensors.torch import load_file  # type: ignore
    except Exception as exc:
        raise ImportError("safetensors is required to load stage checkpoints.") from exc

    state_dict = load_file(ckpt_weights, device="cpu")
    if "lm_head.weight" not in state_dict and "model.lm_head.weight" in state_dict:
        state_dict = {
            (k[len("model.") :] if k.startswith("model.") else k): v for k, v in state_dict.items()
        }

    # NOTE: Some transformers versions disallow passing `state_dict` to `from_pretrained`.
    # We load the base weights first (from a real HF directory), then overwrite with the stage checkpoint.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        text_encoder_path,
        config=config,
        torch_dtype=_resolve_dtype(dtype),
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    try:
        tok_len = int(len(processor.tokenizer))
        emb = model.get_input_embeddings().weight
        if emb is not None and int(emb.shape[0]) != tok_len:
            model.resize_token_embeddings(tok_len)
    except Exception:
        pass

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # Keep behavior non-fatal; callers may still want to proceed (e.g. new special tokens).
        print(
            f"[load_qwen2_5_vl] loaded stage checkpoint with missing={len(missing)} unexpected={len(unexpected)}"
        )
    return processor, model
