"""Shared constants for Qwen-LatentCoT."""

SPECIAL_TOKENS = {
    # CoRT-format special tokens (aligned with SPEC.md).
    "cort_start": "<|cort_start|>",
    "cort_end": "<|cort_end|>",
    "latent_pad": "<|vlat_pad|>",
    "latent_start": "<|vlat_start|>",
    "latent_end": "<|vlat_end|>",
    "observation_start": "<|refl_start|>",
    "observation_end": "<|refl_end|>",
}

IGNORE_TOKEN_ID = -100
DEFAULT_DTYPE = "bfloat16"

STAGE_CHOICES = ("stage1-1", "stage1-2", "stage1-3")
