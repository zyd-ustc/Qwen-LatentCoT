"""Shared constants for Qwen-LatentCoT."""

SPECIAL_TOKENS = {
    "latent_pad": "<abs_vis_token_pad>",
    "latent_start": "<abs_vis_token>",
    "latent_end": "</abs_vis_token>",
    "observation_start": "<observation>",
    "observation_end": "</observation>",
}

IGNORE_TOKEN_ID = -100
DEFAULT_DTYPE = "bfloat16"

STAGE_CHOICES = ("stage1-1", "stage1-2", "stage1-3")
