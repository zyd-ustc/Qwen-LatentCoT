"""Model adapters and wrappers."""

from .special_tokens import (
    SpecialTokenIds,
    add_latent_special_tokens,
    attach_special_ids_to_model,
    resolve_special_token_ids,
)

__all__ = [
    "SpecialTokenIds",
    "add_latent_special_tokens",
    "attach_special_ids_to_model",
    "resolve_special_token_ids",
]
