"""Special-token registration and id helpers."""

from __future__ import annotations

from dataclasses import dataclass

from qwen_latent_cot.constants import SPECIAL_TOKENS


@dataclass
class SpecialTokenIds:
    latent_start: int
    latent_end: int
    latent_pad: int
    observation_start: int
    observation_end: int
    end_of_text: int
    answer_start_pattern: list[int]
    vision_start: int
    vision_end: int
    image_pad: int

    def to_mask_dict(self):
        import torch

        return {
            "v_start": torch.tensor(self.vision_start),
            "v_end": torch.tensor(self.vision_end),
            "img_pad": torch.tensor(self.image_pad),
            "abs_start": torch.tensor(self.latent_start),
            "abs_end": torch.tensor(self.latent_end),
            "abs_pad": torch.tensor(self.latent_pad),
            "obs_start": torch.tensor(self.observation_start),
            "obs_end": torch.tensor(self.observation_end),
            "ans_start": torch.tensor(self.answer_start_pattern),
        }


def add_latent_special_tokens(processor) -> int:
    tokenizer = processor.tokenizer
    before = len(tokenizer)
    tokenizer.add_tokens(SPECIAL_TOKENS["latent_pad"], special_tokens=True)
    tokenizer.add_tokens(SPECIAL_TOKENS["latent_start"], special_tokens=True)
    tokenizer.add_tokens(SPECIAL_TOKENS["latent_end"], special_tokens=True)
    tokenizer.add_tokens(SPECIAL_TOKENS["observation_start"], special_tokens=True)
    tokenizer.add_tokens(SPECIAL_TOKENS["observation_end"], special_tokens=True)
    return len(tokenizer) - before


def resolve_special_token_ids(processor) -> SpecialTokenIds:
    tokenizer = processor.tokenizer

    def token_id(text: str) -> int:
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        if ids.numel() == 1:
            return int(ids.item())
        return int(ids[-1].item())

    answer_start = tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    return SpecialTokenIds(
        latent_start=token_id(SPECIAL_TOKENS["latent_start"]),
        latent_end=token_id(SPECIAL_TOKENS["latent_end"]),
        latent_pad=token_id(SPECIAL_TOKENS["latent_pad"]),
        observation_start=token_id(SPECIAL_TOKENS["observation_start"]),
        observation_end=token_id(SPECIAL_TOKENS["observation_end"]),
        end_of_text=token_id("<|endoftext|>"),
        answer_start_pattern=[int(v.item()) for v in answer_start],
        vision_start=token_id("<|vision_start|>"),
        vision_end=token_id("<|vision_end|>"),
        image_pad=token_id("<|image_pad|>"),
    )


def attach_special_ids_to_model(model, ids: SpecialTokenIds) -> None:
    model.config.latent_token_id = int(ids.latent_pad)
    model.config.latent_start_id = int(ids.latent_start)
    model.config.latent_end_id = int(ids.latent_end)
    model.config.answer_start_pattern = list(ids.answer_start_pattern)
