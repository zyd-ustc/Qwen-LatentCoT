"""Reflection generators based on Qwen2.5-VL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from qwen_latent_cot.prompts import REFLECTION_SYSTEM_PROMPT


@dataclass
class HeuristicReflector:
    """Weight-free fallback reflector for dry runs."""

    template: str = (
        "<observation>"
        "The draft image may be missing fine-grained details or exact style alignment. "
        "Refine object attributes, spatial relations, and color consistency to better match: {goal}."
        "</observation>"
    )

    def reflect(self, prompt: str, image: Image.Image) -> str:
        del image
        return self.template.format(goal=prompt.strip())


class QwenVLReflector:
    """Qwen2.5-VL based reflection generator."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @torch.inference_mode()
    def reflect(self, prompt: str, image: Image.Image) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": REFLECTION_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Goal: {prompt}"},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        generated = self.model.generate(
            **model_inputs,
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

        decoded = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        if "<|im_start|>assistant" in decoded:
            decoded = decoded.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in decoded:
            decoded = decoded.split("<|im_end|>")[0]

        decoded = decoded.strip()
        if "<observation>" not in decoded:
            decoded = f"<observation>{decoded}</observation>"
        return decoded
