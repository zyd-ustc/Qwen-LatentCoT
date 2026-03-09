"""Autoregressive CoRT generation for stage1-4 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from qwen_latent_cot.models import (
    add_latent_special_tokens,
    attach_special_ids_to_model,
    resolve_special_token_ids,
)
from qwen_latent_cot.models.loaders import load_qwen2_5_vl
from qwen_latent_cot.utils import save_json


@dataclass
class CoRTGenerationResult:
    prompt: str
    cort_text: str
    reflections: list[str]
    latent_blocks: list[torch.Tensor]
    turn_count: int


class AutoRegressiveCoRTGenerator:
    def __init__(
        self,
        model_path: str,
        base_model_path: str | None = None,
        dtype: str = "bfloat16",
        latent_size: int = 8,
        temperature: float = 0.0,
        device: str | None = None,
    ) -> None:
        self.processor, self.model = load_qwen2_5_vl(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            base_model_path=base_model_path,
        )
        add_latent_special_tokens(self.processor)
        try:
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
        except Exception:
            pass

        self.token_ids = resolve_special_token_ids(self.processor)
        attach_special_ids_to_model(self.model, self.token_ids)
        self.latent_size = int(latent_size)
        self.temperature = float(temperature)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._forced_structure_ids = {
            int(self.token_ids.cort_start),
            int(self.token_ids.cort_end),
            int(self.token_ids.latent_start),
            int(self.token_ids.latent_end),
            int(self.token_ids.latent_pad),
            int(self.token_ids.observation_start),
            int(self.token_ids.vision_start),
            int(self.token_ids.vision_end),
            int(self.token_ids.image_pad),
        }

    def _build_messages(self, prompt: str, question_image: Image.Image | None) -> list[dict]:
        user_content: list[dict] = []
        if question_image is not None:
            user_content.append({"type": "image", "image": question_image})
        user_content.append({"type": "text", "text": f"Goal: {prompt}"})
        return [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": user_content},
        ]

    def _prepare_inputs(self, prompt: str, question_image: Image.Image | None) -> dict[str, torch.Tensor]:
        messages = self._build_messages(prompt, question_image)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if question_image is None:
            batch = self.processor(text=[text], return_tensors="pt")
        else:
            batch = self.processor(text=[text], images=[question_image], return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items()}

    def _append_token_ids(self, model_inputs: dict[str, torch.Tensor], token_ids: list[int]) -> dict[str, torch.Tensor]:
        if not token_ids:
            return model_inputs
        append = torch.tensor(token_ids, device=self.device, dtype=model_inputs["input_ids"].dtype).unsqueeze(0)
        ones = torch.ones((1, append.size(1)), device=self.device, dtype=model_inputs["attention_mask"].dtype)
        model_inputs["input_ids"] = torch.cat([model_inputs["input_ids"], append], dim=1)
        model_inputs["attention_mask"] = torch.cat([model_inputs["attention_mask"], ones], dim=1)
        return model_inputs

    @torch.inference_mode()
    def _next_logits(self, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pixel_values=model_inputs.get("pixel_values"),
            image_grid_thw=model_inputs.get("image_grid_thw"),
            return_dict=True,
        )
        return outputs.logits[0, -1, :]

    def _pick_token(
        self,
        logits: torch.Tensor,
        *,
        allow_ids: list[int] | None = None,
        ban_ids: set[int] | None = None,
    ) -> int:
        work = logits.detach().clone()
        if allow_ids is not None:
            mask = torch.full_like(work, float("-inf"))
            allow = torch.tensor(allow_ids, device=work.device, dtype=torch.long)
            mask[allow] = work[allow]
            work = mask
        if ban_ids:
            ban = torch.tensor(sorted(ban_ids), device=work.device, dtype=torch.long)
            work[ban] = float("-inf")
        if self.temperature > 0:
            probs = torch.softmax(work / self.temperature, dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())
        return int(torch.argmax(work).item())

    def _decode_span(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self.processor.batch_decode([token_ids], skip_special_tokens=False)[0].strip()

    def _collect_latent_blocks(self, model_inputs: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[str]]:
        with torch.inference_mode():
            outputs = self.model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                pixel_values=model_inputs.get("pixel_values"),
                image_grid_thw=model_inputs.get("image_grid_thw"),
                output_hidden_states=True,
                return_dict=True,
            )

        ids = model_inputs["input_ids"][0]
        hidden = outputs.hidden_states[-1][0]
        latent_blocks: list[torch.Tensor] = []
        reflections: list[str] = []

        current_latent_poss: list[int] = []
        reflection_ids: list[int] = []
        in_reflection = False
        latent_pad_id = int(self.token_ids.latent_pad)

        for idx, token in enumerate(ids.tolist()):
            if token == int(self.token_ids.latent_start):
                current_latent_poss = []
                continue
            if token == latent_pad_id:
                current_latent_poss.append(idx)
                continue
            if token == int(self.token_ids.latent_end):
                block = []
                for pos in current_latent_poss:
                    src = max(0, pos - 1)
                    block.append(hidden[src, :].detach().cpu())
                latent_blocks.append(torch.stack(block, dim=0) if block else torch.empty(0, hidden.size(-1)))
                current_latent_poss = []
                continue
            if token == int(self.token_ids.observation_start):
                in_reflection = True
                reflection_ids = []
                continue
            if token == int(self.token_ids.observation_end):
                if in_reflection:
                    reflections.append(self._decode_span(reflection_ids))
                in_reflection = False
                reflection_ids = []
                continue
            if in_reflection:
                reflection_ids.append(token)

        return latent_blocks, reflections

    @torch.inference_mode()
    def run(
        self,
        prompt: str,
        question_image: Image.Image | None = None,
        max_cort_turns: int = 3,
        max_reflection_tokens: int = 128,
        min_reflection_tokens: int = 1,
    ) -> CoRTGenerationResult:
        model_inputs = self._prepare_inputs(prompt, question_image)
        prefix_len = int(model_inputs["input_ids"].size(1))

        self._append_token_ids(model_inputs, [int(self.token_ids.cort_start)])

        for turn_idx in range(max(1, int(max_cort_turns))):
            self._append_token_ids(
                model_inputs,
                [
                    int(self.token_ids.latent_start),
                    *([int(self.token_ids.latent_pad)] * self.latent_size),
                    int(self.token_ids.latent_end),
                    int(self.token_ids.observation_start),
                ],
            )

            for refl_idx in range(max(1, int(max_reflection_tokens))):
                logits = self._next_logits(model_inputs)
                banned = set(self._forced_structure_ids)
                banned.discard(int(self.token_ids.observation_end))
                if refl_idx < int(min_reflection_tokens):
                    banned.add(int(self.token_ids.observation_end))
                next_id = self._pick_token(logits, ban_ids=banned)
                self._append_token_ids(model_inputs, [next_id])
                if next_id == int(self.token_ids.observation_end):
                    break
            else:
                self._append_token_ids(model_inputs, [int(self.token_ids.observation_end)])

            if turn_idx + 1 >= int(max_cort_turns):
                self._append_token_ids(model_inputs, [int(self.token_ids.cort_end)])
                break

            logits = self._next_logits(model_inputs)
            next_choice = self._pick_token(
                logits,
                allow_ids=[int(self.token_ids.latent_start), int(self.token_ids.cort_end)],
            )
            if next_choice == int(self.token_ids.cort_end):
                self._append_token_ids(model_inputs, [next_choice])
                break

        ids = model_inputs["input_ids"][0, prefix_len:].detach().cpu()
        cort_text = self.processor.batch_decode([ids], skip_special_tokens=False)[0].strip()
        latent_blocks, reflections = self._collect_latent_blocks(model_inputs)

        return CoRTGenerationResult(
            prompt=prompt,
            cort_text=cort_text,
            reflections=reflections,
            latent_blocks=latent_blocks,
            turn_count=len(latent_blocks),
        )

    def run_and_save(
        self,
        prompt: str,
        output_dir: str,
        question_image: Image.Image | None = None,
        max_cort_turns: int = 3,
        max_reflection_tokens: int = 128,
        min_reflection_tokens: int = 1,
    ) -> dict[str, str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        result = self.run(
            prompt=prompt,
            question_image=question_image,
            max_cort_turns=max_cort_turns,
            max_reflection_tokens=max_reflection_tokens,
            min_reflection_tokens=min_reflection_tokens,
        )

        cort_text_path = out_dir / "cort.txt"
        latent_path = out_dir / "latents.pt"
        meta_path = out_dir / "meta.json"

        cort_text_path.write_text(result.cort_text + "\n", encoding="utf-8")
        torch.save(
            {
                "prompt": result.prompt,
                "turn_count": result.turn_count,
                "latent_blocks": result.latent_blocks,
                "reflections": result.reflections,
            },
            latent_path,
        )
        save_json(
            meta_path,
            {
                "prompt": result.prompt,
                "turn_count": result.turn_count,
                "reflections": result.reflections,
                "cort_text": str(cort_text_path),
                "latents": str(latent_path),
            },
        )

        return {
            "cort_text": str(cort_text_path),
            "latents": str(latent_path),
            "meta": str(meta_path),
        }
