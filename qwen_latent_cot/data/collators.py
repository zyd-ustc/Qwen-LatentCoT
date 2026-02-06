"""Stage-specific collators for Stage1-1/1-2/1-3 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from qwen_latent_cot.models import SpecialTokenIds
from qwen_latent_cot.utils import (
    add_latent_pad_after_auxiliary_img,
    build_4d_attn,
    build_4d_attn_wo_helper_images,
    find_ids_poss,
    generate_labels_after_multi_token_start,
    generate_labels_after_multi_token_start_only_allow,
    load_image,
    remove_auxiliary_images,
    replace_img_pad_with_latent_pad,
    replace_latent_placeholder_with_img_pad,
    resize_by_token_budget,
)


@dataclass
class CollatorConfig:
    latent_size: int = 8
    image_resize: str = "global"
    not_use_4d: bool = False
    not_mask_image: bool = False
    mask_latent: bool = False
    observation_tokens_cannot_see_question_image: bool = False
    observation_tokens_only_see_question_and_latent: bool = False
    latent_can_see_all_previous: bool = True
    mask_question_image: bool = False
    only_predict_obs: bool = False
    sft_stage2_global_img_tokens: int = 1500
    sft_stage2_per_img_tokens: int = 1280
    sft_stage3_img_tokens: int = 2000
    sft_stage2_align_poss: str = "obs"


class StageCollator:
    def __init__(
        self,
        processor,
        token_ids: SpecialTokenIds,
        cfg: CollatorConfig,
    ) -> None:
        self.processor = processor
        self.token_ids = token_ids
        self.cfg = cfg

        self.answer_start_tensor = torch.tensor(token_ids.answer_start_pattern, dtype=torch.long)
        self.latent_start_tensor = torch.tensor(token_ids.latent_start, dtype=torch.long)
        self.latent_end_tensor = torch.tensor(token_ids.latent_end, dtype=torch.long)
        self.latent_pad_tensor = torch.tensor(token_ids.latent_pad, dtype=torch.long)
        self.obs_start_tensor = torch.tensor(token_ids.observation_start, dtype=torch.long)
        self.obs_end_tensor = torch.tensor(token_ids.observation_end, dtype=torch.long)
        self.end_pad_tensor = torch.tensor(token_ids.end_of_text, dtype=torch.long)
        self.img_start_tensor = torch.tensor(token_ids.vision_start, dtype=torch.long)
        self.img_end_tensor = torch.tensor(token_ids.vision_end, dtype=torch.long)
        self.img_pad_tensor = torch.tensor(token_ids.image_pad, dtype=torch.long)

        self.special_mask_ids = self.token_ids.to_mask_dict()

    def _process_vision_info(self, examples: list[list[dict]]) -> tuple[list[Any], None]:
        images = []
        for conv in examples:
            for turn in conv:
                for item in turn.get("content", []):
                    if item.get("type") == "image":
                        images.append(load_image(item["image"]))
        return images, None

    def _obs_positions(self, ids: torch.Tensor, end_minus_one: bool = False) -> list[list[int]]:
        starts = find_ids_poss(ids, self.answer_start_tensor, self.obs_start_tensor)
        ends = find_ids_poss(ids, self.answer_start_tensor, self.obs_end_tensor)
        out: list[list[int]] = []

        assert len(starts) == len(ends)
        for start_poss, end_poss in zip(starts, ends):
            poss: list[int] = []
            if start_poss and end_poss:
                assert len(start_poss) == len(end_poss)
                for s, e in zip(start_poss, end_poss):
                    start = s + 1 if end_minus_one else s
                    poss.extend(list(range(start, e)))
            out.append(poss)
        return out

    def collate_stage1_1(self, examples: list[dict]) -> dict:
        batch: dict[str, Any] = {}
        batch["metadata"] = [ex["metadata"] for ex in examples]
        data_examples = [ex["data"] for ex in examples]

        texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in data_examples]
        texts = [replace_latent_placeholder_with_img_pad(text) for text in texts]

        image_inputs, _ = self._process_vision_info(data_examples)
        if self.cfg.image_resize == "global":
            image_inputs, _ = resize_by_token_budget(image_inputs)

        teacher_batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

        batch["teacher_input_ids"] = teacher_batch["input_ids"]
        batch["teacher_attention_mask"] = teacher_batch["attention_mask"]
        batch["teacher_pixel_values"] = teacher_batch["pixel_values"]
        batch["teacher_image_grid_thw"] = teacher_batch["image_grid_thw"]

        batch["teacher_observation_poss"] = self._obs_positions(batch["teacher_input_ids"])
        batch["teacher_labels"] = generate_labels_after_multi_token_start(
            batch["teacher_input_ids"],
            self.answer_start_tensor,
            ignore_ids=[
                self.end_pad_tensor,
                self.img_pad_tensor,
                self.img_start_tensor,
                self.img_end_tensor,
                self.obs_start_tensor,
                self.obs_end_tensor,
            ],
        )

        return batch

    def collate_stage1_2(self, examples: list[dict]) -> dict:
        metadata = [ex["metadata"] for ex in examples]
        data_examples = [ex["data"] for ex in examples]

        texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in data_examples]
        texts = [replace_latent_placeholder_with_img_pad(text) for text in texts]
        texts = add_latent_pad_after_auxiliary_img(texts, self.cfg.latent_size, "<abs_vis_token_pad>")

        image_inputs, _ = self._process_vision_info(data_examples)
        if self.cfg.image_resize == "global":
            image_inputs, _ = resize_by_token_budget(
                image_inputs,
                global_max_pixels=self.cfg.sft_stage2_global_img_tokens * 28 * 28,
                per_img_max_pixels=self.cfg.sft_stage2_per_img_tokens * 28 * 28,
            )

        batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        batch["metadata"] = metadata

        if not self.cfg.not_use_4d:
            attn_mask_4d, _ = build_4d_attn(
                input_ids=batch["input_ids"],
                pad_mask=batch["attention_mask"],
                token_ids=self.special_mask_ids,
                not_mask_image=self.cfg.not_mask_image,
                mask_latent=self.cfg.mask_latent,
                observation_tokens_cannot_see_question_image=self.cfg.observation_tokens_cannot_see_question_image,
                observation_tokens_only_see_question_and_latent=self.cfg.observation_tokens_only_see_question_and_latent,
                latent_can_see_all_previous=self.cfg.latent_can_see_all_previous,
                mask_question_image=self.cfg.mask_question_image,
            )
            batch["attention_mask_4d"] = {"full_attention": attn_mask_4d}

        if self.cfg.sft_stage2_align_poss == "latent_end":
            batch["latent_end_poss"] = find_ids_poss(
                batch["input_ids"], self.answer_start_tensor, self.latent_end_tensor
            )

        batch["observation_poss"] = self._obs_positions(batch["input_ids"])
        if self.cfg.only_predict_obs:
            batch["labels"] = generate_labels_after_multi_token_start_only_allow(
                batch["input_ids"],
                self.answer_start_tensor,
                allowed_poss=batch["observation_poss"],
            )
        else:
            batch["labels"] = generate_labels_after_multi_token_start(
                batch["input_ids"],
                self.answer_start_tensor,
                ignore_ids=[
                    self.end_pad_tensor,
                    self.latent_pad_tensor,
                    self.latent_end_tensor,
                    self.img_pad_tensor,
                    self.img_start_tensor,
                    self.img_end_tensor,
                    self.obs_start_tensor,
                    self.obs_end_tensor,
                ],
            )

        return batch

    def collate_stage1_3(self, examples: list[dict]) -> dict:
        batch: dict[str, Any] = {}
        batch["metadata"] = [ex["metadata"] for ex in examples]
        data_examples = [ex["data"] for ex in examples]

        texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in data_examples]
        texts = [replace_latent_placeholder_with_img_pad(text) for text in texts]

        student_texts = replace_img_pad_with_latent_pad(
            texts, self.cfg.latent_size, "<abs_vis_token_pad>"
        )
        user_examples = remove_auxiliary_images(data_examples)

        user_image_inputs, _ = self._process_vision_info(user_examples)
        user_image_inputs, _ = resize_by_token_budget(
            user_image_inputs,
            global_max_pixels=self.cfg.sft_stage3_img_tokens * 28 * 28,
            per_img_max_pixels=self.cfg.sft_stage3_img_tokens * 28 * 28,
        )

        student_batch = self.processor(
            text=student_texts,
            images=user_image_inputs,
            return_tensors="pt",
            padding=True,
        )

        batch["student_input_ids"] = student_batch["input_ids"]
        batch["student_attention_mask"] = student_batch["attention_mask"]
        batch["student_pixel_values"] = student_batch["pixel_values"]
        batch["student_image_grid_thw"] = student_batch["image_grid_thw"]

        if self.cfg.mask_latent:
            attn_4d = build_4d_attn_wo_helper_images(
                input_ids=batch["student_input_ids"],
                pad_mask=batch["student_attention_mask"],
                token_ids=self.special_mask_ids,
                mask_latent=self.cfg.mask_latent,
            )
            batch["student_attention_mask_4d"] = {"full_attention": attn_4d}

        batch["student_alignment_poss"] = find_ids_poss(
            batch["student_input_ids"], self.answer_start_tensor, self.latent_pad_tensor
        )
        batch["observation_poss"] = self._obs_positions(
            batch["student_input_ids"], end_minus_one=True
        )

        batch["student_labels"] = generate_labels_after_multi_token_start(
            batch["student_input_ids"],
            self.answer_start_tensor,
            ignore_ids=[
                self.img_pad_tensor,
                self.img_start_tensor,
                self.img_end_tensor,
                self.end_pad_tensor,
                self.latent_pad_tensor,
                self.latent_end_tensor,
                self.obs_start_tensor,
                self.obs_end_tensor,
            ],
        )

        return batch
