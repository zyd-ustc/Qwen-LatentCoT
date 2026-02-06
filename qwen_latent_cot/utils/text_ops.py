"""Token/text processing helpers for latent CoT training."""

from __future__ import annotations

from typing import Iterable

import torch


def process_multiple_question_img(question_str: str) -> str:
    if "<abs_vis_token></abs_vis_token>" in question_str:
        question_str = question_str.replace(
            "<|vision_start|><|image_pad|><|vision_end|>", ""
        ).replace(
            "<abs_vis_token></abs_vis_token>",
            "<|vision_start|><|image_pad|><|vision_end|>",
        )
    return question_str


def replace_latent_placeholder_with_img_pad(
    text: str,
    image_pad: str = "<|vision_start|><|image_pad|><|vision_end|>",
    latent_placeholder: str = "<abs_vis_token></abs_vis_token>",
    sep_token: str = "<|im_start|>assistant",
) -> str:
    turns = text.split(sep_token)
    out = process_multiple_question_img(turns[0])
    for turn in turns[1:]:
        if latent_placeholder in turn:
            turn = turn.replace(image_pad, "")
            turn = turn.replace(latent_placeholder, image_pad)
        out += sep_token + turn
    return out


def replace_img_pad_with_latent_pad(
    texts: list[str], latent_size: int, latent_pad_str: str = "<abs_vis_token_pad>"
) -> list[str]:
    update_texts: list[str] = []
    latent_pad_strs = latent_pad_str * latent_size
    for text in texts:
        turns = text.split("<|im_start|>assistant")
        updated = process_multiple_question_img(turns[0])
        for turn in turns[1:]:
            updated += (
                "<|im_start|>assistant"
                + turn.replace(
                    "<|vision_start|><|image_pad|><|vision_end|>",
                    f"<abs_vis_token>{latent_pad_strs}</abs_vis_token>",
                )
            )
        update_texts.append(updated)
    return update_texts


def add_latent_pad_after_auxiliary_img(
    texts: list[str], latent_size: int, latent_pad_str: str = "<abs_vis_token_pad>"
) -> list[str]:
    update_texts: list[str] = []
    latent_pad_strs = latent_pad_str * latent_size
    for text in texts:
        turns = text.split("<|im_start|>assistant")
        updated = process_multiple_question_img(turns[0])
        for turn in turns[1:]:
            updated += (
                "<|im_start|>assistant"
                + turn.replace(
                    "<|vision_start|><|image_pad|><|vision_end|>",
                    "<|vision_start|><|image_pad|><|vision_end|>"
                    + f"<abs_vis_token>{latent_pad_strs}</abs_vis_token>",
                )
            )
        update_texts.append(updated)
    return update_texts


def find_subsequence(
    row: torch.Tensor, pattern: torch.Tensor | list[torch.Tensor], start: int = 0
) -> int:
    seq_len = row.size(0)
    if isinstance(pattern, torch.Tensor):
        max_pat_len = pattern.size(0)
    else:
        max_pat_len = max(p.size(0) for p in pattern)

    for start_idx in range(start, seq_len - max_pat_len + 1):
        if isinstance(pattern, torch.Tensor):
            pat_len = pattern.size(0)
            if torch.all(row[start_idx : start_idx + pat_len] == pattern):
                return start_idx
        else:
            for pat in pattern:
                pat_len = pat.size(0)
                if torch.all(row[start_idx : start_idx + pat_len] == pat):
                    return start_idx
    return -1


def find_ids_poss(
    input_ids: torch.Tensor,
    answer_start_token_pattern: torch.Tensor,
    ids_tensor_or_list: torch.Tensor | list[torch.Tensor],
) -> list[list[int]]:
    batch_poss: list[list[int]] = []
    for i in range(input_ids.shape[0]):
        poss: list[int] = []
        start_idx = find_subsequence(input_ids[i], answer_start_token_pattern, 0)
        while start_idx != -1:
            start_idx = find_subsequence(input_ids[i], ids_tensor_or_list, start_idx + 1)
            if start_idx != -1:
                poss.append(start_idx)
        batch_poss.append(poss)
    return batch_poss


def generate_labels_after_multi_token_start(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    ignore_ids: Iterable[int] | None = None,
) -> torch.Tensor:
    labels = input_ids.clone()
    ignore = list(ignore_ids or [])

    for b in range(labels.size(0)):
        row = labels[b]
        start_idx = find_subsequence(row, start_sequence)
        if start_idx == -1:
            row[:] = -100
        else:
            end_of_subseq = start_idx + start_sequence.size(0)
            row[:end_of_subseq] = -100

        for idx in ignore:
            row[row == int(idx)] = -100

    return labels


def generate_labels_after_multi_token_start_only_allow(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    allowed_poss: list[list[int]],
) -> torch.Tensor:
    labels = input_ids.clone()

    for b in range(labels.size(0)):
        row = labels[b]
        start_idx = find_subsequence(row, start_sequence)
        if start_idx == -1:
            row[:] = -100
            continue

        row[: start_idx + start_sequence.size(0)] = -100
        mask = torch.ones_like(row, dtype=torch.bool)
        if b < len(allowed_poss) and len(allowed_poss[b]) > 0:
            mask[torch.tensor(allowed_poss[b], dtype=torch.long)] = False
        row[mask] = -100

    return labels


def remove_auxiliary_images(examples: list[list[dict]]) -> list[list[dict]]:
    new_examples: list[list[dict]] = []
    for example in examples:
        new_turns: list[dict] = []
        for turn in example:
            turn_copy = dict(turn)
            if turn_copy.get("role") == "assistant":
                turn_copy["content"] = [
                    item for item in turn_copy.get("content", []) if item.get("type") != "image"
                ]
            new_turns.append(turn_copy)
        new_examples.append(new_turns)
    return new_examples
