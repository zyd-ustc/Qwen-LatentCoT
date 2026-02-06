"""Loss helpers for latent distillation."""

from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.nn.functional as F


def alignment_loss(
    teacher_hidden_states: torch.Tensor,
    student_hidden_states: torch.Tensor,
) -> torch.Tensor:
    if teacher_hidden_states.dim() == 3:
        return (1 - F.cosine_similarity(teacher_hidden_states.to(student_hidden_states.device), student_hidden_states, dim=-1)).mean()
    if teacher_hidden_states.dim() == 2:
        return (1 - F.cosine_similarity(teacher_hidden_states.to(student_hidden_states.device), student_hidden_states, dim=-1)).mean()
    if teacher_hidden_states.dim() == 1:
        teacher = teacher_hidden_states.to(student_hidden_states.device)
        return 1 - F.cosine_similarity(student_hidden_states.unsqueeze(0), teacher.unsqueeze(0), dim=-1).squeeze(0)
    raise ValueError(f"Unsupported teacher_hidden_states shape: {tuple(teacher_hidden_states.shape)}")


def compute_latents_only_loss(
    latents: torch.Tensor | list[torch.Tensor],
    loss_for_latents: torch.Tensor,
) -> torch.Tensor:
    def _flatten(x):
        if isinstance(x, (list, tuple)):
            out = []
            for y in x:
                out.extend(_flatten(y))
            return out
        return [x]

    latent_list = _flatten(latents)
    grads = torch.autograd.grad(
        outputs=loss_for_latents,
        inputs=latent_list,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    safe_grads = []
    for value, grad in zip(latent_list, grads):
        if grad is None:
            grad = torch.zeros_like(value)
        safe_grads.append(grad.detach())

    proxy_loss = torch.stack([(v * g).sum() for v, g in zip(latent_list, safe_grads)]).sum()
    return proxy_loss


def load_offline_tensor(
    tensor_dir: str,
    batch_metadata: list[dict],
    alignment_layer: str = "all_layers",
    rep_type: str = "rep",
    align_poss: str = "obs",
) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    for metadata in batch_metadata:
        dataset_name = metadata["dataset_name"]
        sample_id = metadata["sample_id"]
        metadata_info = f"{alignment_layer}_{dataset_name}_{sample_id}"
        if align_poss == "obs":
            fname = f"{rep_type}_{metadata_info}.pt"
        elif align_poss == "latent_end":
            fname = f"{rep_type}_latent_end_{metadata_info}.pt"
        else:
            raise ValueError(f"Unsupported align_poss: {align_poss}")

        path = os.path.join(tensor_dir, fname)
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing offline tensor file: {path}")
        payload = torch.load(path, map_location="cpu")
        values.append(payload["latent"].detach())

    return values
