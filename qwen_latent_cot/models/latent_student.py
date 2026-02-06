"""Latent-mode wrapper over Qwen2.5-VL for Stage1 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from qwen_latent_cot.utils import alignment_loss


@dataclass
class LatentCausalLMOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    hidden_states: Any | None = None
    ce_patch_pos: list[list[int]] | None = None
    ce_patch_vec: list[torch.Tensor] | None = None
    alignment_loss: torch.Tensor | None = None
    loss_dict: dict[str, torch.Tensor] | None = None
    latent_embeds: list[torch.Tensor] | None = None
    mean_emphasize_acc: float | None = None


class LatentQwenVLWrapper(nn.Module):
    """Wraps `Qwen2_5_VLForConditionalGeneration` with latent-mode APIs.

    This wrapper keeps interface compatibility with Monet-style trainer code while
    avoiding hard patching of upstream transformers internals.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def config(self):
        return self.model.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                return self.model.gradient_checkpointing_enable()
            return self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()

    def resize_token_embeddings(self, *args, **kwargs):
        return self.model.resize_token_embeddings(*args, **kwargs)

    def _gather_last_hidden(
        self,
        last_hidden: torch.Tensor,
        positions: list[list[int]] | None,
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        bsz = last_hidden.size(0)
        for b in range(bsz):
            pos_list = positions[b] if positions and b < len(positions) else []
            if len(pos_list) == 0:
                outputs.append(torch.empty(0, last_hidden.size(-1), device=last_hidden.device, dtype=last_hidden.dtype))
            else:
                pos = torch.tensor(pos_list, dtype=torch.long, device=last_hidden.device)
                outputs.append(last_hidden[b, pos, :])
        return outputs

    def _gather_all_layers(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        positions: list[list[int]] | None,
    ) -> list[torch.Tensor]:
        layer_tensors = hidden_states[1:] if len(hidden_states) > 1 else hidden_states
        bsz = layer_tensors[-1].size(0)
        outputs: list[torch.Tensor] = []
        for b in range(bsz):
            pos_list = positions[b] if positions and b < len(positions) else []
            if len(pos_list) == 0:
                outputs.append(
                    torch.empty(
                        len(layer_tensors),
                        0,
                        layer_tensors[-1].size(-1),
                        device=layer_tensors[-1].device,
                        dtype=layer_tensors[-1].dtype,
                    )
                )
            else:
                pos = torch.tensor(pos_list, dtype=torch.long, device=layer_tensors[-1].device)
                picked = [layer[b, pos, :] for layer in layer_tensors]
                outputs.append(torch.stack(picked, dim=0))
        return outputs

    def _compute_alignment(
        self,
        student_last_hidden: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...] | None,
        teacher_hidden_states_for_alignment: list[torch.Tensor] | None,
        alignment_poss: list[list[int]] | None,
    ) -> torch.Tensor:
        if teacher_hidden_states_for_alignment is None:
            return torch.tensor(0.0, device=student_last_hidden.device)

        student_last = self._gather_last_hidden(student_last_hidden, alignment_poss)
        student_all = None

        losses = []
        for b, teacher in enumerate(teacher_hidden_states_for_alignment):
            if teacher is None:
                continue
            teacher = teacher.to(student_last_hidden.device)
            if teacher.dim() == 3:
                if hidden_states is None:
                    raise ValueError("Alignment with all layers requires output_hidden_states=True")
                if student_all is None:
                    student_all = self._gather_all_layers(hidden_states, alignment_poss)
                if student_all[b].numel() == 0:
                    continue
                n = min(teacher.size(1), student_all[b].size(1))
                if n == 0:
                    continue
                losses.append(alignment_loss(teacher[:, :n, :], student_all[b][:, :n, :]))
            elif teacher.dim() == 2:
                if student_last[b].numel() == 0:
                    continue
                n = min(teacher.size(0), student_last[b].size(0))
                if n == 0:
                    continue
                losses.append(alignment_loss(teacher[:n, :], student_last[b][:n, :]))
            elif teacher.dim() == 1:
                if student_last[b].numel() == 0:
                    continue
                losses.append(alignment_loss(teacher, student_last[b][0]))

        if not losses:
            return torch.tensor(0.0, device=student_last_hidden.device)
        return torch.stack(losses).mean()

    def _ce_with_optional_emphasis(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ce_emphasize_poss: list[list[int]] | None,
        ce_emphasize_factor: float,
        compute_emphasize_acc: bool,
    ) -> tuple[torch.Tensor, float | None]:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        bsz, seq, vocab = shift_logits.shape
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(bsz, seq)

        valid = shift_labels != -100
        weight = torch.ones_like(loss_per_token)

        if (
            ce_emphasize_poss is not None
            and len(ce_emphasize_poss) > 0
            and ce_emphasize_factor is not None
            and float(ce_emphasize_factor) != 1.0
        ):
            for b, poss in enumerate(ce_emphasize_poss):
                if b >= bsz or not poss:
                    continue
                pos = torch.tensor(poss, device=logits.device, dtype=torch.long) - 1
                pos = pos[(pos >= 0) & (pos < seq)]
                if pos.numel() > 0:
                    weight[b, pos] = float(ce_emphasize_factor)

        weighted = loss_per_token * weight * valid
        denom = valid.float().sum().clamp_min(1.0)
        ce_loss = weighted.sum() / denom

        mean_acc = None
        if compute_emphasize_acc and ce_emphasize_poss:
            pred = shift_logits.argmax(dim=-1)
            acc_vals = []
            for b, poss in enumerate(ce_emphasize_poss):
                if b >= bsz or not poss:
                    continue
                pos = torch.tensor(poss, device=logits.device, dtype=torch.long) - 1
                pos = pos[(pos >= 0) & (pos < seq)]
                if pos.numel() == 0:
                    continue
                tgt = shift_labels[b, pos]
                mask = tgt != -100
                if mask.any():
                    acc = (pred[b, pos][mask] == tgt[mask]).float().mean().item()
                    acc_vals.append(acc)
            if acc_vals:
                mean_acc = float(sum(acc_vals) / len(acc_vals))

        return ce_loss, mean_acc

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask_4d: dict[str, torch.Tensor] | None = None,
        latent_mode: bool = False,
        teacher_hidden_states_for_alignment: list[torch.Tensor] | None = None,
        alignment_poss: list[list[int]] | None = None,
        ce_patch_pos: list[list[int]] | None = None,
        ce_patch_vec: list[torch.Tensor] | None = None,
        ce_emphasize_factor: float = 1.0,
        ce_emphasize_poss: list[list[int]] | None = None,
        loss_type: list[str] | None = None,
        compute_emphasize_acc: bool = False,
        output_hidden_states: bool = False,
        output_latent_embeds: bool = False,
        **kwargs,
    ) -> LatentCausalLMOutput:
        del kwargs

        loss_type = list(loss_type or [])

        if attention_mask_4d is not None and isinstance(attention_mask_4d, dict):
            if "full_attention" in attention_mask_4d:
                attention_mask = attention_mask_4d["full_attention"]

        inputs_embeds = None
        if (not latent_mode) and ce_patch_pos is not None and ce_patch_vec is not None:
            embedding_layer = self.model.get_input_embeddings()
            inputs_embeds = embedding_layer(input_ids)
            for b, pos_list in enumerate(ce_patch_pos):
                if b >= inputs_embeds.size(0) or not pos_list:
                    continue
                vec = ce_patch_vec[b]
                if vec.numel() == 0:
                    continue
                vec = vec.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                pos = torch.tensor(pos_list, device=inputs_embeds.device, dtype=torch.long)
                take = min(pos.numel(), vec.size(0))
                if take > 0:
                    inputs_embeds[b, pos[:take], :] = vec[:take]

        outputs = self.model(
            input_ids=None if inputs_embeds is not None else input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=(output_hidden_states or latent_mode or ("alignment" in loss_type)),
            return_dict=True,
            labels=None,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states
        last_hidden = hidden_states[-1] if hidden_states is not None else None

        # Latent positions and extracted vectors are always derived from current input IDs.
        latent_token_id = int(getattr(self.model.config, "latent_token_id", -1))
        ce_patch_pos_out: list[list[int]] = []
        ce_patch_vec_out: list[torch.Tensor] = []
        latent_embeds_out: list[torch.Tensor] = []

        if latent_token_id != -1 and last_hidden is not None:
            for b in range(input_ids.size(0)):
                pos = (input_ids[b] == latent_token_id).nonzero(as_tuple=False).flatten().tolist()
                ce_patch_pos_out.append(pos)
                latents: list[torch.Tensor] = []
                for p in pos:
                    source_idx = max(0, p - 1)
                    latents.append(last_hidden[b, source_idx, :])
                if latents:
                    latent_tensor = torch.stack(latents, dim=0)
                else:
                    latent_tensor = torch.empty(
                        0,
                        last_hidden.size(-1),
                        device=last_hidden.device,
                        dtype=last_hidden.dtype,
                    )
                ce_patch_vec_out.append(latent_tensor)
                latent_embeds_out.append(latent_tensor)

        align_loss = torch.tensor(0.0, device=logits.device)
        if "alignment" in loss_type and last_hidden is not None:
            align_loss = self._compute_alignment(
                last_hidden,
                hidden_states,
                teacher_hidden_states_for_alignment,
                alignment_poss,
            )

        ce_loss = None
        mean_emphasize_acc = None
        if "ce" in loss_type and labels is not None:
            ce_loss, mean_emphasize_acc = self._ce_with_optional_emphasis(
                logits,
                labels,
                ce_emphasize_poss,
                ce_emphasize_factor,
                compute_emphasize_acc,
            )

        total_loss = None
        loss_dict: dict[str, torch.Tensor] = {}

        if ce_loss is not None:
            total_loss = ce_loss if total_loss is None else total_loss + ce_loss
            loss_dict["ce"] = ce_loss
        if "alignment" in loss_type:
            total_loss = align_loss if total_loss is None else total_loss + align_loss
            loss_dict["alignment"] = align_loss

        return LatentCausalLMOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=hidden_states if output_hidden_states else None,
            ce_patch_pos=ce_patch_pos_out if latent_mode else ce_patch_pos,
            ce_patch_vec=ce_patch_vec_out if latent_mode else ce_patch_vec,
            alignment_loss=align_loss,
            loss_dict=loss_dict,
            latent_embeds=latent_embeds_out if output_latent_embeds else None,
            mean_emphasize_acc=mean_emphasize_acc,
        )


def freeze_visual_encoder(model: nn.Module) -> None:
    visual = getattr(model, "visual", None)
    if visual is None:
        return
    for p in visual.parameters():
        p.requires_grad = False
