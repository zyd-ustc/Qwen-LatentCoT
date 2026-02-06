"""Custom trainers for stage1-1/1-2/1-3."""

from __future__ import annotations

import gc
from typing import Any

import torch
from transformers import Trainer

from qwen_latent_cot.utils import compute_latents_only_loss, load_offline_tensor


class Stage11Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_ce_cum = 0.0
        self.teacher_ce_steps = 0
        self.obs_acc_cum = 0.0
        self.obs_acc_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        outputs = model(
            input_ids=inputs["teacher_input_ids"],
            attention_mask=inputs["teacher_attention_mask"],
            pixel_values=inputs.get("teacher_pixel_values"),
            image_grid_thw=inputs.get("teacher_image_grid_thw"),
            labels=inputs["teacher_labels"],
            latent_mode=False,
            loss_type=["ce"],
            ce_emphasize_poss=inputs.get("teacher_observation_poss"),
            ce_emphasize_factor=float(getattr(self.args, "ce_emphasize_factor", 1.0)),
            compute_emphasize_acc=True,
        )

        loss = outputs.loss
        self.teacher_ce_cum += float(loss.detach().item())
        self.teacher_ce_steps += 1
        if outputs.mean_emphasize_acc is not None:
            self.obs_acc_cum += float(outputs.mean_emphasize_acc)
            self.obs_acc_steps += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        merged = dict(logs)
        if self.teacher_ce_steps > 0:
            merged["teacher_ce_loss"] = round(self.teacher_ce_cum / self.teacher_ce_steps, 6)
            self.teacher_ce_cum = 0.0
            self.teacher_ce_steps = 0
        if self.obs_acc_steps > 0:
            merged["observation_token_acc"] = round(self.obs_acc_cum / self.obs_acc_steps, 6)
            self.obs_acc_cum = 0.0
            self.obs_acc_steps = 0
        return super().log(merged, start_time)


class Stage12Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce_cum = 0.0
        self.ce_steps = 0
        self.al_cum = 0.0
        self.al_steps = 0
        self.obs_acc_cum = 0.0
        self.obs_acc_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch

        latent_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            latent_mode=True,
            loss_type=[],
            attention_mask_4d=inputs.get("attention_mask_4d"),
        )

        teacher_reps = None
        if float(getattr(self.args, "alignment_weight", 1.0)) != 0:
            teacher_reps = load_offline_tensor(
                getattr(self.args, "teacher_reps_dir"),
                batch_metadata=inputs["metadata"],
                alignment_layer=getattr(self.args, "alignment_layer", "all_layers"),
                rep_type="rep",
                align_poss=getattr(self.args, "sft_stage2_align_poss", "obs"),
            )

        loss_type = ["ce"]
        if teacher_reps is not None:
            loss_type.append("alignment")

        alignment_poss = inputs.get("observation_poss")
        if getattr(self.args, "sft_stage2_align_poss", "obs") == "latent_end":
            alignment_poss = inputs.get("latent_end_poss")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            labels=inputs["labels"],
            attention_mask_4d=inputs.get("attention_mask_4d"),
            latent_mode=False,
            ce_patch_pos=latent_outputs.ce_patch_pos,
            ce_patch_vec=latent_outputs.ce_patch_vec,
            ce_emphasize_poss=inputs.get("observation_poss"),
            ce_emphasize_factor=float(getattr(self.args, "ce_emphasize_factor", 1.0)),
            teacher_hidden_states_for_alignment=teacher_reps,
            alignment_poss=alignment_poss,
            loss_type=loss_type,
            compute_emphasize_acc=True,
        )

        ce_loss = outputs.loss_dict.get("ce", torch.tensor(0.0, device=outputs.loss.device))
        al_loss = outputs.loss_dict.get("alignment", torch.tensor(0.0, device=outputs.loss.device))

        alignment_weight = float(getattr(self.args, "alignment_weight", 1.0))
        emphasize_latent_weight = float(getattr(self.args, "emphasize_latent_weight", 1.0))

        if teacher_reps is not None and emphasize_latent_weight != 0.0 and al_loss.item() != 0.0:
            latent_only = compute_latents_only_loss(
                latent_outputs.ce_patch_vec,
                alignment_weight * al_loss,
            )
            loss = ce_loss + emphasize_latent_weight * latent_only
        else:
            loss = ce_loss + alignment_weight * al_loss

        self.ce_cum += float(ce_loss.detach().item())
        self.ce_steps += 1
        self.al_cum += float(al_loss.detach().item())
        self.al_steps += 1

        if outputs.mean_emphasize_acc is not None:
            self.obs_acc_cum += float(outputs.mean_emphasize_acc)
            self.obs_acc_steps += 1

        step = int(getattr(self.state, "global_step", 0) or 0)
        if step % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        merged = dict(logs)
        if self.ce_steps > 0:
            merged["teacher_ce_loss"] = round(self.ce_cum / self.ce_steps, 6)
            self.ce_cum = 0.0
            self.ce_steps = 0
        if self.al_steps > 0:
            merged["alignment_loss"] = round(self.al_cum / self.al_steps, 6)
            self.al_cum = 0.0
            self.al_steps = 0
        if self.obs_acc_steps > 0:
            merged["observation_token_acc"] = round(self.obs_acc_cum / self.obs_acc_steps, 6)
            self.obs_acc_cum = 0.0
            self.obs_acc_steps = 0
        return super().log(merged, start_time)


class Stage13Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce_cum = 0.0
        self.ce_steps = 0
        self.al_cum = 0.0
        self.al_steps = 0
        self.obs_acc_cum = 0.0
        self.obs_acc_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch

        teacher_latents = load_offline_tensor(
            getattr(self.args, "teacher_latent_dir"),
            batch_metadata=inputs["metadata"],
            alignment_layer=getattr(self.args, "alignment_layer", "all_layers"),
            rep_type="latent",
            align_poss="obs",
        )

        latent_outputs = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
            pixel_values=inputs.get("student_pixel_values"),
            image_grid_thw=inputs.get("student_image_grid_thw"),
            latent_mode=True,
            loss_type=[],
            teacher_hidden_states_for_alignment=teacher_latents,
            alignment_poss=inputs.get("student_alignment_poss"),
            attention_mask_4d=inputs.get("student_attention_mask_4d"),
        )

        outputs = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
            pixel_values=inputs.get("student_pixel_values"),
            image_grid_thw=inputs.get("student_image_grid_thw"),
            labels=inputs["student_labels"],
            latent_mode=False,
            ce_patch_pos=latent_outputs.ce_patch_pos,
            ce_patch_vec=latent_outputs.ce_patch_vec,
            teacher_hidden_states_for_alignment=teacher_latents,
            alignment_poss=inputs.get("student_alignment_poss"),
            ce_emphasize_poss=inputs.get("observation_poss"),
            ce_emphasize_factor=float(getattr(self.args, "ce_emphasize_factor", 1.0)),
            loss_type=["ce", "alignment"],
            compute_emphasize_acc=True,
            attention_mask_4d=inputs.get("student_attention_mask_4d"),
        )

        ce_loss = outputs.loss_dict.get("ce", torch.tensor(0.0, device=outputs.loss.device))
        al_loss = outputs.loss_dict.get("alignment", torch.tensor(0.0, device=outputs.loss.device))

        alignment_weight = float(getattr(self.args, "alignment_weight", 1.0))
        loss = ce_loss + alignment_weight * al_loss

        self.ce_cum += float(ce_loss.detach().item())
        self.ce_steps += 1
        self.al_cum += float(al_loss.detach().item())
        self.al_steps += 1

        if outputs.mean_emphasize_acc is not None:
            self.obs_acc_cum += float(outputs.mean_emphasize_acc)
            self.obs_acc_steps += 1

        step = int(getattr(self.state, "global_step", 0) or 0)
        if step > 0 and step % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        merged = dict(logs)
        if self.ce_steps > 0:
            merged["student_ce_loss"] = round(self.ce_cum / self.ce_steps, 6)
            self.ce_cum = 0.0
            self.ce_steps = 0
        if self.al_steps > 0:
            merged["alignment_loss"] = round(self.al_cum / self.al_steps, 6)
            self.al_cum = 0.0
            self.al_steps = 0
        if self.obs_acc_steps > 0:
            merged["observation_token_acc"] = round(self.obs_acc_cum / self.obs_acc_steps, 6)
            self.obs_acc_cum = 0.0
            self.obs_acc_steps = 0
        return super().log(merged, start_time)
