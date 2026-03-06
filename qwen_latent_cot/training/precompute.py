"""Teacher latent/representation precomputation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from qwen_latent_cot.data import LatentCoTDataset
from qwen_latent_cot.data.collators import CollatorConfig, StageCollator
from qwen_latent_cot.models import (
    add_latent_special_tokens,
    attach_special_ids_to_model,
    resolve_special_token_ids,
)
from qwen_latent_cot.models.latent_student import LatentQwenVLWrapper, freeze_visual_encoder
from qwen_latent_cot.models.loaders import load_qwen2_5_vl
from qwen_latent_cot.utils import build_logger, find_ids_poss, seed_everything


@dataclass
class PrecomputeConfig:
    model_path: str
    data_paths: list[str]
    output_dir: str
    dataset_root: str = ""
    qwen_image_edit_root: str = ""
    deepspeed: str = ""
    allow_no_observation: bool = False
    shuffle_train: bool = False
    seed: int = 42
    dtype: str = "bfloat16"
    batch_size: int = 1
    latent_size: int = 8
    image_resize: str = "global"
    not_use_4d: bool = False
    not_mask_image: bool = False
    mask_latent: bool = False
    observation_tokens_cannot_see_question_image: bool = False
    observation_tokens_only_see_question_and_latent: bool = False
    latent_can_see_all_previous: bool = True
    mask_question_image: bool = False
    sft_stage2_align_poss: str = "obs"
    output_hidden_states: bool = True
    output_latent_embeds: bool = False
    log_file: str | None = None
    save_every: int = 2000


def _resolve_rank_world(cfg: PrecomputeConfig, logger) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if cfg.deepspeed and world_size > 1:
        import deepspeed

        deepspeed.init_distributed()
        rank = int(os.environ.get("RANK", str(rank)))
        world_size = int(os.environ.get("WORLD_SIZE", str(world_size)))
        local_rank = int(os.environ.get("LOCAL_RANK", str(local_rank)))
        logger.info(
            "DeepSpeed distributed enabled: rank=%d world_size=%d local_rank=%d",
            rank,
            world_size,
            local_rank,
        )
    return rank, world_size, local_rank


def _place_and_wrap_model(model, cfg: PrecomputeConfig, local_rank: int):
    if torch.cuda.is_available():
        if cfg.deepspeed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if cfg.deepspeed:
        import deepspeed

        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            # Precompute is inference-only; avoid optimizer construction (e.g. CPUAdam JIT build).
            model_parameters=None,
            config=cfg.deepspeed,
        )
        model_engine.eval()
        return model_engine, device

    model.to(device)
    model.eval()
    return model, device


def _shard_dataset_for_rank(dataset: LatentCoTDataset, rank: int, world_size: int):
    if world_size <= 1:
        return dataset
    indices = list(range(rank, len(dataset), world_size))
    return Subset(dataset, indices)


def _create_model_and_collator(
    cfg: PrecomputeConfig,
    *,
    stage1_1_vae_roundtrip: bool = False,
    stage1_23_noise_vision: bool = False,
):
    processor, base_model = load_qwen2_5_vl(
        cfg.model_path,
        dtype=cfg.dtype,
        base_model_path=(cfg.qwen_image_edit_root or None),
    )
    add_latent_special_tokens(processor)
    try:
        base_model.resize_token_embeddings(len(processor.tokenizer))
    except Exception:
        pass

    token_ids = resolve_special_token_ids(processor)
    attach_special_ids_to_model(base_model, token_ids)
    freeze_visual_encoder(base_model)

    model = LatentQwenVLWrapper(base_model)
    model.eval()

    collator = StageCollator(
        processor=processor,
        token_ids=token_ids,
        cfg=CollatorConfig(
            latent_size=cfg.latent_size,
            image_resize=cfg.image_resize,
            not_use_4d=cfg.not_use_4d,
            not_mask_image=cfg.not_mask_image,
            mask_latent=cfg.mask_latent,
            observation_tokens_cannot_see_question_image=cfg.observation_tokens_cannot_see_question_image,
            observation_tokens_only_see_question_and_latent=cfg.observation_tokens_only_see_question_and_latent,
            latent_can_see_all_previous=cfg.latent_can_see_all_previous,
            mask_question_image=cfg.mask_question_image,
            sft_stage2_align_poss=cfg.sft_stage2_align_poss,
            qwen_image_edit_root=(
                cfg.qwen_image_edit_root
                or (cfg.model_path if os.path.isdir(os.path.join(cfg.model_path, "vae")) else None)
            ),
            stage1_1_vae_roundtrip=(
                stage1_1_vae_roundtrip
                and bool(
                    cfg.qwen_image_edit_root
                    or os.path.isdir(os.path.join(cfg.model_path, "vae"))
                )
            ),
            stage1_23_noise_vision=stage1_23_noise_vision,
        ),
    )
    return processor, model, collator, token_ids


def _prefix(cfg: PrecomputeConfig) -> str:
    return "last_layer" if cfg.output_latent_embeds else "all_layers"


def _extract_positions_hidden(
    hidden_states: tuple[torch.Tensor, ...],
    positions: list[list[int]],
    output_all_layers: bool,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    layers = hidden_states[1:] if len(hidden_states) > 1 else hidden_states
    last = layers[-1]

    for b, pos in enumerate(positions):
        if not pos:
            if output_all_layers:
                outputs.append(
                    torch.empty(
                        len(layers),
                        0,
                        last.size(-1),
                        device=last.device,
                        dtype=last.dtype,
                    )
                )
            else:
                outputs.append(
                    torch.empty(0, last.size(-1), device=last.device, dtype=last.dtype)
                )
            continue

        idx = torch.tensor(pos, dtype=torch.long, device=last.device)
        if output_all_layers:
            stacked = torch.stack([layer[b, idx, :] for layer in layers], dim=0)
            outputs.append(stacked)
        else:
            outputs.append(last[b, idx, :])

    return outputs


def run_precompute_teacher_latents(cfg: PrecomputeConfig) -> None:
    seed_everything(cfg.seed)
    logger = build_logger("qwen_latent_cot.precompute_latent", cfg.log_file)
    rank, world_size, local_rank = _resolve_rank_world(cfg, logger)

    full_dataset = LatentCoTDataset(
        data_paths=cfg.data_paths,
        dataset_root=cfg.dataset_root,
        allow_no_observation=cfg.allow_no_observation,
        shuffle=cfg.shuffle_train,
        seed=cfg.seed,
    )
    dataset = _shard_dataset_for_rank(full_dataset, rank, world_size)
    logger.info(
        "Loaded %d samples on rank %d (global total: %d)",
        len(dataset),
        rank,
        len(full_dataset),
    )

    processor, model, collator, _ = _create_model_and_collator(
        cfg,
        stage1_23_noise_vision=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator.collate_stage1_2,
    )

    model, device = _place_and_wrap_model(model, cfg, local_rank)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Precompute teacher latents", disable=(rank != 0)):
            pixel_values = batch.get("pixel_values", None)
            image_grid_thw = batch.get("image_grid_thw", None)
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask_4d=batch.get("attention_mask_4d"),
                latent_mode=True,
                output_latent_embeds=cfg.output_latent_embeds,
                output_hidden_states=cfg.output_hidden_states,
                loss_type=[],
            )

            if cfg.output_latent_embeds:
                reps = outputs.latent_embeds
            else:
                reps = outputs.ce_patch_vec

            for i, rep in enumerate(reps):
                md = batch["metadata"][i]
                metadata_info = f"{_prefix(cfg)}_{md['dataset_name']}_{md['sample_id']}"
                path = out_dir / f"latent_{metadata_info}.pt"
                torch.save({"metadata_info": metadata_info, "latent": rep.detach().cpu()}, path)

    logger.info("Saved teacher latents to %s", out_dir)


def run_precompute_teacher_reps(cfg: PrecomputeConfig) -> None:
    seed_everything(cfg.seed)
    logger = build_logger("qwen_latent_cot.precompute_rep", cfg.log_file)
    rank, world_size, local_rank = _resolve_rank_world(cfg, logger)

    full_dataset = LatentCoTDataset(
        data_paths=cfg.data_paths,
        dataset_root=cfg.dataset_root,
        allow_no_observation=cfg.allow_no_observation,
        shuffle=cfg.shuffle_train,
        seed=cfg.seed,
    )
    dataset = _shard_dataset_for_rank(full_dataset, rank, world_size)
    logger.info(
        "Loaded %d samples on rank %d (global total: %d)",
        len(dataset),
        rank,
        len(full_dataset),
    )

    processor, model, collator, token_ids = _create_model_and_collator(
        cfg,
        stage1_1_vae_roundtrip=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator.collate_stage1_1,
    )

    model, device = _place_and_wrap_model(model, cfg, local_rank)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = (
        f".precompute_rep_checkpoint_rank{rank}.json"
        if world_size > 1
        else ".precompute_rep_checkpoint.json"
    )
    checkpoint_path = out_dir / checkpoint_name
    completed: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = set(json.load(f))
        logger.info("Resuming: %d samples already completed", len(completed))

    answer_start = torch.tensor(token_ids.answer_start_pattern, dtype=torch.long)
    latent_end = torch.tensor(token_ids.latent_end, dtype=torch.long)
    total_saved = 0

    def _save_checkpoint() -> None:
        with open(checkpoint_path, "w") as f:
            json.dump(sorted(completed), f, indent=0)
        logger.info("Checkpoint saved: %d samples completed", len(completed))

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Precompute teacher reps", disable=(rank != 0)):
            batch_metadata_infos = [
                f"{_prefix(cfg)}_{md['dataset_name']}_{md['sample_id']}"
                for md in batch["metadata"]
            ]
            if all(mi in completed for mi in batch_metadata_infos):
                continue
            pixel_values = batch.get("teacher_pixel_values", None)
            image_grid_thw = batch.get("teacher_image_grid_thw", None)
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            outputs = model(
                input_ids=batch["teacher_input_ids"].to(device),
                attention_mask=batch["teacher_attention_mask"].to(device),
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                latent_mode=False,
                output_hidden_states=True,
                loss_type=[],
            )

            if cfg.sft_stage2_align_poss == "obs":
                alignment_poss = batch["teacher_observation_poss"]
            else:
                alignment_poss = find_ids_poss(
                    batch["teacher_input_ids"],
                    answer_start,
                    latent_end,
                )

            reps = _extract_positions_hidden(
                outputs.hidden_states,
                alignment_poss,
                output_all_layers=cfg.output_hidden_states,
            )

            for i, rep in enumerate(reps):
                md = batch["metadata"][i]
                metadata_info = f"{_prefix(cfg)}_{md['dataset_name']}_{md['sample_id']}"
                if metadata_info in completed:
                    continue
                if cfg.sft_stage2_align_poss == "obs":
                    fname = f"rep_{metadata_info}.pt"
                else:
                    fname = f"rep_latent_end_{metadata_info}.pt"
                torch.save(
                    {"metadata_info": metadata_info, "latent": rep.detach().cpu()},
                    out_dir / fname,
                )
                completed.add(metadata_info)
                total_saved += 1
                if total_saved > 0 and total_saved % cfg.save_every == 0:
                    _save_checkpoint()

    if total_saved > 0 and total_saved % cfg.save_every != 0:
        _save_checkpoint()
    logger.info("Saved teacher reps to %s", out_dir)
