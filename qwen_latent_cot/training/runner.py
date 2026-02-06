"""Training entry for stage1-1/1-2/1-3."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import TrainingArguments

from qwen_latent_cot.data import LatentCoTDataset
from qwen_latent_cot.data.collators import CollatorConfig, StageCollator
from qwen_latent_cot.models import (
    add_latent_special_tokens,
    attach_special_ids_to_model,
    resolve_special_token_ids,
)
from qwen_latent_cot.models.latent_student import LatentQwenVLWrapper, freeze_visual_encoder
from qwen_latent_cot.models.loaders import load_qwen2_5_vl
from qwen_latent_cot.training.trainers import Stage11Trainer, Stage12Trainer, Stage13Trainer
from qwen_latent_cot.utils import build_logger, seed_everything


@dataclass
class TrainConfig:
    stage: str
    model_path: str
    data_paths: list[str]
    output_dir: str
    dataset_root: str = ""
    allow_no_observation: bool = False
    shuffle_train: bool = False
    seed: int = 42
    dtype: str = "bfloat16"
    epochs: int = 1
    batch_size: int = 1
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 10
    save_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 5
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
    ce_emphasize_factor: float = 1.0
    alignment_layer: str = "all_layers"
    alignment_weight: float = 1.0
    emphasize_latent_weight: float = 1.0
    teacher_reps_dir: str | None = None
    teacher_latent_dir: str | None = None
    resume_from_checkpoint: bool = False
    log_file: str | None = None


def run_training(cfg: TrainConfig) -> None:
    logger = build_logger("qwen_latent_cot.train", cfg.log_file)
    seed_everything(cfg.seed)

    processor, base_model = load_qwen2_5_vl(cfg.model_path, dtype=cfg.dtype)
    add_latent_special_tokens(processor)
    try:
        base_model.resize_token_embeddings(len(processor.tokenizer))
    except Exception:
        pass

    token_ids = resolve_special_token_ids(processor)
    attach_special_ids_to_model(base_model, token_ids)
    freeze_visual_encoder(base_model)

    model = LatentQwenVLWrapper(base_model)

    dataset = LatentCoTDataset(
        data_paths=cfg.data_paths,
        dataset_root=cfg.dataset_root,
        allow_no_observation=cfg.allow_no_observation,
        shuffle=cfg.shuffle_train,
        seed=cfg.seed,
    )
    logger.info("Loaded %d valid samples", len(dataset))

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
            only_predict_obs=cfg.only_predict_obs,
            sft_stage2_global_img_tokens=cfg.sft_stage2_global_img_tokens,
            sft_stage2_per_img_tokens=cfg.sft_stage2_per_img_tokens,
            sft_stage3_img_tokens=cfg.sft_stage3_img_tokens,
            sft_stage2_align_poss=cfg.sft_stage2_align_poss,
        ),
    )

    if cfg.stage == "stage1-1":
        trainer_cls = Stage11Trainer
        data_collator = collator.collate_stage1_1
    elif cfg.stage == "stage1-2":
        if not cfg.teacher_reps_dir:
            raise ValueError("stage1-2 requires `teacher_reps_dir`.")
        trainer_cls = Stage12Trainer
        data_collator = collator.collate_stage1_2
    elif cfg.stage == "stage1-3":
        if not cfg.teacher_latent_dir:
            raise ValueError("stage1-3 requires `teacher_latent_dir`.")
        trainer_cls = Stage13Trainer
        data_collator = collator.collate_stage1_3
    else:
        raise ValueError(f"Unsupported stage: {cfg.stage}")

    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        save_strategy="steps",
        logging_strategy="steps",
        bf16=(cfg.dtype.lower() in {"bf16", "bfloat16"}),
        fp16=(cfg.dtype.lower() in {"fp16", "float16"}),
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        gradient_checkpointing=True,
    )

    # Make custom fields available in Trainer subclasses.
    setattr(train_args, "ce_emphasize_factor", cfg.ce_emphasize_factor)
    setattr(train_args, "alignment_layer", cfg.alignment_layer)
    setattr(train_args, "alignment_weight", cfg.alignment_weight)
    setattr(train_args, "emphasize_latent_weight", cfg.emphasize_latent_weight)
    setattr(train_args, "teacher_reps_dir", cfg.teacher_reps_dir)
    setattr(train_args, "teacher_latent_dir", cfg.teacher_latent_dir)
    setattr(train_args, "sft_stage2_align_poss", cfg.sft_stage2_align_poss)

    trainer = trainer_cls(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)

    logger.info("Training finished. Saved to %s", cfg.output_dir)
