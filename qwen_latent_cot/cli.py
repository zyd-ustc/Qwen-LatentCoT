"""Command-line entrypoint for Qwen-LatentCoT."""

from __future__ import annotations

import argparse
import os

import torch

from qwen_latent_cot.inference import ReflectionRegenerationPipeline
from qwen_latent_cot.models.qwen_image_backend import (
    LocalQwenImageBackend,
    MockQwenImageBackend,
    OpenAICompatQwenImageBackend,
)
from qwen_latent_cot.models.reflector import HeuristicReflector, QwenVLReflector
from qwen_latent_cot.training import (
    PrecomputeConfig,
    TrainConfig,
    run_precompute_teacher_latents,
    run_precompute_teacher_reps,
    run_training,
)
from qwen_latent_cot.utils import build_logger


def _base_training_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--allow-no-observation", action="store_true")
    parser.add_argument("--shuffle-train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--image-resize", type=str, default="global", choices=["global", "none"])

    parser.add_argument("--not-use-4d", action="store_true")
    parser.add_argument("--not-mask-image", action="store_true")
    parser.add_argument("--mask-latent", action="store_true")
    parser.add_argument("--observation-tokens-cannot-see-question-image", action="store_true")
    parser.add_argument("--observation-tokens-only-see-question-and-latent", action="store_true")
    parser.add_argument("--latent-can-see-all-previous", action="store_true", default=True)
    parser.add_argument("--disable-latent-see-all-previous", action="store_false", dest="latent_can_see_all_previous")
    parser.add_argument("--mask-question-image", action="store_true")

    parser.add_argument("--sft-stage2-align-poss", type=str, default="obs", choices=["obs", "latent_end"])
    parser.add_argument("--sft-stage2-global-img-tokens", type=int, default=1500)
    parser.add_argument("--sft-stage2-per-img-tokens", type=int, default=1280)
    parser.add_argument("--sft-stage3-img-tokens", type=int, default=2000)

    parser.add_argument("--log-file", type=str, default=None)


def _add_train_parser(subparsers) -> None:
    p = subparsers.add_parser("train", help="Run stage1-1/1-2/1-3 training")
    _base_training_parser(p)

    p.add_argument("--stage", type=str, required=True, choices=["stage1-1", "stage1-2", "stage1-3"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-total-limit", type=int, default=5)

    p.add_argument("--ce-emphasize-factor", type=float, default=1.0)
    p.add_argument("--alignment-layer", type=str, default="all_layers")
    p.add_argument("--alignment-weight", type=float, default=1.0)
    p.add_argument("--emphasize-latent-weight", type=float, default=1.0)
    p.add_argument("--only-predict-obs", action="store_true")

    p.add_argument("--teacher-reps-dir", type=str, default=None)
    p.add_argument("--teacher-latent-dir", type=str, default=None)
    p.add_argument("--resume-from-checkpoint", action="store_true")


def _add_precompute_parsers(subparsers) -> None:
    latent = subparsers.add_parser("precompute-latent", help="Precompute teacher latent targets")
    _base_training_parser(latent)
    latent.add_argument("--output-hidden-states", action="store_true")
    latent.add_argument("--output-latent-embeds", action="store_true")

    rep = subparsers.add_parser("precompute-rep", help="Precompute teacher hidden states")
    _base_training_parser(rep)
    rep.add_argument("--output-hidden-states", action="store_true")


def _add_infer_parser(subparsers) -> None:
    p = subparsers.add_parser("infer", help="Run draft->reflection->refine pipeline")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--backend", type=str, default="mock", choices=["mock", "local", "openai_compat"])
    p.add_argument("--qwen-image-model", type=str, default=None)
    p.add_argument("--openai-base-url", type=str, default=None)
    p.add_argument("--openai-api-key", type=str, default=None)
    p.add_argument("--openai-model", type=str, default="qwen-image")

    p.add_argument("--reflector", type=str, default="heuristic", choices=["heuristic", "qwen_vl"])
    p.add_argument("--reflector-model", type=str, default=None)


def _build_image_backend(args):
    if args.backend == "mock":
        return MockQwenImageBackend()
    if args.backend == "local":
        if not args.qwen_image_model:
            raise ValueError("--qwen-image-model is required for --backend local")
        return LocalQwenImageBackend(model_path=args.qwen_image_model)
    if args.backend == "openai_compat":
        base_url = args.openai_base_url or os.environ.get("OPENAI_BASE_URL")
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not base_url or not api_key:
            raise ValueError("OPENAI-compatible backend requires --openai-base-url and --openai-api-key")
        return OpenAICompatQwenImageBackend(base_url=base_url, api_key=api_key, model=args.openai_model)
    raise ValueError(f"Unsupported backend: {args.backend}")


def _build_reflector(args):
    if args.reflector == "heuristic":
        return HeuristicReflector()
    if args.reflector == "qwen_vl":
        if not args.reflector_model:
            raise ValueError("--reflector-model is required for --reflector qwen_vl")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        return QwenVLReflector(model_path=args.reflector_model, dtype=dtype)
    raise ValueError(f"Unsupported reflector: {args.reflector}")


def main() -> None:
    parser = argparse.ArgumentParser("qwen-latent-cot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_infer_parser(subparsers)
    _add_train_parser(subparsers)
    _add_precompute_parsers(subparsers)

    args = parser.parse_args()

    if args.command == "infer":
        logger = build_logger("qwen_latent_cot.infer")
        image_backend = _build_image_backend(args)
        reflector = _build_reflector(args)
        pipeline = ReflectionRegenerationPipeline(image_backend=image_backend, reflector=reflector)
        result = pipeline.run_and_save(
            prompt=args.prompt,
            output_dir=args.output_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        logger.info("Saved draft image: %s", result["draft"])
        logger.info("Saved refined image: %s", result["refined"])
        logger.info("Saved metadata: %s", result["meta"])
        logger.info("Reflection: %s", result["reflection"])
        return

    if args.command == "train":
        cfg = TrainConfig(
            stage=args.stage,
            model_path=args.model_path,
            data_paths=args.data_path,
            output_dir=args.output_dir,
            dataset_root=args.dataset_root,
            allow_no_observation=args.allow_no_observation,
            shuffle_train=args.shuffle_train,
            seed=args.seed,
            dtype=args.dtype,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit,
            latent_size=args.latent_size,
            image_resize=args.image_resize,
            not_use_4d=args.not_use_4d,
            not_mask_image=args.not_mask_image,
            mask_latent=args.mask_latent,
            observation_tokens_cannot_see_question_image=args.observation_tokens_cannot_see_question_image,
            observation_tokens_only_see_question_and_latent=args.observation_tokens_only_see_question_and_latent,
            latent_can_see_all_previous=args.latent_can_see_all_previous,
            mask_question_image=args.mask_question_image,
            only_predict_obs=args.only_predict_obs,
            sft_stage2_global_img_tokens=args.sft_stage2_global_img_tokens,
            sft_stage2_per_img_tokens=args.sft_stage2_per_img_tokens,
            sft_stage3_img_tokens=args.sft_stage3_img_tokens,
            sft_stage2_align_poss=args.sft_stage2_align_poss,
            ce_emphasize_factor=args.ce_emphasize_factor,
            alignment_layer=args.alignment_layer,
            alignment_weight=args.alignment_weight,
            emphasize_latent_weight=args.emphasize_latent_weight,
            teacher_reps_dir=args.teacher_reps_dir,
            teacher_latent_dir=args.teacher_latent_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            log_file=args.log_file,
        )
        run_training(cfg)
        return

    if args.command in {"precompute-latent", "precompute-rep"}:
        cfg = PrecomputeConfig(
            model_path=args.model_path,
            data_paths=args.data_path,
            output_dir=args.output_dir,
            dataset_root=args.dataset_root,
            allow_no_observation=args.allow_no_observation,
            shuffle_train=args.shuffle_train,
            seed=args.seed,
            dtype=args.dtype,
            batch_size=args.batch_size,
            latent_size=args.latent_size,
            image_resize=args.image_resize,
            not_use_4d=args.not_use_4d,
            not_mask_image=args.not_mask_image,
            mask_latent=args.mask_latent,
            observation_tokens_cannot_see_question_image=args.observation_tokens_cannot_see_question_image,
            observation_tokens_only_see_question_and_latent=args.observation_tokens_only_see_question_and_latent,
            latent_can_see_all_previous=args.latent_can_see_all_previous,
            mask_question_image=args.mask_question_image,
            sft_stage2_align_poss=args.sft_stage2_align_poss,
            output_hidden_states=args.output_hidden_states,
            output_latent_embeds=getattr(args, "output_latent_embeds", False),
            log_file=args.log_file,
        )

        if args.command == "precompute-latent":
            run_precompute_teacher_latents(cfg)
        else:
            run_precompute_teacher_reps(cfg)
        return


if __name__ == "__main__":
    main()
