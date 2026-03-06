"""Single-sample checkpoint comparison for stage-wise inference."""

from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from qwen_latent_cot.inference.pipeline import ReflectionRegenerationPipeline
from qwen_latent_cot.models.qwen_image_backend import LocalQwenImageBackend
from qwen_latent_cot.models.reflector import QwenVLReflector
from qwen_latent_cot.utils import build_logger, load_json, save_json


@dataclass
class InferCompareConfig:
    data_paths: list[str]
    output_dir: str
    qwen_image_base_model: str = ""
    stage2_checkpoint: str = ""
    reflector_base_model: str = ""
    stage1_1_checkpoint: str = ""
    stage1_2_checkpoint: str = ""
    stage1_3_checkpoint: str = ""
    untrained_reflector_model: str = ""
    stage2_reflector_checkpoint: str = ""
    stages: list[str] | None = None
    sample_seed: int | None = None
    gen_seed: int | None = 42
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    aspect_ratio: str = "1:1"
    dtype: str = "bfloat16"
    init_image: str = "zeros"
    sample_source: str = "imgedit"


def _iter_sample_dirs(intermediate_path: Path) -> list[Path]:
    direct_samples: list[Path] = []
    source_samples: list[Path] = []

    for child in sorted(intermediate_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / "meta.json").is_file():
            direct_samples.append(child)
            continue
        for sub in sorted(child.iterdir()):
            if sub.is_dir() and (sub / "meta.json").is_file():
                source_samples.append(sub)
    return direct_samples or source_samples


def _resolve_gt_image(sample_dir: Path, meta: dict) -> str:
    gt_img_key = str(meta.get("gt_img", "") or "").strip()
    if gt_img_key not in {"img1", "img2"}:
        num_turns = meta.get("num_turns")
        if num_turns == 2:
            gt_img_key = "img2"
        else:
            gt_img_key = "img1"
    return str(sample_dir / f"{gt_img_key}.png")


def _pick_random_sample(data_paths: list[str], seed: int | None, sample_source: str = "imgedit") -> dict:
    sample_dirs: list[Path] = []
    for path in data_paths:
        root = Path(path)
        if not root.is_dir():
            raise ValueError(f"Invalid data path: {path}")
        sample_dirs.extend(_iter_sample_dirs(root))

    if not sample_dirs:
        raise RuntimeError("No valid sample dirs found from --data-path.")

    source_filter = str(sample_source or "").strip().lower()
    if source_filter and source_filter != "all":
        filtered_sample_dirs: list[Path] = []
        for sample_dir in sample_dirs:
            try:
                meta = load_json(str(sample_dir / "meta.json"))
            except Exception:
                continue
            source = str((meta or {}).get("source", "") or "").strip().lower()
            if source == source_filter:
                filtered_sample_dirs.append(sample_dir)
        sample_dirs = filtered_sample_dirs
        if not sample_dirs:
            raise RuntimeError(
                f"No valid sample dirs found for source='{source_filter}' from --data-path."
            )

    if seed is None:
        sample_dir = random.choice(sample_dirs)
    else:
        rng = random.Random(seed)
        sample_dir = rng.choice(sample_dirs)
    meta = load_json(str(sample_dir / "meta.json"))
    if not isinstance(meta, dict):
        raise RuntimeError(f"Invalid meta.json: {sample_dir / 'meta.json'}")

    prompt = str(meta.get("prompt", "") or "").strip()
    if not prompt:
        raise RuntimeError(f"Missing prompt in {sample_dir / 'meta.json'}")

    return {
        "sample_dir": str(sample_dir),
        "prompt": prompt,
        "meta": meta,
        "gt_image": _resolve_gt_image(sample_dir, meta),
    }


def _build_init_image(mode: str, backend: LocalQwenImageBackend) -> Image.Image | None:
    if mode == "none":
        return None
    if mode == "zeros":
        return Image.new("RGB", (int(backend.width), int(backend.height)), (0, 0, 0))
    raise ValueError(f"Unsupported init image mode: {mode}")


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_single_case(
    *,
    tag: str,
    prompt: str,
    output_dir: Path,
    image_model: str,
    reflector_model: str,
    reflector_base_model: str,
    cfg: InferCompareConfig,
    logger,
) -> dict:
    logger.info("Running case `%s` with image=%s reflector=%s", tag, image_model, reflector_model)
    backend = LocalQwenImageBackend(
        model_path=image_model,
        dtype=cfg.dtype,
        aspect_ratio=cfg.aspect_ratio,
    )
    reflector = QwenVLReflector(
        model_path=reflector_model,
        base_model_path=reflector_base_model,
    )
    pipeline = ReflectionRegenerationPipeline(image_backend=backend, reflector=reflector)

    case_dir = output_dir / tag
    init_image = _build_init_image(cfg.init_image, backend)
    result = pipeline.run_and_save(
        prompt=prompt,
        output_dir=str(case_dir),
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        seed=cfg.gen_seed,
        init_image=init_image,
    )

    del pipeline
    del reflector
    del backend
    _cleanup_cuda()
    return result


def run_infer_compare(cfg: InferCompareConfig) -> None:
    logger = build_logger("qwen_latent_cot.infer_compare")
    sample = _pick_random_sample(cfg.data_paths, seed=cfg.sample_seed, sample_source=cfg.sample_source)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supported = ["untrained", "stage1-1", "stage1-2", "stage1-3", "stage2"]
    selected = cfg.stages or supported
    stages: list[str] = []
    for item in selected:
        tag = str(item).strip()
        if tag not in supported:
            raise ValueError(f"Unsupported stage: {tag}. choices={supported}")
        if tag not in stages:
            stages.append(tag)
    if not stages:
        raise ValueError("No stage selected. Please set --stages.")

    def _require(name: str, value: str) -> str:
        if not str(value or "").strip():
            raise ValueError(f"{name} is required for selected stages={stages}")
        return str(value).strip()

    baseline_reflector = str(cfg.untrained_reflector_model or "").strip() or str(cfg.reflector_base_model or "").strip()
    stage2_reflector = (
        str(cfg.stage2_reflector_checkpoint or "").strip()
        or str(cfg.stage1_3_checkpoint or "").strip()
        or str(cfg.reflector_base_model or "").strip()
    )

    runs: dict[str, dict] = {}
    for tag in stages:
        if tag == "untrained":
            image_model = _require("--qwen-image-base-model", cfg.qwen_image_base_model)
            reflector_model = _require(
                "--untrained-reflector-model or --reflector-base-model",
                baseline_reflector,
            )
        elif tag == "stage1-1":
            image_model = _require("--qwen-image-base-model", cfg.qwen_image_base_model)
            reflector_model = _require("--stage1-1-checkpoint", cfg.stage1_1_checkpoint)
        elif tag == "stage1-2":
            image_model = _require("--qwen-image-base-model", cfg.qwen_image_base_model)
            reflector_model = _require("--stage1-2-checkpoint", cfg.stage1_2_checkpoint)
        elif tag == "stage1-3":
            image_model = _require("--qwen-image-base-model", cfg.qwen_image_base_model)
            reflector_model = _require("--stage1-3-checkpoint", cfg.stage1_3_checkpoint)
        elif tag == "stage2":
            image_model = _require("--stage2-checkpoint", cfg.stage2_checkpoint)
            reflector_model = _require(
                "--stage2-reflector-checkpoint or --stage1-3-checkpoint or --reflector-base-model",
                stage2_reflector,
            )
        else:
            raise ValueError(f"Unsupported stage: {tag}. choices={supported}")

        runs[tag] = _run_single_case(
            tag=tag,
            prompt=sample["prompt"],
            output_dir=output_dir,
            image_model=image_model,
            reflector_model=reflector_model,
            reflector_base_model=cfg.reflector_base_model,
            cfg=cfg,
            logger=logger,
        )

    summary = {
        "sample": {
            "prompt": sample["prompt"],
            "sample_dir": sample["sample_dir"],
            "gt_image": sample["gt_image"],
            "meta": sample["meta"],
        },
        "settings": {
            "stages": stages,
            "sample_source": cfg.sample_source,
            "num_inference_steps": cfg.num_inference_steps,
            "guidance_scale": cfg.guidance_scale,
            "gen_seed": cfg.gen_seed,
            "sample_seed": cfg.sample_seed,
            "aspect_ratio": cfg.aspect_ratio,
            "dtype": cfg.dtype,
            "init_image": cfg.init_image,
        },
        "cases": runs,
    }
    save_json(output_dir / "compare_summary.json", summary)
    logger.info("Finished compare run. Summary saved to %s", output_dir / "compare_summary.json")
