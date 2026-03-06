"""Stage2 end-to-end generation training."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qwen_latent_cot.data import Stage2Dataset
from qwen_latent_cot.utils import build_logger, seed_everything


@dataclass
class Stage2TrainConfig:
    qwen_image_model_path: str
    data_paths: list[str]
    output_dir: str
    dataset_root: str = ""
    deepspeed: str = ""
    seed: int = 42
    dtype: str = "bfloat16"
    epochs: int = 1
    batch_size: int = 1
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 10
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    max_train_steps: int = 0
    image_size: int = 512
    diffusion_weight: float = 1.0
    recon_weight: float = 0.1
    lpips_weight: float = 0.0
    use_prompt: bool = True
    use_reflection: bool = True
    use_vlat_tokens: bool = True
    latent_token_repeat: int = 8
    use_prev_image: bool = False
    shuffle_train: bool = False
    log_file: str | None = None


def _dtype_from_name(name: str) -> torch.dtype:
    key = name.lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _to_tensor(images: list, image_size: int) -> torch.Tensor:
    import numpy as np

    arrs = []
    for img in images:
        img = img.resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr * 2.0) - 1.0
        arrs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(arrs, dim=0)


def _collate_stage2(batch: list[dict], image_size: int) -> dict:
    return {
        "condition_text": [x["condition_text"] for x in batch],
        "prompt": [x["prompt"] for x in batch],
        "reflection": [x["reflection"] for x in batch],
        "gt_images": _to_tensor([x["gt_image"] for x in batch], image_size=image_size),
        "prev_images": None
        if all(x["prev_image"] is None for x in batch)
        else _to_tensor(
            [x["prev_image"] if x["prev_image"] is not None else x["gt_image"] for x in batch],
            image_size=image_size,
        ),
        "metadata": [x["metadata"] for x in batch],
    }


def _init_pipe_and_trainable(cfg: Stage2TrainConfig, logger):
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        cfg.qwen_image_model_path,
        torch_dtype=_dtype_from_name(cfg.dtype),
        trust_remote_code=True,
    )

    trainable = None
    for cand in ("transformer", "unet"):
        if hasattr(pipe, cand):
            trainable = getattr(pipe, cand)
            break
    if trainable is None:
        raise RuntimeError("Cannot find trainable denoiser module (`transformer` or `unet`) in pipeline.")

    for maybe_freeze in ("vae", "text_encoder", "text_encoder_2", "image_encoder"):
        mod = getattr(pipe, maybe_freeze, None)
        if mod is not None:
            mod.requires_grad_(False)
            mod.eval()

    trainable.requires_grad_(True)
    trainable.train()
    logger.info("Stage2 trainable module: %s", type(trainable).__name__)
    return pipe, trainable


def _encode_prompt(pipe, texts: list[str], device: torch.device):
    if not hasattr(pipe, "encode_prompt"):
        return {}
    try:
        out = pipe.encode_prompt(
            prompt=texts,
            device=device,
            do_classifier_free_guidance=False,
        )
    except TypeError:
        try:
            out = pipe.encode_prompt(prompt=texts, device=device)
        except Exception:
            return {}
    if isinstance(out, tuple):
        # Common diffusers return shape:
        # (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        prompt_embeds = out[0] if len(out) > 0 else None
        pooled = out[2] if len(out) > 2 else None
        res = {}
        if prompt_embeds is not None:
            res["encoder_hidden_states"] = prompt_embeds
        if pooled is not None:
            res["pooled_projections"] = pooled
        return res
    if isinstance(out, torch.Tensor):
        return {"encoder_hidden_states": out}
    return {}


def _call_denoiser(module, noisy_latents, timesteps, prompt_kwargs: dict):
    sig = inspect.signature(module.forward)
    kwargs = {}

    if "sample" in sig.parameters:
        kwargs["sample"] = noisy_latents
    elif "hidden_states" in sig.parameters:
        kwargs["hidden_states"] = noisy_latents
    else:
        kwargs["sample"] = noisy_latents

    if "timestep" in sig.parameters:
        kwargs["timestep"] = timesteps
    elif "timesteps" in sig.parameters:
        kwargs["timesteps"] = timesteps

    if "encoder_hidden_states" in sig.parameters and "encoder_hidden_states" in prompt_kwargs:
        kwargs["encoder_hidden_states"] = prompt_kwargs["encoder_hidden_states"]
    if "pooled_projections" in sig.parameters and "pooled_projections" in prompt_kwargs:
        kwargs["pooled_projections"] = prompt_kwargs["pooled_projections"]

    out = module(**kwargs)
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "sample"):
        return out.sample
    return out


def _compute_stage2_losses(
    *,
    pipe,
    denoiser,
    gt_images: torch.Tensor,
    cond_text: list[str],
    device: torch.device,
    diffusion_weight: float,
    recon_weight: float,
):
    dtype = next(denoiser.parameters()).dtype
    images = gt_images.to(device=device, dtype=torch.float32)

    if getattr(pipe, "vae", None) is not None:
        latents = pipe.vae.encode(images).latent_dist.sample()
        scale = float(getattr(pipe.vae.config, "scaling_factor", 1.0))
        latents = latents * scale
    else:
        latents = images

    bsz = latents.size(0)
    noise = torch.randn_like(latents)
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is None:
        raise RuntimeError("Pipeline scheduler is required for stage2 diffusion loss.")
    timesteps = torch.randint(
        low=0,
        high=int(getattr(scheduler.config, "num_train_timesteps", 1000)),
        size=(bsz,),
        device=device,
        dtype=torch.long,
    )
    noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(dtype=dtype)

    prompt_kwargs = _encode_prompt(pipe, cond_text, device=device)
    prompt_kwargs = {
        k: (v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) else v)
        for k, v in prompt_kwargs.items()
    }
    model_pred = _call_denoiser(denoiser, noisy_latents, timesteps, prompt_kwargs)

    target = noise.to(dtype=model_pred.dtype)
    diffusion_loss = F.mse_loss(model_pred.float(), target.float())

    recon_loss = torch.tensor(0.0, device=device)
    if recon_weight > 0:
        alphas_cumprod = scheduler.alphas_cumprod.to(device)
        a_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(dtype=model_pred.dtype)
        pred_x0 = (noisy_latents - (1.0 - a_t).sqrt() * model_pred) / a_t.sqrt().clamp(min=1e-6)

        if getattr(pipe, "vae", None) is not None:
            scale = float(getattr(pipe.vae.config, "scaling_factor", 1.0))
            pred_img = pipe.vae.decode(pred_x0 / scale).sample
        else:
            pred_img = pred_x0
        recon_loss = F.l1_loss(pred_img.float(), images.float())

    total_loss = (diffusion_weight * diffusion_loss) + (recon_weight * recon_loss)
    return total_loss, diffusion_loss.detach(), recon_loss.detach()


def run_stage2_training(cfg: Stage2TrainConfig) -> None:
    from accelerate import Accelerator, DeepSpeedPlugin

    seed_everything(cfg.seed)
    logger = build_logger("qwen_latent_cot.stage2", cfg.log_file)

    ds_plugin = DeepSpeedPlugin(hf_ds_config=cfg.deepspeed) if cfg.deepspeed else None
    mixed_precision = "no"
    if cfg.dtype.lower() in {"bf16", "bfloat16"}:
        mixed_precision = "bf16"
    elif cfg.dtype.lower() in {"fp16", "float16"}:
        mixed_precision = "fp16"

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum_steps,
        mixed_precision=mixed_precision,
        deepspeed_plugin=ds_plugin,
    )
    device = accelerator.device

    dataset = Stage2Dataset(
        data_paths=cfg.data_paths,
        dataset_root=cfg.dataset_root,
        use_prompt=cfg.use_prompt,
        use_reflection=cfg.use_reflection,
        use_vlat_tokens=cfg.use_vlat_tokens,
        latent_token_repeat=cfg.latent_token_repeat,
        use_prev_image=cfg.use_prev_image,
        shuffle=cfg.shuffle_train,
        seed=cfg.seed,
    )
    logger.info("Loaded %d stage2 samples", len(dataset))
    if len(dataset) == 0:
        raise RuntimeError("Stage2 dataset is empty. Please check `--data-path`.")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train,
        collate_fn=lambda x: _collate_stage2(x, image_size=cfg.image_size),
    )

    pipe, denoiser = _init_pipe_and_trainable(cfg, logger)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=cfg.learning_rate)

    denoiser, optimizer, dataloader = accelerator.prepare(denoiser, optimizer, dataloader)
    if hasattr(pipe, "to"):
        pipe.to(device)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.epochs):
        for batch in dataloader:
            with accelerator.accumulate(denoiser):
                loss, diff_loss, rec_loss = _compute_stage2_losses(
                    pipe=pipe,
                    denoiser=denoiser,
                    gt_images=batch["gt_images"],
                    cond_text=batch["condition_text"],
                    device=device,
                    diffusion_weight=cfg.diffusion_weight,
                    recon_weight=cfg.recon_weight,
                )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if accelerator.is_main_process and (global_step % cfg.logging_steps == 0):
                logger.info(
                    "step=%d loss=%.6f diffusion=%.6f recon=%.6f",
                    global_step,
                    float(loss.detach().item()),
                    float(diff_loss.item()),
                    float(rec_loss.item()),
                )

            if accelerator.is_main_process and (global_step % cfg.save_steps == 0):
                unwrapped = accelerator.unwrap_model(denoiser)
                if hasattr(pipe, "transformer"):
                    pipe.transformer = unwrapped
                elif hasattr(pipe, "unet"):
                    pipe.unet = unwrapped
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                pipe.save_pretrained(str(ckpt_dir))
                logger.info("Saved stage2 checkpoint to %s", ckpt_dir)

            if cfg.max_train_steps > 0 and global_step >= cfg.max_train_steps:
                break

        if cfg.max_train_steps > 0 and global_step >= cfg.max_train_steps:
            break
        if accelerator.is_main_process:
            logger.info("Finished epoch %d", epoch + 1)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(denoiser)
        if hasattr(pipe, "transformer"):
            pipe.transformer = unwrapped
        elif hasattr(pipe, "unet"):
            pipe.unet = unwrapped
        pipe.save_pretrained(str(output_dir))
        logger.info("Stage2 training finished. Saved to %s", output_dir)

