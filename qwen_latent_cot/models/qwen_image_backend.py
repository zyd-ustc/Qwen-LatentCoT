"""Backends for Qwen-image generation."""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from qwen_latent_cot.utils import render_mock_image


class ImageGeneratorBackend(Protocol):
    def generate(
        self,
        prompt: str,
        image: Image.Image | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int | None = None,
    ) -> Image.Image:
        ...


@dataclass
class MockQwenImageBackend:
    width: int = 1024
    height: int = 1024

    def generate(
        self,
        prompt: str,
        image: Image.Image | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int | None = None,
    ) -> Image.Image:
        del image, num_inference_steps, guidance_scale, seed
        return render_mock_image(prompt, width=self.width, height=self.height)


class LocalQwenImageBackend:
    """Backend for local Qwen-image model loaded via diffusers.

    Uses ``DiffusionPipeline.from_pretrained`` to load the QwenImagePipeline
    from a local checkpoint directory (e.g. ``qwen_latent_cot/models/Qwen-Image``).
    """

    # Default resolutions (width, height) keyed by aspect ratio label.
    ASPECT_RATIOS: dict[str, tuple[int, int]] = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        device: str | None = None,
        dtype: str = "bfloat16",
        aspect_ratio: str = "1:1",
    ) -> None:
        import torch as _torch

        try:
            from diffusers import DiffusionPipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "diffusers is required for the local Qwen-image backend. "
                "Install the latest version: pip install git+https://github.com/huggingface/diffusers"
            ) from exc

        _dtype_map = {
            "float16": _torch.float16,
            "fp16": _torch.float16,
            "bfloat16": _torch.bfloat16,
            "bf16": _torch.bfloat16,
            "float32": _torch.float32,
            "fp32": _torch.float32,
        }
        torch_dtype = _dtype_map.get(dtype.lower(), _torch.bfloat16)

        if device is None:
            device = "cuda" if _torch.cuda.is_available() else "cpu"

        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.pipe = self.pipe.to(device)

        if aspect_ratio not in self.ASPECT_RATIOS:
            raise ValueError(
                f"Unsupported aspect_ratio '{aspect_ratio}'. "
                f"Choose from {list(self.ASPECT_RATIOS.keys())}"
            )
        self.width, self.height = self.ASPECT_RATIOS[aspect_ratio]

    def generate(
        self,
        prompt: str,
        image: Image.Image | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int | None = None,
    ) -> Image.Image:
        import torch as _torch

        # Build reproducible generator if seed is given.
        generator = None
        if seed is not None:
            generator = _torch.Generator(device=self.device).manual_seed(seed)

        kwargs: dict = dict(
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=num_inference_steps,
            true_cfg_scale=guidance_scale,
            generator=generator,
        )

        if image is None:
            kwargs["width"] = self.width
            kwargs["height"] = self.height
        else:
            kwargs["image"] = image

        try:
            out = self.pipe(**kwargs)
        except TypeError as exc:
            if image is not None and "image" in kwargs and "image" in str(exc):
                kwargs.pop("image", None)
                kwargs["width"] = self.width
                kwargs["height"] = self.height
                out = self.pipe(**kwargs)
            else:
                raise

        if hasattr(out, "images") and out.images:
            return out.images[0]
        if isinstance(out, Image.Image):
            return out
        raise RuntimeError("Unsupported output from local Qwen-image backend")


class OpenAICompatQwenImageBackend:
    """Backend for OpenAI-compatible image generation endpoints.

    Expected API:
    POST {base_url}/images/generations
    {
      "model": "qwen-image",
      "prompt": "...",
      ...
    }
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body)

    def generate(
        self,
        prompt: str,
        image: Image.Image | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int | None = None,
    ) -> Image.Image:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if image is not None:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload["image"] = base64.b64encode(buffer.getvalue()).decode("utf-8")

        data = self._post_json("/images/generations", payload)
        if "data" not in data or not data["data"]:
            raise RuntimeError(f"Invalid response from image backend: {data}")

        item = data["data"][0]
        if "b64_json" not in item:
            raise RuntimeError(f"Missing b64_json in response: {data}")

        image_bytes = base64.b64decode(item["b64_json"])
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
