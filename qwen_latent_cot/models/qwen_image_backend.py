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
    """Backend for local Qwen-image runtime.

    This backend expects an importable local runtime implementation. For example,
    a custom package exposing `QwenImagePipeline` with `from_pretrained` and
    `__call__/edit` APIs.
    """

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        try:
            from qwen_image import QwenImagePipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Local Qwen-image runtime not found. Install your local `qwen_image` package "
                "or use `--backend mock`/`--backend openai_compat`."
            ) from exc

        self.pipe = QwenImagePipeline.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device=device,
            dtype=dtype,
        )

    def generate(
        self,
        prompt: str,
        image: Image.Image | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int | None = None,
    ) -> Image.Image:
        if image is None:
            out = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        else:
            if not hasattr(self.pipe, "edit"):
                raise RuntimeError("Current local Qwen-image backend does not support image editing")
            out = self.pipe.edit(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )

        if hasattr(out, "images"):
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
