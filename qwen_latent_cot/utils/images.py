"""Image helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


def load_image(path: str | Path | bytes | dict | Image.Image) -> Image.Image:
    if isinstance(path, Image.Image):
        img = path
    elif isinstance(path, (bytes, bytearray)):
        img = Image.open(BytesIO(path))
    elif isinstance(path, dict):
        # 仅支持通过 bytes 直接读图，不解析 path 键
        raw = path.get("bytes")
        if raw is not None:
            img = Image.open(BytesIO(raw))
        else:
            raise ValueError("Image dict must contain 'bytes'.")
    else:
        img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def save_image(img: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def resize_by_token_budget(
    images: list[Image.Image],
    global_max_pixels: int = 2000 * 28 * 28,
    per_img_max_pixels: int = 1280 * 28 * 28,
    divisor: int = 28,
) -> tuple[list[Image.Image], list[tuple[int, int]] | None]:
    total = sum(img.width * img.height for img in images)
    if total <= global_max_pixels:
        return images, None

    import math

    ratio = math.sqrt(global_max_pixels / total)
    processed: list[Image.Image] = []
    new_sizes: list[tuple[int, int]] = []

    for img in images:
        w, h = int(img.width * ratio), int(img.height * ratio)
        w = max(divisor, (w // divisor) * divisor)
        h = max(divisor, (h // divisor) * divisor)

        if w * h > per_img_max_pixels:
            r = math.sqrt(per_img_max_pixels / (w * h))
            w = max(divisor, (int(w * r) // divisor) * divisor)
            h = max(divisor, (int(h * r) // divisor) * divisor)

        processed.append(img.resize((w, h), Image.BICUBIC))
        new_sizes.append((w, h))

    return processed, new_sizes


def render_mock_image(prompt: str, width: int = 1024, height: int = 1024) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(236, 240, 245))
    draw = ImageDraw.Draw(img)
    text = prompt.strip().replace("\n", " ")[:180] or "(empty prompt)"
    draw.rectangle((24, 24, width - 24, height - 24), outline=(36, 60, 92), width=3)
    draw.text((40, 40), "MOCK Qwen-Image", fill=(20, 35, 56))
    draw.text((40, 90), text, fill=(20, 35, 56))
    return img
