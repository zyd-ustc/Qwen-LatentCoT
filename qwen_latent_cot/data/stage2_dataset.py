"""Stage2 dataset for end-to-end final image supervision."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from qwen_latent_cot.utils import load_json


@dataclass
class Stage2Sample:
    prompt: str
    reflection: str
    condition_text: str
    final_image_path: str
    prev_image_path: str | None
    metadata: dict


def _clean_reflection_text(text: str) -> str:
    s = str(text or "")
    for tok in (
        "<|reflection_start|>",
        "<|reflection_end|>",
        "<|analysis_start|>",
        "<|analysis_end|>",
        "<|final_start|>",
        "<|final_end|>",
    ):
        s = s.replace(tok, "")
    s = s.strip()
    return "\n".join(line.strip() for line in s.splitlines()).strip()


def _iter_sample_dirs(intermediate_path: Path) -> list[Path]:
    direct_samples: list[Path] = []
    source_samples: list[Path] = []

    for child in sorted(intermediate_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / "meta.json").is_file():
            direct_samples.append(child)
            continue
        has_nested = False
        for sub in sorted(child.iterdir()):
            if not sub.is_dir():
                continue
            if (sub / "meta.json").is_file():
                source_samples.append(sub)
                has_nested = True
        if has_nested:
            continue
    return direct_samples or source_samples


def _build_condition_text(
    *,
    prompt: str,
    reflection: str,
    use_prompt: bool,
    use_reflection: bool,
    use_vlat_tokens: bool,
    latent_token_repeat: int,
) -> str:
    parts: list[str] = []
    if use_prompt:
        parts.append(f"Goal: {prompt}")
    if use_reflection and reflection:
        parts.append(f"Reflection: {reflection}")
    if use_vlat_tokens and latent_token_repeat > 0:
        latent = " ".join(["<|vlat_start|><|vlat_end|>"] * latent_token_repeat)
        parts.append(f"LatentHints: {latent}")
    if not parts:
        # Always provide at least one condition text field.
        parts.append(f"Goal: {prompt}")
    return "\n".join(parts).strip()


class Stage2Dataset(Dataset):
    """Load CoRT intermediate data and expose Stage2 train tuples."""

    def __init__(
        self,
        data_paths: list[str],
        dataset_root: str = "",
        use_prompt: bool = True,
        use_reflection: bool = True,
        use_vlat_tokens: bool = True,
        latent_token_repeat: int = 8,
        use_prev_image: bool = False,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.samples: list[Stage2Sample] = []

        for path in data_paths:
            path_obj = Path(path)
            if not path_obj.is_dir():
                raise ValueError(f"Only intermediate directories are supported: {path}")

            for sample_dir in _iter_sample_dirs(path_obj):
                meta_path = sample_dir / "meta.json"
                meta = load_json(str(meta_path))
                if not isinstance(meta, dict):
                    continue

                prompt = str(meta.get("prompt", "") or "").strip()
                source = str(meta.get("source", "") or "").strip()
                source_id = str(meta.get("source_id", "") or "").strip()
                num_turns = meta.get("num_turns")
                if not prompt or not source or not source_id or not isinstance(num_turns, int):
                    continue

                gt_img_key = str(meta.get("gt_img", "") or "").strip()
                if gt_img_key not in {"img1", "img2"}:
                    # 0-turn samples are stored as img1 GT in intermediate format.
                    gt_img_key = "img1"
                final_path = sample_dir / f"{gt_img_key}.png"
                if not final_path.is_file():
                    continue

                prev_path: Path | None = None
                if use_prev_image:
                    # For 2-turn prefer img1 as previous step; otherwise img0.
                    cand = sample_dir / ("img1.png" if num_turns >= 2 else "img0.png")
                    if cand.is_file():
                        prev_path = cand

                reflections: list[str] = []
                refl1 = _clean_reflection_text(meta.get("reflection1"))
                refl2 = _clean_reflection_text(meta.get("reflection2"))
                if refl1:
                    reflections.append(refl1)
                if refl2:
                    reflections.append(refl2)
                reflection_text = "\n".join(reflections).strip()

                condition_text = _build_condition_text(
                    prompt=prompt,
                    reflection=reflection_text,
                    use_prompt=use_prompt,
                    use_reflection=use_reflection,
                    use_vlat_tokens=use_vlat_tokens,
                    latent_token_repeat=latent_token_repeat,
                )

                final_str = str(final_path)
                if dataset_root and not os.path.isabs(final_str):
                    final_str = os.path.join(dataset_root, final_str)
                prev_str = str(prev_path) if prev_path is not None else None
                if prev_str and dataset_root and not os.path.isabs(prev_str):
                    prev_str = os.path.join(dataset_root, prev_str)

                self.samples.append(
                    Stage2Sample(
                        prompt=prompt,
                        reflection=reflection_text,
                        condition_text=condition_text,
                        final_image_path=final_str,
                        prev_image_path=prev_str,
                        metadata={
                            "dataset_name": source,
                            "sample_id": f"{source}_{source_id}",
                            "num_turns": num_turns,
                            "gt_img": gt_img_key,
                        },
                    )
                )

        if shuffle:
            import random

            rng = random.Random(seed)
            rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        gt_img = Image.open(sample.final_image_path).convert("RGB")
        prev_img = Image.open(sample.prev_image_path).convert("RGB") if sample.prev_image_path else None
        return {
            "condition_text": sample.condition_text,
            "prompt": sample.prompt,
            "reflection": sample.reflection,
            "gt_image": gt_img,
            "prev_image": prev_img,
            "metadata": sample.metadata,
        }

