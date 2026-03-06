"""Training/eval datasets."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from .preprocess import preprocess_sample
from qwen_latent_cot.utils import load_json


class LatentCoTDataset(Dataset):
    """Simple list-backed dataset for latent CoT training."""

    def __init__(
        self,
        data_paths: list[str],
        dataset_root: str = "",
        allow_no_observation: bool = False,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self._dataset_root = dataset_root
        self._allow_no_observation = allow_no_observation

        rows: list[dict] = []
        for path in data_paths:
            path_obj = Path(path)
            if not path_obj.is_dir():
                raise ValueError(
                    f"Only cort_kturn intermediate directories are supported: {path}"
                )
            rows.extend(_load_cort_kturn_intermediate(path_obj))

        processed: list[dict] = []
        for sample in rows:
            item = preprocess_sample(
                sample,
                dataset_root=dataset_root,
                allow_no_observation=allow_no_observation,
            )
            if item is not None:
                processed.append(item)

        if shuffle:
            import random

            rng = random.Random(seed)
            rng.shuffle(processed)

        self.samples = processed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def _clean_reflection_text(text: str) -> str:
    """Remove wrapper markers and normalize whitespace."""
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
    # Keep newlines reasonably stable but remove extreme spacing.
    s = s.strip()
    return "\n".join(line.strip() for line in s.splitlines()).strip()


def _read_turn_value(meta: dict) -> int | None:
    turn = meta.get("num_turns")
    if isinstance(turn, int) and turn in {0, 1, 2}:
        return turn
    return None


def _resolve_turn_images(sample_dir: Path, num_turns: int) -> list[str]:
    # Intermediate layout mapping:
    # 0-turn -> use img1 as GT; 1-turn -> img0,img1; 2-turn -> img0,img1,img2
    names_by_turn = {
        0: ["img1"],
        1: ["img0", "img1"],
        2: ["img0", "img1", "img2"],
    }
    names = names_by_turn[num_turns]
    paths: list[str] = []
    for name in names:
        p = sample_dir / f"{name}.png"
        if not p.is_file():
            return []
        paths.append(str(p))
    return paths


def _extract_reflections(meta: dict, num_turns: int) -> list[str]:
    if num_turns == 0:
        return []
    refl1 = _clean_reflection_text(meta.get("reflection1"))
    if not refl1:
        return []
    if num_turns == 1:
        return [refl1]
    refl2 = _clean_reflection_text(meta.get("reflection2"))
    if not refl2:
        return []
    return [refl1, refl2]


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


def _load_cort_kturn_intermediate(intermediate_path: Path) -> list[dict]:
    out: list[dict] = []
    system = {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    }
    sample_dirs = _iter_sample_dirs(intermediate_path)
    if not sample_dirs:
        raise ValueError(
            f"Invalid cort_kturn intermediate directory (no sample meta.json found): {intermediate_path}"
        )

    for sample_dir in sample_dirs:
        meta_path = sample_dir / "meta.json"
        meta = load_json(str(meta_path))
        if not isinstance(meta, dict):
            continue

        num_turns = _read_turn_value(meta)
        if num_turns is None:
            continue
        prompt = str(meta.get("prompt", "") or "").strip()
        source = str(meta.get("source", "") or "").strip()
        source_id = str(meta.get("source_id", "") or "").strip()
        if not prompt or not source or not source_id:
            continue

        image_paths = _resolve_turn_images(sample_dir, num_turns)
        if not image_paths:
            continue

        reflections = _extract_reflections(meta, num_turns)
        if num_turns > 0 and len(reflections) != num_turns:
            continue

        assistant_content: list[dict] = [{"type": "text", "text": "<|cort_start|>"}]
        for i, image_path in enumerate(image_paths):
            assistant_content.append({"type": "text", "text": "<|vlat_start|><|vlat_end|>"})
            assistant_content.append({"type": "image", "image": image_path})
            if i < len(reflections):
                assistant_content.append(
                    {"type": "text", "text": f"<|refl_start|>{reflections[i]}<|refl_end|>"}
                )
        assistant_content.append({"type": "text", "text": "<|cort_end|>"})

        out.append(
            {
                "data": [
                    system,
                    {"role": "user", "content": [{"type": "text", "text": f"Goal: {prompt}"}]},
                    {"role": "assistant", "content": assistant_content},
                ],
                "metadata": {
                    "dataset_name": source,
                    "sample_id": f"{source}_{source_id}",
                    "num_turns": num_turns,
                },
            }
        )
    return out
