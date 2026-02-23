"""Training/eval datasets."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from .preprocess import preprocess_sample
from qwen_latent_cot.utils import load_json, load_jsonl


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
            if path_obj.is_dir():
                dataset_name = "1turn_reflect"
                if "cot_triplet" in str(path_obj).lower():
                    dataset_name = "cot_triplet"
                intermediate_rows = _load_reflect_intermediate(path_obj, dataset_name=dataset_name)
                if not intermediate_rows:
                    raise ValueError(
                        f"Unsupported data directory (expect intermediate layout): {path}"
                    )
                rows.extend(intermediate_rows)
                continue
            if path_obj.suffix == ".jsonl":
                loaded = load_jsonl(path)
            elif path_obj.suffix == ".json":
                loaded = load_json(path)
            else:
                raise ValueError(f"Unsupported data file: {path}")
            if isinstance(loaded, dict):
                loaded = [loaded]
            rows.extend(loaded)

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


def _format_problem_fix(problem: str, fix: str) -> str:
    problem = str(problem or "").strip()
    fix = str(fix or "").strip()
    if problem and fix:
        return f"Problem: {problem}\nFix: {fix}"
    return problem or fix


def _resolve_intermediate_image_path(dir_path: Path, sample_id: str) -> str | None:
    candidates = [
        dir_path / "img_gen",
        dir_path / "images",
        dir_path.parent / "img_gen",
        dir_path.parent / "images",
    ]
    exts = [".png", ".jpg", ".jpeg", ".webp"]

    for base in candidates:
        if not base.is_dir():
            continue
        for ext in exts:
            p = base / f"{sample_id}{ext}"
            if p.is_file():
                return str(p)
    return None


def _load_reflect_intermediate(dir_path: Path, dataset_name: str) -> list[dict]:
    """Load reflect dataset from intermediate directory layout."""
    gen_path = dir_path / "gen.json"
    refl_dir = dir_path / "reflections"
    if not gen_path.is_file() or not refl_dir.is_dir():
        return []

    loaded = load_json(str(gen_path))
    if isinstance(loaded, dict):
        loaded = [loaded]
    if not isinstance(loaded, list):
        return []

    system = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]}
    out: list[dict] = []
    for row in loaded:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id", "") or "").strip()
        prompt = str(row.get("input_prompt", "") or "").strip()
        if not sample_id or not prompt:
            continue

        refl_path = refl_dir / f"{sample_id}.json"
        if not refl_path.is_file():
            continue
        refl = load_json(str(refl_path))
        if not isinstance(refl, dict):
            continue

        result = refl.get("result", {})
        if not isinstance(result, dict):
            result = {}
        obs_text = _format_problem_fix(result.get("problem", ""), result.get("fix", ""))
        obs_text = _clean_reflection_text(obs_text)
        if not obs_text:
            continue

        img_path = _resolve_intermediate_image_path(dir_path, sample_id)
        if img_path is None:
            continue

        user = {"role": "user", "content": [{"type": "text", "text": f"Goal: {prompt}"}]}
        assistant = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<abs_vis_token></abs_vis_token>"},
                {"type": "image", "image": img_path},
                {"type": "text", "text": f"<observation>{obs_text}</observation>"},
            ],
        }
        out.append(
            {
                "data": [system, user, assistant],
                "metadata": {"dataset_name": dataset_name, "sample_id": sample_id},
            }
        )

    return out
