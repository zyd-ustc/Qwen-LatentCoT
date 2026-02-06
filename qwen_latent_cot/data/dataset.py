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
        rows: list[dict] = []
        for path in data_paths:
            path_obj = Path(path)
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
