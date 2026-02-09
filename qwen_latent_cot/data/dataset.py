"""Training/eval datasets."""

from __future__ import annotations

from bisect import bisect_right
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
        self._parquet = False
        self._parquet_files: list[str] = []
        self._file_offsets: list[int] = []
        self._file_infos: list[dict] = []
        self._total_rows = 0
        self._rounds_per_row = 2
        self._dataset_root = dataset_root
        self._allow_no_observation = allow_no_observation

        rows: list[dict] = []
        for path in data_paths:
            path_obj = Path(path)
            if path_obj.is_dir():
                parquet_files = sorted(str(p) for p in path_obj.glob("*.parquet"))
                if parquet_files:
                    self._parquet_files.extend(parquet_files)
                    continue
                raise ValueError(f"No parquet files found in directory: {path}")
            if path_obj.suffix == ".parquet":
                self._parquet_files.append(str(path_obj))
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

        if self._parquet_files:
            if shuffle:
                raise ValueError("Shuffle is not supported for parquet datasets.")
            try:
                import pyarrow.parquet as pq  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "pyarrow is required to read parquet datasets. "
                    "Install it with: pip install pyarrow"
                ) from exc

            total = 0
            offsets: list[int] = []
            infos: list[dict] = []
            for file_path in self._parquet_files:
                pf = pq.ParquetFile(file_path)
                row_group_sizes = [
                    pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)
                ]
                row_group_offsets: list[int] = []
                rg_total = 0
                for size in row_group_sizes:
                    row_group_offsets.append(rg_total)
                    rg_total += size
                infos.append(
                    {
                        "path": file_path,
                        "row_group_sizes": row_group_sizes,
                        "row_group_offsets": row_group_offsets,
                        "rows": rg_total,
                    }
                )
                total += rg_total
                offsets.append(total)

            self._parquet = True
            self._file_offsets = offsets
            self._file_infos = infos
            self._total_rows = total
            self.samples = []
            return

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
        if self._parquet:
            return self._total_rows * self._rounds_per_row
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        if not self._parquet:
            return self.samples[index]

        row_idx = index // self._rounds_per_row
        round_idx = index % self._rounds_per_row
        file_pos = bisect_right(self._file_offsets, row_idx)
        file_info = self._file_infos[file_pos]
        prev_total = 0 if file_pos == 0 else self._file_offsets[file_pos - 1]
        file_row_idx = row_idx - prev_total

        rg_offsets = file_info["row_group_offsets"]
        rg_idx = bisect_right(rg_offsets, file_row_idx) - 1
        rg_idx = max(0, rg_idx)
        rg_row_idx = file_row_idx - rg_offsets[rg_idx]

        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(file_info["path"])
        table = pf.read_row_group(rg_idx)
        row = table.slice(rg_row_idx, 1).to_pydict()
        row = {k: v[0] for k, v in row.items()}

        for _ in range(50):
            sample = _cot_triplet_to_sample(row, round_idx)
            if sample is not None:
                sample = preprocess_sample(
                    sample,
                    dataset_root=self._dataset_root,
                    allow_no_observation=self._allow_no_observation,
                )
            if sample is not None:
                return sample
            # Try next sample if current row is invalid.
            row_idx = (row_idx + 1) % max(self._total_rows, 1)
            file_pos = bisect_right(self._file_offsets, row_idx)
            file_info = self._file_infos[file_pos]
            prev_total = 0 if file_pos == 0 else self._file_offsets[file_pos - 1]
            file_row_idx = row_idx - prev_total
            rg_offsets = file_info["row_group_offsets"]
            rg_idx = bisect_right(rg_offsets, file_row_idx) - 1
            rg_idx = max(0, rg_idx)
            rg_row_idx = file_row_idx - rg_offsets[rg_idx]
            pf = pq.ParquetFile(file_info["path"])
            table = pf.read_row_group(rg_idx)
            row = table.slice(rg_row_idx, 1).to_pydict()
            row = {k: v[0] for k, v in row.items()}
        raise ValueError("Invalid sample generated from parquet row.")


def _cot_triplet_to_sample(row: dict, round_idx: int) -> dict:
    prompt = str(row.get("prompt", "") or "")
    sample_id = row.get("sample_id", row.get("sample_idx", "unknown"))
    sample_id = str(sample_id)
    img0 = row.get("img0")
    img1 = row.get("img1")
    img2 = row.get("img2")
    reflection1 = row.get("reflection1", "")
    reflection2 = row.get("reflection2", "")

    system = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]}

    if round_idx == 0:
        base_img = img0 or img1 or img2
        if base_img is None:
            return None
        user_text = f"Goal: {prompt}"
        user_content = [
            {"type": "image", "image": base_img},
            {"type": "text", "text": user_text},
        ]
        assistant_text = f"<observation>{reflection1 or ''}</observation>"
        metadata = {"dataset_name": "cot_triplet", "sample_id": f"{sample_id}:r1"}
    else:
        base_img = img1 or img0 or img2
        if base_img is None:
            return None
        user_text = f"Goal: {prompt}\nPrevious reflection: {reflection1 or ''}"
        user_content = [
            {"type": "image", "image": base_img},
            {"type": "text", "text": user_text},
        ]
        assistant_text = f"<observation>{reflection2 or ''}</observation>"
        metadata = {"dataset_name": "cot_triplet", "sample_id": f"{sample_id}:r2"}

    assistant = {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
    return {"data": [system, {"role": "user", "content": user_content}, assistant], "metadata": metadata}
