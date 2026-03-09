#!/usr/bin/env python3
"""Validate outputs produced by `qwen_latent_cot.cli infer-cort`."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qwen_latent_cot.utils import load_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--latent-size", type=int, default=8)
    p.add_argument("--min-turns", type=int, default=1)
    p.add_argument("--expected-turns", type=int, default=0)
    p.add_argument("--allow-missing-cort-end", action="store_true")
    return p.parse_args()


def _count_token(text: str, token: str) -> int:
    return text.count(token)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    meta_path = out_dir / "meta.json"
    cort_path = out_dir / "cort.txt"
    latents_path = out_dir / "latents.pt"

    missing = [str(p) for p in (meta_path, cort_path, latents_path) if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing infer-cort outputs: {missing}")

    meta = load_json(str(meta_path))
    if not isinstance(meta, dict):
        raise SystemExit(f"Invalid meta.json: {meta_path}")

    cort_text = cort_path.read_text(encoding="utf-8").strip()
    payload = torch.load(latents_path, map_location="cpu")

    latent_blocks = list(payload.get("latent_blocks", []))
    reflections = list(payload.get("reflections", []))
    turn_count = int(payload.get("turn_count", len(latent_blocks)))

    errors: list[str] = []

    if "<|cort_start|>" not in cort_text:
        errors.append("missing <|cort_start|>")
    if (not args.allow_missing_cort_end) and "<|cort_end|>" not in cort_text:
        errors.append("missing <|cort_end|>")

    vlat_start_count = _count_token(cort_text, "<|vlat_start|>")
    vlat_end_count = _count_token(cort_text, "<|vlat_end|>")
    refl_start_count = _count_token(cort_text, "<|refl_start|>")
    refl_end_count = _count_token(cort_text, "<|refl_end|>")

    if vlat_start_count != vlat_end_count:
        errors.append(f"latent block markers mismatch: start={vlat_start_count}, end={vlat_end_count}")
    if refl_start_count != refl_end_count:
        errors.append(f"reflection markers mismatch: start={refl_start_count}, end={refl_end_count}")
    if turn_count != len(latent_blocks):
        errors.append(f"turn_count mismatch: meta={turn_count}, latent_blocks={len(latent_blocks)}")
    if turn_count != len(reflections):
        errors.append(f"turn_count mismatch: meta={turn_count}, reflections={len(reflections)}")
    if turn_count < int(args.min_turns):
        errors.append(f"turn_count={turn_count} < min_turns={args.min_turns}")
    if args.expected_turns > 0 and turn_count != int(args.expected_turns):
        errors.append(f"turn_count={turn_count} != expected_turns={args.expected_turns}")

    for idx, block in enumerate(latent_blocks):
        if not isinstance(block, torch.Tensor):
            errors.append(f"latent block {idx} is not a tensor")
            continue
        if block.dim() != 2:
            errors.append(f"latent block {idx} rank is {block.dim()}, expected 2")
            continue
        if block.size(0) != int(args.latent_size):
            errors.append(
                f"latent block {idx} length is {block.size(0)}, expected latent_size={args.latent_size}"
            )
        if block.size(1) <= 0:
            errors.append(f"latent block {idx} hidden dim is invalid: {block.size(1)}")

    if errors:
        raise SystemExit("infer-cort validation failed:\n- " + "\n- ".join(errors))

    print("infer-cort validation passed")
    print(f"output_dir: {out_dir}")
    print(f"turn_count: {turn_count}")
    print(f"latent_blocks: {len(latent_blocks)}")
    print(f"reflections: {len(reflections)}")


if __name__ == "__main__":
    main()
