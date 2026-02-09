"""Dataset preprocessing utilities."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path


def preprocess_sample(
    sample: dict,
    dataset_root: str = "",
    allow_no_observation: bool = False,
) -> dict | None:
    """Validate and normalize one training sample.

    Expected input format:
    {
      "data": [...chat turns...],
      "metadata": {"sample_id": ..., "dataset_name": ...}
    }
    """
    if "data" not in sample:
        return None

    data = deepcopy(sample["data"])
    metadata = deepcopy(sample.get("metadata", {}))

    n_img = 0
    n_img_pad = 0
    seen_observation = False

    for i, turn in enumerate(data):
        role = turn.get("role")
        content = turn.get("content", [])
        seen_assistant_img = False if role == "assistant" else None

        for j, item in enumerate(content):
            if item.get("type") == "image":
                img_value = item.get("image")
                if img_value is None:
                    return None
                if isinstance(img_value, (str, Path)):
                    if dataset_root and not os.path.isabs(str(img_value)):
                        img_value = os.path.join(dataset_root, str(img_value))
                item["image"] = img_value

                if role == "assistant":
                    n_img += 1
                    seen_assistant_img = True
                    if j > 0 and content[j - 1].get("type") == "text":
                        text_before = content[j - 1].get("text", "")
                        if "<abs_vis_token></abs_vis_token>" not in text_before:
                            return None

            elif item.get("type") == "text" and role == "assistant":
                text = item.get("text", "")
                n_img_pad += text.count("<abs_vis_token></abs_vis_token>")

                if "<observation>" in text:
                    seen_observation = True
                if "<observation>" in text and not seen_assistant_img:
                    text = text.replace("<observation>", "").replace("</observation>", "")
                    item["text"] = text

        data[i]["content"] = content

    if n_img != n_img_pad:
        return None
    if not seen_observation and not allow_no_observation:
        return None

    if "sample_id" not in metadata:
        metadata["sample_id"] = str(sample.get("sample_id", "unknown"))
    if "dataset_name" not in metadata:
        metadata["dataset_name"] = str(sample.get("dataset_name", "default"))

    return {"data": data, "metadata": metadata}
