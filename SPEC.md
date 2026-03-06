# CoRT K-Turn Parquet Data Specification

This document describes the parquet format, field definitions, special token conventions, and how to read and assemble training sequences for CoRT.

## Special Tokens

Six special tokens are stored in each parquet shard's **schema metadata**, keyed with the `cort.token.` prefix.

| Key | Token | Purpose | Has Loss |
|-----|-------|---------|:--------:|
| `cort.token.cort_start` | `<\|cort_start\|>` | CoRT chain start | No |
| `cort.token.cort_end` | `<\|cort_end\|>` | CoRT chain end (termination signal) | Yes |
| `cort.token.vlat_start` | `<\|vlat_start\|>` | Visual latent block start | No |
| `cort.token.vlat_end` | `<\|vlat_end\|>` | Visual latent block end | No |
| `cort.token.refl_start` | `<\|refl_start\|>` | Reflection block start | No |
| `cort.token.refl_end` | `<\|refl_end\|>` | Reflection block end | No |

### Text Tags (plain text, CE loss)

- `<problem>...</problem>` — Problem description
- `<fix>...</fix>` — Fix instruction

### Reading Special Tokens

```python
import pyarrow.parquet as pq

schema = pq.read_schema("kturn_0000.parquet")
special_tokens = {
    k.decode().removeprefix("cort.token."): v.decode()
    for k, v in schema.metadata.items()
    if k.decode().startswith("cort.token.")
}
# {'cort_start': '<|cort_start|>', 'cort_end': '<|cort_end|>', ...}
```

All shards share the same metadata; reading any single shard is sufficient.

## Parquet Schema

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string | Unique ID, format `{source}_{source_id}` |
| `prompt` | string | Generation instruction text |
| `num_turns` | int32 | Number of reflection turns (0/1/2) |
| `img0` | binary | PNG bytes — first image (GT for 0-turn) |
| `img1` | binary \| null | PNG bytes — second image (1/2-turn), null for 0-turn |
| `img2` | binary \| null | PNG bytes — third image (2-turn only), null otherwise |
| `reflection1` | string \| null | First reflection text (with `<problem>`/`<fix>` tags) |
| `reflection2` | string \| null | Second reflection text (2-turn only) |
| `source` | string | Data source: `echo4o` / `opengpt4o` / `imgedit` |
| `gt_img` | string | Ground truth image field name: `"img1"` or `"img2"` |

### Image Fields and Ground Truth

| num_turns | img0 | img1 | img2 | gt_img | Notes |
|:---------:|------|------|------|:------:|-------|
| 0 | GT image | null | null | `"img1"` | img0 stores the original img1 (GT) from intermediate |
| 1 | Generated | GT image | null | `"img1"` | img0=BAGEL generated, img1=GT |
| 2 | Original | First edit | GT image | `"img2"` | Three-stage edit chain |

> **Note**: For 0-turn, `gt_img="img1"` refers to img1 in the intermediate format. During packing, it is written into the parquet `img0` field.

## Data Sources

| Source | num_turns | Samples | Description |
|--------|:---------:|--------:|-------------|
| `echo4o` | 0 | 6,545 | Echo-4o-Image, simple instructions, no reflection needed |
| `opengpt4o` | 1 | 11,054 | OpenGPT-4o-Image + VLM reflection |
| `imgedit` | 2 | 9,849 | ImgEdit cot_triplet, two-round editing |

Total: **27,448** samples across 27 shards (~2GB/shard, zstd compressed).

## Sample Data

### 0-turn (echo4o)

```json
{
  "sample_id": "echo4o_3540",
  "prompt": "a photo of earrings and a drum",
  "num_turns": 0,
  "img0": "<PNG bytes — GT image>",
  "img1": null,
  "img2": null,
  "reflection1": null,
  "reflection2": null,
  "source": "echo4o",
  "gt_img": "img1"
}
```

### 1-turn (opengpt4o)

```json
{
  "sample_id": "opengpt4o_5212",
  "prompt": "Ancient temple courtyard, frontal isometric view...",
  "num_turns": 1,
  "img0": "<PNG bytes — BAGEL generated>",
  "img1": "<PNG bytes — GT image>",
  "img2": null,
  "reflection1": "<problem>The image contains two paper boats instead of the single boat specified in the prompt, violating the count requirement.</problem>\n<fix>Remove the distant paper boat to ensure only one boat drifts along the stream as described.</fix>",
  "reflection2": null,
  "source": "opengpt4o",
  "gt_img": "img1"
}
```

### 2-turn (imgedit)

```json
{
  "sample_id": "imgedit_00014_00001_000011599",
  "prompt": "A black leather steering wheel with vibrant green stitching...",
  "num_turns": 2,
  "img0": "<PNG bytes — original>",
  "img1": "<PNG bytes — first edit>",
  "img2": "<PNG bytes — GT final>",
  "reflection1": "<problem>Replace the current steering wheel with a black leather one...</problem>\n<fix>add a steering wheel in the left-central area...</fix>",
  "reflection2": "<problem>Img1 is wrong because the steering wheel's center emblem is the standard Bentley logo...</problem>\n<fix>Replace the steering wheel's center emblem with a green-accented Bentley logo...</fix>",
  "source": "imgedit",
  "gt_img": "img2"
}
```

## Training Sequence Assembly

The parquet stores raw fields. The training-side parser is responsible for assembling token sequences.

### K-Turn General Format

```
[prompt] <|cort_start|> { <|vlat_start|> [visual_k] <|vlat_end|> <|refl_start|> <problem>Pk</problem> <fix>Fk</fix> <|refl_end|> } x K  <|vlat_start|> [visual_K] <|vlat_end|> <|cort_end|>
```

### Expanded by num_turns

**0-turn** (no reflection):
```
{prompt} <|cort_start|> <|vlat_start|> [img0] <|vlat_end|> <|cort_end|>
```

**1-turn** (one reflection):
```
{prompt} <|cort_start|> <|vlat_start|> [img0] <|vlat_end|> <|refl_start|> {reflection1} <|refl_end|> <|vlat_start|> [img1] <|vlat_end|> <|cort_end|>
```

**2-turn** (two reflections):
```
{prompt} <|cort_start|> <|vlat_start|> [img0] <|vlat_end|> <|refl_start|> {reflection1} <|refl_end|> <|vlat_start|> [img1] <|vlat_end|> <|refl_start|> {reflection2} <|refl_end|> <|vlat_start|> [img2] <|vlat_end|> <|cort_end|>
```

Where `[imgN]` represents latent tokens produced by passing PNG bytes through the visual encoder.

### Parser Pseudocode

```python
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO

# 1. Read special tokens
schema = pq.read_schema("kturn_0000.parquet")
tokens = {
    k.decode().removeprefix("cort.token."): v.decode()
    for k, v in schema.metadata.items()
    if k.decode().startswith("cort.token.")
}

# 2. Register with tokenizer
tokenizer.add_special_tokens({"additional_special_tokens": list(tokens.values())})

# 3. Read data
table = pq.read_table("kturn_0000.parquet")

# 4. Assemble sequence
for row in table.to_pydict():
    parts = [row["prompt"], tokens["cort_start"]]

    for i in range(row["num_turns"] + 1):
        img_bytes = row[f"img{i}"]
        img = Image.open(BytesIO(img_bytes))
        visual_tokens = encode_image(img)  # your visual encoder

        parts.extend([tokens["vlat_start"], visual_tokens, tokens["vlat_end"]])

        refl = row.get(f"reflection{i + 1}")
        if refl is not None:
            parts.extend([tokens["refl_start"], refl, tokens["refl_end"]])

    parts.append(tokens["cort_end"])
    # parts -> tokenize -> training sample
```

## File Layout

```
/data0/data/cort_kturn/parquets/
  kturn_0000.parquet
  kturn_0001.parquet
  ...
  kturn_0026.parquet
  manifest.json          # Stats: sample counts, source distribution, turn distribution
```

`manifest.json` example:

```json
{
  "total_samples": 27448,
  "num_shards": 27,
  "source_counts": {"opengpt4o": 11054, "imgedit": 9849, "echo4o": 6545},
  "turn_counts": {"0": 6545, "1": 11054, "2": 9849}
}
```
