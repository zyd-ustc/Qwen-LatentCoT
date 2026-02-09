# Qwen-LatentCoT

基于 `docs/Qwen_LatentCoT.md` 与 `refs/Monet`/`refs/CoRT` 思路实现的完整代码骨架，覆盖：

1. Qwen-image 推理闭环：`生图 -> 反思 -> 再生图`
2. 训练 Stage1-1 / Stage1-2 / Stage1-3
3. Teacher latent / teacher reps 预计算

当前仓库默认支持 **无权重可运行**（mock 推理 + 工具链验证）。当你提供真实权重后可直接切换到真实后端。

## 项目结构

- `qwen_latent_cot/cli.py`：统一命令入口
- `qwen_latent_cot/inference/pipeline.py`：两轮推理 pipeline
- `qwen_latent_cot/data/collators.py`：stage1-1/1-2/1-3 数据拼接与 mask
- `qwen_latent_cot/models/latent_student.py`：latent-mode 包装与 alignment/CE 计算
- `qwen_latent_cot/training/runner.py`：训练主流程
- `qwen_latent_cot/training/precompute.py`：teacher latent/rep 预计算
- `configs/`：示例配置
- `scripts/`：一键脚本

## 安装

```bash
cd Qwen-LatentCoT
pip install -e .
```

## 推理（生图 -> 反思 -> 再生图）

### 1) 无权重 mock 验证

```bash
python -m qwen_latent_cot.cli infer \
  --prompt "A red car parked on a beach at sunset" \
  --output-dir ./outputs/demo \
  --backend mock \
  --reflector heuristic
```

输出：
- `draft.png`
- `refined.png`
- `result.json`（包含 reflection 与 refine_prompt）

### 2) 使用本地权重推理

将 Qwen-Image 模型权重放在 `qwen_latent_cot/models/Qwen-Image/` 下（或任意路径），然后：

```bash
python -m qwen_latent_cot.cli infer \
  --prompt "A red car parked on a beach at sunset" \
  --output-dir ./outputs/demo_local \
  --backend local \
  --qwen-image-model qwen_latent_cot/models/Qwen-Image \
  --reflector heuristic \
  --num-inference-steps 50 \
  --guidance-scale 4.0 \
  --aspect-ratio 1:1
```

也可以用快捷脚本（默认读取 `qwen_latent_cot/models/Qwen-Image`）：

```bash
BACKEND=local QWEN_IMAGE_MODEL=qwen_latent_cot/models/Qwen-Image \
  bash scripts/infer_two_turn.sh "A red car parked on a beach at sunset"

如果你已经把权重放到默认目录，可以省略 `QWEN_IMAGE_MODEL`：

```bash
BACKEND=local bash scripts/infer_two_turn.sh "A red car parked on a beach at sunset"
```
```

支持的画面比例：`1:1`、`16:9`、`9:16`、`4:3`、`3:4`、`3:2`、`2:3`。

> **依赖提示**：本地推理需安装最新版 diffusers：`pip install git+https://github.com/huggingface/diffusers`

- 反思器：`--reflector qwen_vl --reflector-model <qwen2.5vl_path>`（加载 Qwen2.5-VL 进行反思）

## 训练 Stage1-1 / 1-2 / 1-3

### Stage1-1

```bash
python -m qwen_latent_cot.cli train \
  --stage stage1-1 \
  --model-path /path/to/qwen2.5-vl-7b \
  --data-path /path/to/train.jsonl \
  --output-dir ./checkpoints/stage1_1
```

### 预计算 teacher reps（Stage1-2 使用）

```bash
python -m qwen_latent_cot.cli precompute-rep \
  --model-path ./checkpoints/stage1_1 \
  --data-path /path/to/train.jsonl \
  --output-dir ./artifacts/teacher_reps \
  --output-hidden-states
```

### Stage1-2

```bash
python -m qwen_latent_cot.cli train \
  --stage stage1-2 \
  --model-path ./checkpoints/stage1_1 \
  --data-path /path/to/train.jsonl \
  --teacher-reps-dir ./artifacts/teacher_reps \
  --output-dir ./checkpoints/stage1_2
```

### 预计算 teacher latents（Stage1-3 使用）

```bash
python -m qwen_latent_cot.cli precompute-latent \
  --model-path ./checkpoints/stage1_2 \
  --data-path /path/to/train.jsonl \
  --output-dir ./artifacts/teacher_latents \
  --output-latent-embeds
```

### Stage1-3

```bash
python -m qwen_latent_cot.cli train \
  --stage stage1-3 \
  --model-path ./checkpoints/stage1_2 \
  --data-path /path/to/train.jsonl \
  --teacher-latent-dir ./artifacts/teacher_latents \
  --output-dir ./checkpoints/stage1_3
```

## 数据格式

示例见 `examples/sample_train.jsonl`。每条记录必须包含：

- `data`: Qwen chat 格式（system/user/assistant）
- `metadata`: `{dataset_name, sample_id}`
- assistant 图像前的文本里要有 `<abs_vis_token></abs_vis_token>`
- 反思文本用 `<observation>...</observation>` 包裹

## 快捷脚本

- `scripts/infer_two_turn.sh`
- `scripts/train_stage1_1.sh`
- `scripts/precompute_teacher_reps.sh`
- `scripts/train_stage1_2.sh`
- `scripts/precompute_teacher_latents.sh`
- `scripts/train_stage1_3.sh`

## 注意事项

- 当前实现支持无权重开发与接口联调；真实训练/推理需提供：
  - Qwen-image 生成权重（或可兼容服务）
  - Qwen2.5-VL-7B 权重
- Stage1-2/1-3 对齐损失依赖离线 teacher 文件命名约定：
  - `rep_{alignment_layer}_{dataset_name}_{sample_id}.pt`
  - `latent_{alignment_layer|last_layer}_{dataset_name}_{sample_id}.pt`
