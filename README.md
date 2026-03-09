# Qwen-LatentCoT

Qwen-LatentCoT 是一个面向多轮图像编辑/反思数据的训练与推理项目，核心目标是让模型学会：

1. 先生成图像（draft）
2. 读图反思（reflection）
3. 再生成更符合目标的图像（refine）

训练部分按 `Stage1-1 -> Stage1-2 -> Stage1-3 -> Stage1-4` 分阶段推进，并通过两类离线教师信号进行蒸馏：

- `teacher reps`（中间隐藏状态）
- `teacher latents`（latent token 对应表征）

---

## 1. 项目解决什么问题

传统图像生成/编辑往往只对最终像素监督，缺少“中间思考过程”对齐。Qwen-LatentCoT 把中间过程拆成可监督的结构：

- 用 `<|vlat_*|>` 表达视觉 latent 段
- 用 `<|refl_*|>` 与 `<problem>/<fix>` 表达反思文本段
- 用分阶段训练逐步引入更强的 teacher 监督

这样做的好处是：

- 训练更稳定（先学基础，再学对齐，再学 latent 对齐）
- 各阶段目标明确，便于排查问题
- 可以独立复用 precompute 产物

---

## 2. 仓库结构

- `qwen_latent_cot/cli.py`：统一命令入口（`infer` / `train` / `precompute-*`）
- `qwen_latent_cot/inference/pipeline.py`：`draft -> reflection -> refine` 推理闭环
- `qwen_latent_cot/data/collators.py`：三阶段数据组装、label 构造、4D attention mask
- `qwen_latent_cot/models/latent_student.py`：latent mode 包装、CE loss、alignment loss
- `qwen_latent_cot/training/runner.py`：训练主流程（构建模型、数据、Trainer）
- `qwen_latent_cot/training/precompute.py`：教师信号离线预计算
- `configs/`：DeepSpeed 等配置
- `scripts/`：常用训练/推理脚本
- `SPEC.md`：CoRT K-Turn parquet 数据规范

---

## 3. 安装

```bash
cd /home/ydzhi/Qwen-LatentCoT
pip install -e .
```

建议在独立环境中安装，并先确认 `torch/transformers/accelerate/deepspeed` 版本兼容。

---

## 4. 推理流程（Draft -> Reflection -> Refine）

### 4.1 原理

`ReflectionRegenerationPipeline`（`qwen_latent_cot/inference/pipeline.py`）做三件事：

1. 用 image backend 根据 `prompt` 生成 `draft`
2. 用 reflector 对 `draft` 做反思，产出 `reflection`
3. 拼接 `refine_prompt`，再以 `draft` 为参考生成 `refined`

输出包含：

- `draft.png`
- `refined.png`
- `result.json`（含 `prompt/reflection/refine_prompt`）

### 4.2 无权重 smoke test

```bash
python -m qwen_latent_cot.cli infer \
  --prompt "A red car parked on a beach at sunset" \
  --output-dir ./outputs/demo \
  --backend mock \
  --reflector heuristic
```

### 4.3 本地权重推理

```bash
python -m qwen_latent_cot.cli infer \
  --prompt "A red car parked on a beach at sunset" \
  --output-dir ./outputs/demo_local \
  --backend local \
  --qwen-image-model /path/to/Qwen-Image \
  --reflector heuristic \
  --num-inference-steps 50 \
  --guidance-scale 4.0 \
  --aspect-ratio 1:1
```

可选反思器：

- `--reflector heuristic`
- `--reflector qwen_vl --reflector-model /path/to/qwen2.5-vl`

### 4.4 随机单样本 checkpoint 对比（未训练 / 1-1 / 1-2 / 1-3 / 2）

```bash
python -m qwen_latent_cot.cli infer-compare-stages \
  --data-path /data0/data/cort_kturn/intermediate \
  --output-dir ./outputs/compare_one_sample \
  --stages stage1-2 stage2 \
  --qwen-image-base-model /data1/weights/Qwen-Image \
  --stage2-checkpoint /data2/checkpoints/stage2_e2e \
  --reflector-base-model /data1/weights/Qwen-Image-Edit \
  --stage1-1-checkpoint /data2/checkpoints/stage1_1 \
  --stage1-2-checkpoint /data2/checkpoints/stage1_2 \
  --stage1-3-checkpoint /data2/checkpoints/stage1_3
```

说明：

- 只会随机抽取 1 条样本（由 `--sample-seed` 控制）
- 不传 `--sample-seed` 时，每次运行都会重新随机抽样
- `--stages` 可选，支持单个或多个（如 `stage1-2`、`stage2`）
- 每个 stage 会分别输出 `draft.png` / `refined.png` / `result.json`
- 汇总结果写入 `compare_summary.json`

### 4.5 Stage1-4 CoRT 自回归推理

`infer-cort` 用 stage1-4 checkpoint 直接生成一段结构化 CoRT：

- 先输出 `<|cort_start|>`
- 每轮输出一个固定长度 latent block：`<|vlat_start|><|vlat_pad|>*K<|vlat_end|>`
- 再输出一段 `<|refl_start|>...<|refl_end|>`
- 最后输出 `<|cort_end|>`

命令：

```bash
python -m qwen_latent_cot.cli infer-cort \
  --model-path ./checkpoints/stage1_4 \
  --base-model-path /path/to/Qwen2.5-VL-or-Qwen-Image-Edit \
  --prompt "Turn the scene into a rainy cyberpunk street at night." \
  --output-dir ./outputs/infer_cort \
  --latent-size 8 \
  --max-cort-turns 3 \
  --max-reflection-tokens 128
```

输出：

- `cort.txt`：完整 CoRT 文本
- `latents.pt`：每轮 latent block hidden states 与 reflection
- `meta.json`：摘要信息

快速校验：

```bash
bash scripts/infer_cort_validate.sh
```

---

## 5. 训练总览（推荐顺序）

完整训练链路：

1. `Stage1-1`：基础 CE 学习
2. `precompute-rep`：离线提取 teacher hidden states
3. `Stage1-2`：CE + representation alignment
4. `precompute-latent`：离线提取 teacher latent targets
5. `Stage1-3`：CE + latent alignment
6. `Stage1-4`：CoRT 结构生成 + latent alignment
7. `Stage2`：End-to-end final image generation supervision（Qwen-Image）

每一步都依赖前一步产物，因此建议严格按顺序执行。

---

## 6. 每个 Step 的原理

### 6.1 Stage1-1（基础阶段）

命令：

```bash
python -m qwen_latent_cot.cli train \
  --stage stage1-1 \
  --model-path /path/to/base_or_stage_ckpt \
  --qwen-image-edit-root /path/to/Qwen-Image-Edit \
  --data-path /path/to/train.jsonl \
  --output-dir ./checkpoints/stage1_1
```

做什么：

- 使用 `collate_stage1_1` 组装 teacher 视角输入
- 构造 `teacher_labels`
- 用 `Stage11Trainer` 仅计算 CE（可选 observation token 强调）

为什么：

- 先让模型具备稳定的文本-视觉条件建模能力
- 为后续对齐阶段提供可用初始化

---

### 6.2 precompute-rep（给 Stage1-2 的教师中间表示）

命令：

```bash
python -m qwen_latent_cot.cli precompute-rep \
  --model-path ./checkpoints/stage1_1 \
  --qwen-image-edit-root /path/to/Qwen-Image-Edit \
  --data-path /path/to/train.jsonl \
  --output-dir ./artifacts/teacher_reps \
  --output-hidden-states
```

做什么：

- 以前向推理方式遍历数据
- 从 `hidden_states` 中抽取对齐位置（`obs` 或 `latent_end`）的表示
- 按样本保存为 `rep_*.pt`

为什么：

- Stage1-2 训练时不再在线跑 teacher，降低显存与计算开销
- 离线文件支持断点恢复和复用

---

### 6.3 Stage1-2（核心：CE + Rep 对齐）

命令：

```bash
python -m qwen_latent_cot.cli train \
  --stage stage1-2 \
  --model-path ./checkpoints/stage1_1 \
  --qwen-image-edit-root /path/to/Qwen-Image-Edit \
  --data-path /path/to/train.jsonl \
  --teacher-reps-dir ./artifacts/teacher_reps \
  --output-dir ./checkpoints/stage1_2
```

做什么（两次 forward）：

1. `latent_mode=True`：拿到 latent patch 位置与向量（`ce_patch_pos/ce_patch_vec`）
2. `latent_mode=False`：把 patch 向量回填到输入 embedding，再算
   - `ce_loss`
   - `alignment_loss`（与 teacher reps 对齐）

损失组合（简化）：

- 默认：`loss = ce + alignment_weight * alignment`
- 可选 latent 强调：通过 `compute_latents_only_loss` 构造 proxy loss

为什么：

- 既保证最终答案 token 预测能力（CE）
- 又约束中间表示与 teacher 一致（alignment）

---

### 6.4 precompute-latent（给 Stage1-3 的教师 latent）

命令：

```bash
python -m qwen_latent_cot.cli precompute-latent \
  --model-path ./checkpoints/stage1_2 \
  --qwen-image-edit-root /path/to/Qwen-Image-Edit \
  --data-path /path/to/train.jsonl \
  --output-dir ./artifacts/teacher_latents \
  --output-latent-embeds
```

做什么：

- 运行 stage1-2 风格输入
- 导出 latent token 相关向量（`latent_*.pt`）

为什么：

- Stage1-3 直接对齐“latent 目标”而非普通 hidden states
- 让模型在多轮链路中学习更稳定的 latent 过渡

---

### 6.5 Stage1-3（CE + Latent 对齐）

命令：

```bash
python -m qwen_latent_cot.cli train \
  --stage stage1-3 \
  --model-path ./checkpoints/stage1_2 \
  --qwen-image-edit-root /path/to/Qwen-Image-Edit \
  --data-path /path/to/train.jsonl \
  --teacher-latent-dir ./artifacts/teacher_latents \
  --output-dir ./checkpoints/stage1_3
```

做什么：

- 加载离线 `teacher_latents`
- 在 student 输入上进行 CE + alignment 联合优化

为什么：

- 把 latent 通道的蒸馏目标继续强化到最终阶段
- 提升多轮编辑链中的稳定性与可控性

---

### 6.6 Stage1-4（可自回归 CoRT 生成）

命令：

```bash
python -m qwen_latent_cot.cli train \
  --stage stage1-4 \
  --model-path ./checkpoints/stage1_3 \
  --qwen-image-edit-root /path/to/Qwen-Image-Edit \
  --data-path /path/to/train.jsonl \
  --teacher-latent-dir ./artifacts/teacher_latents \
  --output-dir ./checkpoints/stage1_4 \
  --stage1-4-structure-ce-weight 1.0 \
  --alignment-weight 1.0
```

做什么：

- 删除 assistant 侧真实图像，只保留 question image 和 CoRT token 序列
- 对完整 CoRT 结构做 CE：
  - `<|cort_start|> / <|cort_end|>`
  - `<|vlat_start|><|vlat_pad|>*K<|vlat_end|>`
  - `<|refl_start|> ... <|refl_end|>`
- 同时对 `vlat_pad` 对应 hidden states 做 latent alignment

为什么：

- 让模型不再只是“在预留 latent 槽位里写向量”
- 而是学会自己输出 CoRT 结构，再在结构内部生成 reflection
- 为 `infer-cort` 提供可控的自回归 rollout 能力

当前实现边界：

- 这是最小可跑版本，先做 teacher-forced CoRT SFT + latent alignment
- 还没有加入 scheduled sampling / free-running rollout 训练
- 推理端使用 grammar-constrained decoder 保证 latent block 长度固定

---

### 6.7 Stage2（End-to-end Generation，监督最终图像）

命令：

```bash
python -m qwen_latent_cot.cli train-stage2 \
  --qwen-image-model-path /path/to/Qwen-Image \
  --data-path /data0/data/cort_kturn/intermediate \
  --output-dir ./checkpoints/stage2_e2e \
  --batch-size 1 \
  --epochs 1 \
  --image-size 512 \
  --diffusion-weight 1.0 \
  --recon-weight 0.1 \
  --deepspeed /home/ydzhi/Qwen-LatentCoT/configs/stage2/deepspeed_zero2_bf16.json
```

做什么：

- 训练对象切换为 Qwen-Image 生成器（`transformer` 或 `unet`）
- 样本由 `Stage2Dataset` 产出：`condition_text + final_gt_image (+ prev_image 可选)`
- 损失是混合形式：
  - `L_diff`：扩散主损失（噪声预测）
  - `L_rec`：重建辅助损失（L1）
  - 总损失：`L = w_diff * L_diff + w_rec * L_rec`（`lpips_weight` 预留参数）

条件（conditioning）如何构造：

- 默认包含：`prompt + reflection + vlat token 序列`
- 可通过参数关闭：`--no-use-reflection`、`--no-use-vlat-tokens`、`--no-use-prompt`
- `--latent-token-repeat` 控制 `<|vlat_start|><|vlat_end|>` 重复次数

为什么：

- Stage1 系列主要学习“中间过程对齐”（latent/reps）
- Stage2 直接监督“最终图像质量”，让生成器端端到端学会把 CoRT 条件映射到目标图像

---

## 7. 数据格式与关键 token

示例见 `examples/sample_train.jsonl`，每条样本至少包含：

- `data`：Qwen chat 格式消息
- `metadata`：`{dataset_name, sample_id}`

关键 token 约定：

- `<|vlat_start|> ... <|vlat_end|>`：视觉 latent 段
- `<|refl_start|> ... <|refl_end|>`：反思段
- `<problem>...</problem>` / `<fix>...</fix>`：反思结构化文本

更多 parquet 字段与拼接规则见 `SPEC.md`。

---

## 8. 常见配置与注意事项

- `--qwen-image-edit-root`：当 `--model-path` 是 stage checkpoint（不含完整 base config）时必填
- `--teacher-reps-dir`：Stage1-2 必填
- `--teacher-latent-dir`：Stage1-3 / Stage1-4 必填
- `--stage1-4-structure-ce-weight`：控制 CoRT 结构 CE 权重
- `--sft-stage2-align-poss`：`obs` 或 `latent_end`
- `--not-use-4d` / `--mask-latent` 等影响注意力可见域
- `train-stage2` 入口用于最终图像监督，主要参数：
  - `--qwen-image-model-path`
  - `--diffusion-weight` / `--recon-weight`
  - `--latent-token-repeat` 与 `--no-use-*` 条件开关
- Stage2 推荐 deepspeed 配置：`configs/stage2/deepspeed_zero2_bf16.json`
- 多卡训练必须走分布式（DeepSpeed 或 torchrun），不要落到 DataParallel

---

## 9. 常用脚本

- `scripts/infer_two_turn.sh`
- `scripts/train_stage1_1.sh`
- `scripts/precompute_teacher_reps.sh`
- `scripts/train_stage1_2.sh`
- `scripts/precompute_teacher_latents.sh`
- `scripts/train_stage1_3.sh`
- `scripts/train_stage1_4.sh`
- `scripts/train_stage2.sh`
- `scripts/infer_cort_validate.sh`
- `scripts/validate_infer_cort.py`
- `scripts/infer_compare_stages_one_sample.sh`

---

## 10. 一句话总结

这个项目不是“只训最终答案”，而是把图像编辑中的中间思维（latent + reflection）变成可监督对象，分阶段逐步蒸馏到 student 模型中。
