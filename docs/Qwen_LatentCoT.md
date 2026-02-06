# Latent Visual CoT 完整实现方案

基于 **Qwen-Image** 架构（Qwen2.5-VL + MMDiT）和 **Monet** 的 Latent Reasoning 训练范式，本方案详细描述如何实现 Latent-multi-turn-CoT 的蒸馏训练。

该方案的核心在于将 Qwen-Image 的 LLM 部分（Qwen2.5-VL）从单纯的文本/条件编码器，通过**蒸馏（Distillation）**转化为一个能生成"隐式视觉思维（Visual Latent）"的推理模型，并用这些 Latent 驱动 MMDiT 生成图像。

---

## 核心定义：在 Qwen-Image 中什么是 "Visual Latent"？

在 Qwen-Image 架构中，图像生成通常由 Qwen2.5-VL 提取的语义特征（Semantic Features）和 VAE 提取的细节特征共同作为条件输入给 MMDiT。

*   **Visual Latent ($Z$)**: 我们定义 $Z$ 为 Qwen2.5-VL 在 `<latent>` 和 `</latent>` 之间生成的连续 Hidden States。
*   **作用**: $Z$ 承载了原本需要显式生成的 `Reflection` 文本和辅助图像 `img0` 的信息，作为 MMDiT 的**Conditioning Context**。

---

## 与 Monet `src/main.py` 对齐的实现细节

下面的细节直接对应 `refs/Monet/src/main.py`，用于把方案落到可运行代码上。

### 1) Token 与模型配置

```python
# main.py 里已固定添加
<abs_vis_token_pad>, <abs_vis_token>, </abs_vis_token>, <observation>, </observation>

# config 里注册 latent token id
model.config.latent_token_id = latent_pad_idx
model.config.latent_start_id = latent_start_idx
model.config.latent_end_id = latent_end_idx
model.config.answer_start_pattern = "<|im_start|>assistant"

# 视觉编码冻结（只训 LLM 路径）
for p in model.visual.parameters():
    p.requires_grad = False
```

**落地要求**：
* **所有数据都通过** `processor.apply_chat_template` 构造成 Qwen2.5-VL 的对话格式。
* **所有 latent 占位**必须在 `assistant` 段落内，且和图像 pad 互斥替换。

---

### 2) 三个训练阶段与 main.py 对应关系

在 Monet 的实现里，训练阶段直接由 `args.stage` 驱动：

* `sft_stage1` → Warm-up + Teacher 监督（可显式 Reflection）
* `sft_stage2` → Teacher Latent 条件蒸馏（看得到辅助图）
* `sft_stage3` → Student Latent 蒸馏（看不到辅助图，仅看问题图）

这三个阶段分别使用不同的 `collate_fn_*`，且 loss 与 attention mask 配置不同。

---

### 3) Stage 1 (sft_stage1): 显式思维热身 + Teacher 监督

核心逻辑对应 `collate_fn_sft_stage1`：

* **替换** `<abs_vis_token></abs_vis_token>` → `<|vision_start|><|image_pad|><|vision_end|>`  
  保证 teacher 看到**真实图像**而非 latent。
* **构造 labels**：`generate_labels_after_multi_token_start`  
  忽略 `<|endoftext|>`、所有视觉相关 token、`<observation>` 标记。
* **记录 observation 区间**（`teacher_observation_poss`），后续做 observation-only loss。

**数据格式建议（stage1）**：

```json
{
  "data": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
      {"type": "image", "image": "img0"},
      {"type": "text", "text": "Goal: ..."}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "<observation>...reflection...</observation>"}
    ]}
  ],
  "metadata": {"sample_id": "..."}
}
```

---

### 4) Stage 2 (sft_stage2): Teacher Latent 条件蒸馏（可见辅助图）

对应 `collate_fn_sft_stage2`，这是关键的 **Teacher latent 监督**：

1. 先把 `<abs_vis_token></abs_vis_token>` 替换成图像 pad，保证能读入所有图像。
2. 再在每个辅助图之后插入 `<abs_vis_token_pad>` * latent_size，用于 latent 对齐。
3. 构建 **4D attention mask**，控制 latent / observation 是否能看到 question image。

**关键开关（args）**：

* `--not_use_4d`：关闭 4D mask（默认不开）
* `--mask_question_image`：强制 observation 看不到问题图  
* `--observation_tokens_cannot_see_question_image`  
* `--observation_tokens_only_see_question_and_latent`  
* `--latent_can_see_all_previous`

**labels 构造**：

* `only_predict_obs` → 只训练 observation tokens  
* 否则忽略 latent pad / latent end / image tokens。

---

### 5) Stage 3 (sft_stage3): Student Latent 蒸馏（不见辅助图）

对应 `collate_fn_sft_stage3`，真正的 “盲生成”：

* 先把 `<|vision_start|><|image_pad|><|vision_end|>` **替换成**  
  `<abs_vis_token><abs_vis_token_pad>...</abs_vis_token>`（student 输入）
* 再删除 `assistant` 的辅助图，仅保留 **question images**
* `student_alignment_poss` = latent pad 位置，用于 alignment loss
* 可选 4D mask：`build_4d_attn_wo_helper_images`

**对齐信号**：latent pad positions 作为 `alignment_poss`，用于对齐 teacher latents。

---

### 6) Loss 设计（与训练器一致）

Stage 2/3 的 CustomTrainer 会用到这些 training_args：

* `alignment_layer`：对齐哪一层的 hidden states  
* `alignment_weight`：对齐损失权重  
* `ce_emphasize_factor`：CE 强调  
* `emphasize_latent_weight`：latent 对齐权重  
* `teacher_reps_dir` / `teacher_latent_dir`：预计算 teacher 输出

简化 loss 逻辑：

* **CE Loss**: 训练 observation / reflection  
* **Alignment Loss**: latent pad 对齐 teacher hidden / latent  
* **可选**: 使用 `only_predict_obs` 限定 tokens

---

### 7) Image Resize 策略（main.py 已实现）

* `--image_resize global`：全局 token budget resize  
* `--image_resize clear_question_img`：仅对 question image 做清晰化 resize  
* stage2 / stage3 各自有 `sft_stage*_img_tokens` 参数控制预算

---

## 完善后的训练方案 (Refined Pipeline)

### Stage 1-1: Warm-up (显式思维热身 / Teacher 监督)

**目标**: 激活 Qwen-Image 对交错格式（Interleaved）数据的处理能力，学会"先画草图/看图，再反思，再画精修图"的显式流程。

**数据构造**: `User Prompt` → `Aux Image (img0)` → `Reflection Text` → `Target Image (img1*)`

**具体实现** (对齐 `refs/Monet/src/main.py`):

```python
# 数据格式：COT Triplet
# Round 1: prompt + img0 -> reflection1
# Round 2: prompt + img0 + reflection1 + img1 -> reflection2

SYSTEMPROMPT1 = "Based on the image and the editing goal, analyze what needs to be changed and provide editing instructions."
SYSTEMPROMPT2 = "Based on the original image, your previous edit result, analyze if the edit achieved the goal and provide further instructions."

def collate_fn_sft_stage1(examples):
    texts = []
    images = []
    
    for ex in examples:
        round_type = ex["round"]
        
        if round_type == 1:
            # Round 1: [img0] + prompt + systemprompt1 -> reflection1
            text = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
                f"Goal: {ex['prompt']}\n{SYSTEMPROMPT1}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex['target']}<|im_end|>"
            )
            texts.append(text)
            images.append([load_image(ex["image"])])
        else:
            # Round 2: [img0] + prompt + reflection1 + [img1] -> reflection2
            text = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
                f"Goal: {ex['prompt']}\n"
                f"Previous reflection: {ex['reflection1']}\n"
                f"<|vision_start|><|image_pad|><|vision_end|>\n"
                f"{SYSTEMPROMPT2}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex['target']}<|im_end|>"
            )
            texts.append(text)
            images.append([load_image(ex["image0"]), load_image(ex["image1"])])
    
    return processor(text=texts, images=images, return_tensors="pt", padding=True)
```

**Loss**: 标准的 Next Token Prediction (NTP) 用于文本，Flow Matching Loss 用于图像。

---

### Stage 1-2: Construct Teacher Latents (构建蒸馏目标)

**目标**: 利用"上帝视角"（能看到 `img0`）生成高质量的 Latent $Z_{teacher}$，作为后续蒸馏的 Target。此阶段**不训练模型生成能力，只为了获取 Target**。

**核心实现** (对齐 `refs/Monet/src/main.py` 的 token + 4D mask 逻辑):

```python
# 1. 添加 Latent 相关的特殊 Token
processor.tokenizer.add_tokens("<abs_vis_token_pad>", special_tokens=True)  # Latent 占位符
processor.tokenizer.add_tokens("<abs_vis_token>", special_tokens=True)       # Latent 开始
processor.tokenizer.add_tokens("</abs_vis_token>", special_tokens=True)      # Latent 结束
processor.tokenizer.add_tokens("<observation>", special_tokens=True)         # 观察开始
processor.tokenizer.add_tokens("</observation>", special_tokens=True)        # 观察结束

# 2. 配置模型
model.config.latent_token_id = int(latent_pad_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)

# 3. 构建 4D Attention Mask (Monet 核心)
# 关键：Latent 生成时可以看到 img0，但后续 Reflection 生成时屏蔽 img0
def build_4d_attn(input_ids, pad_mask, token_ids, **kwargs):
    """
    构建 4D Attention Mask，实现：
    - Latent tokens 可以 Attention 到 Prompt 和 img0
    - Observation/Reflection tokens 只能 Attention 到 Prompt 和 Latent，屏蔽 img0
    """
    B, S = input_ids.shape
    attn_mask = torch.zeros(B, 1, S, S, dtype=torch.bool)
    
    for b in range(B):
        # 找到各区段位置
        img_positions = find_image_positions(input_ids[b], token_ids)
        latent_positions = find_latent_positions(input_ids[b], token_ids)
        obs_positions = find_observation_positions(input_ids[b], token_ids)
        
        # 基础因果 mask
        attn_mask[b, 0] = torch.tril(torch.ones(S, S, dtype=torch.bool))
        
        # Observation tokens 不能看到 question image
        if kwargs.get('observation_tokens_cannot_see_question_image'):
            for obs_pos in obs_positions:
                for img_pos in img_positions[:1]:  # 只屏蔽第一张图（问题图）
                    attn_mask[b, 0, obs_pos, img_pos] = False
    
    return attn_mask

# 4. 预计算 Teacher Latents
def collate_fn_precompute_teacher_latents(examples):
    # 将 <abs_vis_token></abs_vis_token> 替换为 <|vision_start|><|image_pad|><|vision_end|>
    texts = [replace_latent_placeholder_with_img_pad(text) for text in texts]
    
    # 在每个辅助图像后添加 latent pad tokens
    texts = add_latent_pad_after_auxiliary_img(texts, latent_size, "<abs_vis_token_pad>")
    
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    # 构建 4D attention mask
    attn_mask_4d, segs = build_4d_attn(
        input_ids=batch["input_ids"],
        pad_mask=batch["attention_mask"],
        token_ids=SPECIAL_id,
        not_mask_image=args.not_mask_image,
        mask_latent=args.mask_latent,
        observation_tokens_cannot_see_question_image=True,  # 关键！
        observation_tokens_only_see_question_and_latent=True,
    )
    batch["attention_mask_4d"] = {"full_attention": attn_mask_4d}
    return batch

# 5. 运行 Teacher 前向传播并保存 Latents
with torch.inference_mode():
    for batch in dataloader:
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            image_grid_thw=batch['image_grid_thw'],
            latent_mode=True,  # 关键：启用 latent 模式
            output_latent_embeds=True,  # 输出 latent embeddings
            output_hidden_states=True,  # 输出所有层的 hidden states
        )
        
        # 保存 teacher latents
        for b in range(len(outputs.latent_embeds)):
            save_path = f"latent_{dataset_name}_{sample_id}.pt"
            torch.save({
                'latent': outputs.latent_embeds[b].cpu(),
                'hidden_states': outputs.hidden_states[b].cpu()  # 可选：保存所有层
            }, save_path)
```

---

### Stage 1-3: Latent Distillation (核心蒸馏：盲生成)

**目标**: 训练 Student 模型在**看不到** `img0` 的情况下，仅凭 `Prompt` 就能"脑补"出 $Z_{student}$，并逼近 $Z_{teacher}$。

**核心实现** (对齐 Monet 的 `modeling_qwen2_5_vl_monet.py`):

```python
# 1. Alignment Loss 函数定义
def alignment_loss(teacher_hidden_states: torch.Tensor, student_hidden_states: torch.Tensor):
    """
    计算 Teacher 和 Student hidden states 之间的对齐损失
    使用 Cosine Similarity Loss
    """
    if teacher_hidden_states.dim() == 3:  # [num_layer, num_align, dim]
        # 对齐所有层
        loss = (1 - F.cosine_similarity(
            teacher_hidden_states.to(student_hidden_states.device),
            student_hidden_states
        )).mean()
    elif teacher_hidden_states.dim() == 1:  # [dim]
        # 只对齐最后一层
        loss = 1 - F.cosine_similarity(student_hidden_states, teacher_hidden_states, dim=0)
    return loss

# 2. Latent Mode Forward (关键实现)
class Qwen2_5_VLModel(nn.Module):
    def forward(self, latent_mode=False, teacher_hidden_states_for_alignment=None, ...):
        if latent_mode:
            # 2.1 构建 latent token 位置列表
            latent_lists = [
                (input_ids[b] == self.config.latent_token_id).nonzero().flatten().tolist()
                for b in range(batch_size)
            ]
            
            total_align_loss = None
            ce_patch_pos = [[] for _ in range(batch_size)]
            ce_patch_vec = [[] for _ in range(batch_size)]
            
            for b in range(batch_size):
                latent_pos = latent_lists[b]
                prev_idx = ans_start
                
                for pos in latent_pos + [seq_len]:
                    # 2.2 Forward 非 latent 区段（文本和图像 tokens）
                    if pos > prev_idx:
                        out = self.language_model(
                            inputs_embeds=seq_embeds[:, prev_idx:pos, :],
                            position_ids=pos_ids[:, :, prev_idx:pos],
                            attention_mask=attn_mask,
                            past_key_values=past_kv,
                            output_hidden_states=True,
                        )
                        
                        # 2.3 计算 Observation tokens 的对齐损失 (Stage 2)
                        if teacher_hidden_states_for_alignment is not None:
                            for align_pos in alignment_positions:
                                if align_pos in range(prev_idx, pos):
                                    student_hidden = out.hidden_states[:, align_pos - prev_idx, :]
                                    align_loss += alignment_loss(
                                        teacher_hidden_states_for_alignment[b][:, align_ptr, :],
                                        student_hidden
                                    )
                        
                        past_kv = out.past_key_values
                    
                    if pos == seq_len:
                        break
                    
                    # 2.4 Forward Latent Token
                    # 关键：用前一个 token 的 hidden state 作为 latent embedding
                    if pos > 0:
                        prev_hidden = batch_last_hidden_state[b, pos - 1, :].unsqueeze(0).unsqueeze(0)
                        latent_embed = prev_hidden.clone() if self.training else prev_hidden.detach()
                        
                        # 收集用于 CE Loss 的 latent vectors
                        ce_patch_pos[b].append(pos)
                        ce_patch_vec[b].append(latent_embed[0, 0])
                    
                    step_out = self.language_model(
                        inputs_embeds=latent_embed.detach(),  # 关键：detach 防止梯度回传到 embedding
                        position_ids=pos_ids[:, :, pos:pos+1],
                        past_key_values=past_kv,
                        output_hidden_states=True,
                    )
                    
                    # 2.5 计算 Latent Token 的对齐损失 (Stage 3)
                    if teacher_hidden_states_for_alignment is not None:
                        if teacher_hidden_states_for_alignment[0].dim() == 2:
                            # 只对齐 latent embeddings (last layer)
                            align_loss += alignment_loss(
                                teacher_hidden_states_for_alignment[b][align_ptr],
                                latent_embed[0, 0]
                            )
                        elif teacher_hidden_states_for_alignment[0].dim() == 3:
                            # 对齐所有层的 hidden states
                            align_loss += alignment_loss(
                                teacher_hidden_states_for_alignment[b][:, align_ptr, :],
                                step_out.hidden_states
                            )
                    
                    past_kv = step_out.past_key_values
                    prev_idx = pos + 1
            
            return Qwen2_5_VLModelOutputWithPast(
                alignment_loss=total_align_loss,
                ce_patch_pos=ce_patch_pos,
                ce_patch_vec=ce_patch_vec,
            )

# 3. 完整的蒸馏训练 Forward
class Qwen2_5_VLForConditionalGeneration(nn.Module):
    def forward(self, loss_type=[], teacher_hidden_states_for_alignment=None, ...):
        # 3.1 Latent Mode Forward
        outputs = self.model(
            latent_mode=latent_mode,
            teacher_hidden_states_for_alignment=teacher_hidden_states_for_alignment,
            alignment_poss=alignment_poss,
            ce_patch_pos=ce_patch_pos,
            ce_patch_vec=ce_patch_vec,
        )
        
        loss = 0.
        loss_dict = {}
        
        # 3.2 CE Loss (Next Token Prediction)
        if "ce" in loss_type:
            logits = self.lm_head(outputs.last_hidden_state)
            loss_ce = self.loss_function(logits=logits, labels=labels, vocab_size=logits.size(-1))
            loss += loss_ce
            loss_dict['ce'] = loss_ce
        
        # 3.3 Alignment Loss (蒸馏损失)
        if 'alignment' in loss_type:
            loss += outputs.alignment_loss
            loss_dict['alignment'] = outputs.alignment_loss
        
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            loss_dict=loss_dict,
        )
```

**Loss 设计 (双重蒸馏)**:

1.  **Latent Alignment Loss (Cosine Similarity)**:
    $$L_{align} = 1 - \cos(Z_{student}, Z_{teacher})$$
    
    *   **关键技巧**: **Latent-only Backpropagation**。计算此 Loss 时，使用 `detach()` 确保梯度**只回传**给生成 $Z$ 的路径。

2.  **Reflection NTP Loss**: 基于生成的 $Z_{student}$ 预测后续的 Reflection Text，保证 $Z$ 具备语义可解释性。

3.  **Generation Loss (Optional)**: 将 $Z_{student}$ 输入 MMDiT，计算生成 `img1*` 的 Flow Matching Loss。

---

### Stage 2: Multi-Turn Latent Reasoning (多轮迭代)

**目标**: 在 Latent 空间完成"草图 → 反思 → 修改"的闭环。

**流程**:
1.  **Drafting**: 模型基于 Prompt 生成第一轮 Latent $Z_0$（相当于脑海中的 Draft）
2.  **Reflection**: 基于 $Z_0$，模型生成文本 Reflection（或隐式 Reflection Latent $R_0$）
3.  **Refinement**: 基于 $Z_0$ + Reflection，模型生成第二轮 Latent $Z_1$

**蒸馏实现**:

```python
# 多轮蒸馏训练
class MultiTurnLatentDistillationTrainer:
    def __init__(self, model, teacher_latents_dir):
        self.model = model
        self.teacher_latents = self.load_teacher_latents(teacher_latents_dir)
    
    def train_step(self, batch):
        # 1. 加载 Teacher Latents
        teacher_z0 = batch['teacher_latent_z0']  # 第一轮 latent
        teacher_z1 = batch['teacher_latent_z1']  # 第二轮 latent
        
        # 2. Student Forward (不看 img0)
        # 第一轮：只基于 Prompt 生成 Z0
        outputs_round1 = self.model(
            input_ids=batch['input_ids_round1'],
            latent_mode=True,
            teacher_hidden_states_for_alignment=[teacher_z0],
            loss_type=['alignment', 'ce'],
        )
        
        # 3. 将 Student 的 Z0 注入到第二轮输入
        student_z0 = outputs_round1.ce_patch_vec
        
        # 第二轮：基于 Prompt + Z0 + Reflection 生成 Z1
        outputs_round2 = self.model(
            input_ids=batch['input_ids_round2'],
            latent_mode=True,
            ce_patch_vec=student_z0,  # 注入 Z0
            teacher_hidden_states_for_alignment=[teacher_z1],
            loss_type=['alignment', 'ce'],
        )
        
        # 4. 计算总损失
        total_loss = outputs_round1.loss + outputs_round2.loss
        return total_loss
```

---

### Stage 3: End-to-End Generation & Alignment

**目标**: 确保最终的 Latent 能被 MMDiT 完美解码为图像。

**实现**:

```python
class LatentToImageGenerator:
    def __init__(self, llm_model, mmdit_model, vae):
        self.llm = llm_model
        self.mmdit = mmdit_model
        self.vae = vae
    
    def forward(self, prompt, num_inference_steps=50):
        # 1. LLM 生成 Latent Z
        llm_outputs = self.llm(
            input_ids=prompt_ids,
            latent_mode=True,
            output_latent_embeds=True,
        )
        z_final = llm_outputs.latent_embeds  # [B, num_latents, hidden_dim]
        
        # 2. 将 Z 投影到 MMDiT 条件空间
        # 使用 Qwen-Image 的语义特征接口
        semantic_condition = self.project_to_mmdit(z_final)
        
        # 3. MMDiT 生成图像
        # 初始化噪声
        latents = torch.randn((B, 4, H//8, W//8), device=z_final.device)
        
        # Flow Matching 采样
        for t in timesteps:
            velocity = self.mmdit(
                latents,
                timestep=t,
                encoder_hidden_states=semantic_condition,  # Z 作为条件
            )
            latents = latents + velocity * dt
        
        # 4. VAE 解码
        images = self.vae.decode(latents)
        return images
    
    def compute_flow_matching_loss(self, z_final, target_images):
        """计算 Flow Matching Loss"""
        # 编码目标图像
        target_latents = self.vae.encode(target_images)
        
        # 采样时间步
        t = torch.rand(B, device=z_final.device)
        
        # 插值
        noise = torch.randn_like(target_latents)
        noisy_latents = (1 - t) * target_latents + t * noise
        
        # 预测速度场
        semantic_condition = self.project_to_mmdit(z_final)
        predicted_velocity = self.mmdit(noisy_latents, t, semantic_condition)
        
        # 真实速度场
        true_velocity = noise - target_latents
        
        # MSE Loss
        loss = F.mse_loss(predicted_velocity, true_velocity)
        return loss
```

---

## 关键技术点总结 (Qwen-Image 特异性)

### 1. 双流条件的融合

Qwen-Image 使用 VAE (Pixel) 和 LLM (Semantic) 双流条件。

**策略**: Latent CoT $Z$ 应替代 LLM Semantic stream。在 Stage 3 推理时，MMDiT 的输入条件变为：`Empty VAE Latent` + `Visual Latent Z`。这意味着 $Z$ 必须同时承载语义和结构布局信息。

### 2. 特殊 Token 路由

```python
# 引入特殊 tokens
SPECIAL_TOKENS = {
    "<abs_vis_token>": latent_start,     # 开始 latent 生成
    "</abs_vis_token>": latent_end,      # 结束 latent 生成
    "<abs_vis_token_pad>": latent_pad,   # latent 占位符
    "<observation>": obs_start,           # 观察开始
    "</observation>": obs_end,            # 观察结束
}

# 模型 Router 逻辑
def generate_with_latent_routing(self, input_ids, ...):
    while not finished:
        logits = self.forward(input_ids)
        next_token = logits.argmax(-1)
        
        if next_token == latent_start_id:
            # 进入 Latent 模式：停止采样离散 token，直接输出连续 hidden state
            for _ in range(num_latent_tokens):
                hidden = self.get_last_hidden_state()
                latent_tokens.append(hidden)
                # 将 hidden state 作为下一步输入
                input_ids = self.embed_hidden_as_input(hidden)
        else:
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
```

### 3. 梯度阻断 (Stop-Gradient)

这是 Monet 成功的关键。在 Stage 1-3 计算对齐 Loss 时，使用 `detach()` 确保只有 Latent 生成相关的参数被更新：

```python
# 关键：detach latent embedding 防止梯度回传污染
latent_embed = prev_hidden.detach()  # Stop gradient here!

# 计算对齐损失时，teacher 也要 detach
loss = alignment_loss(
    teacher_hidden_states.detach(),  # Teacher 不参与梯度
    student_hidden_states             # 只有 student 参与梯度
)
```

### 4. 数据构建

利用 Qwen-Image 强大的理解能力，对训练数据进行 **Recaptioning** 和 **Quality Filtering**，确保用于蒸馏的 Teacher 数据本身是高质量的。

```python
# 数据格式示例 (COT Triplet)
{
    "sample_id": "xxx",
    "prompt": "将猫的颜色改为橙色",
    "img0": {"bytes": <原始图像>},           # 原始图
    "img1": {"bytes": <中间结果图像>},        # 第一轮编辑结果
    "img2": {"bytes": <最终结果图像>},        # 最终编辑结果
    "reflection1": "<|reflection_start|>分析：原图中有一只灰色的猫...<|reflection_end|>",
    "reflection2": "<|reflection_start|>评估：第一轮编辑成功改变了颜色...<|reflection_end|>"
}
```

---

## 训练脚本示例

```bash
#!/bin/bash
# Stage 1: Warm-up 训练 (sft_stage1)
cd /root/autodl-tmp/Monet

torchrun --nproc-per-node=8 --master-port=29501 -m src.main \
  --stage sft_stage1 \
  --epochs 4 \
  --bsz 1 \
  --grad_accum_steps 16 \
  --data_path ./data \
  --load_model_path ./models/Qwen2.5-VL-7B \
  --save_model_path ./checkpoints/stage1_warmup \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --wandb_name stage1_warmup \
  --shuffle_train \
  --max_img_tokens 1280
```

```bash
#!/bin/bash
# Stage 2: Teacher Latent 蒸馏 (sft_stage2)
torchrun --nproc-per-node=8 -m src.main \
  --stage sft_stage2 \
  --epochs 6 \
  --bsz 1 \
  --grad_accum_steps 32 \
  --data_path ./data \
  --teacher_latent_dir ./teacher_latents \
  --load_model_path ./checkpoints/stage1_warmup \
  --save_model_path ./checkpoints/stage2_teacher_latent \
  --loss_type ce alignment \
  --alignment_weight 1.0 \
  --latent_size 10 \
  --mask_question_image \
  --observation_tokens_cannot_see_question_image
```

```bash
#!/bin/bash
# Stage 3: Student Latent 蒸馏 (sft_stage3)
torchrun --nproc-per-node=8 -m src.main \
  --stage sft_stage3 \
  --epochs 10 \
  --bsz 1 \
  --grad_accum_steps 32 \
  --data_path ./data \
  --teacher_latent_dir ./teacher_latents \
  --load_model_path ./checkpoints/stage2_teacher_latent \
  --save_model_path ./checkpoints/stage3_student_latent \
  --loss_type ce alignment \
  --alignment_weight 1.0 \
  --latent_size 10 \
  --mask_question_image \
  --observation_tokens_cannot_see_question_image
```

---

## 推理流程

```python
# 推理时的 Latent 生成
def inference_with_latent_cot(model, prompt, num_latent_tokens=10):
    # 1. 编码 prompt
    input_ids = tokenizer.encode(prompt)
    
    # 2. 添加 latent 开始 token
    input_ids = input_ids + [latent_start_id]
    
    # 3. 生成 latent tokens
    latent_embeddings = []
    hidden_state = model.get_hidden_state(input_ids)
    
    for _ in range(num_latent_tokens):
        # 用上一个 hidden state 作为当前 latent embedding
        latent_embeddings.append(hidden_state)
        # Forward 一步
        hidden_state = model.forward_one_step(hidden_state)
    
    # 4. 添加 latent 结束 token
    input_ids = input_ids + [latent_end_id]
    
    # 5. 继续生成文本 reflection（可选）
    output_ids = model.generate(
        input_ids,
        latent_embeddings=latent_embeddings,
        max_new_tokens=256,
    )
    
    # 6. 将 latent 传给 MMDiT 生成图像
    z_final = torch.stack(latent_embeddings, dim=1)
    image = mmdit_generate(z_final)
    
    return image, tokenizer.decode(output_ids)
```

---

通过这套方案，Qwen-Image 将不再只是简单地将文本映射为像素，而是通过"内隐想象（Latent Generation）"和"内隐反思（Latent Reflection）"来规划生成过程，实现真正的 System-2 视觉生成。
