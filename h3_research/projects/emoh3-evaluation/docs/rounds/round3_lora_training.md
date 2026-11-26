# Round 3 产出 + 下一步验证配方（LoRA → 推理验证）

## 本 round 已达成：8×H100 训练链路全通（方法支柱 2 可行性实测）

- 训练环境 `h3ft`（/gluster_hdd_cv/user/jinzili/env/h3ft）：Python 3.11 + torch 2.11.0+cu128 +
  pinned diffusers（H3 集成 commit abc5e9b，in-memory 命名 to_qkv/linear_1/linear_2）+
  transformers 5.16.1 + peft 0.20.0 + deepspeed 0.19.6。
- 训练数据：`/gluster_hdd_cv/user/jinzili/tmp/pilot/traindata/train.jsonl`（9 条情绪视频，
  reference=情绪肖像，caption=情绪描述）→ cache `/gluster_hdd_cv/user/jinzili/tmp/pilot/traindata/cache/`
  （9 个 .pt：video latents + 真实 audio latents + Qwen3VL layer-50 text）。
- **8×H100 训练跑完 300 步**：`runs/lora/checkpoint-{50..300}.pt`，8 卡 ~72GB/~88% util，
  loss 0.4–1.1 稳定。LoRA = r16 α32，target modules = to_qkv/to_out.0/linear_1/linear_2（108 个 PEFT 张量）。
- 关键修复（都记录在本 session）：
  1. prepare_cache `row.get("target_video", row["video"])` 的 eager-default bug → `or` 修复。
  2. torchaudio 2.11 需要 torchcodec（装不上/装上有库冲突）→ 改 stdlib `wave` 模块读 32kHz s16 立体声。
  3. **034 内核无 IPv6**，torch 2.11 TCPStore 连 `::ffff:ip` 必失败 → train.py 改 `file://` FileStore init。

## 下一步：用训练好的 LoRA 做推理验证（拿论文核心结果表）

**目标**：基座 vs LoRA 在同样 8 个跨情绪 case 上跑生成，Qwen2-VL 判定情绪保真，期望 LoRA 提升（base 7/8 失败）。

**推理 API（已探明，pinned diffusers / h3ft env）**：
```python
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3Ref2VABlocks, MiniMaxH3Reference,
)
# 1. 加载 Ref2VA 管线（本地权重目录即可）
pipe = MiniMaxH3Ref2VABlocks().init_pipeline("/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3")
pipe.load_components(dtype=torch.bfloat16)   # transformer 注册为 transformer_ref
# 2. 注入 LoRA（照抄官方 inference_minimax_h3.py 的 load_lora_adapter 逻辑）
#    - 从 runs/lora/checkpoint-300.pt 取 ck["model"]（已是 .lora_A/.lora_B.default.weight 裸键）
#    - transformer.add_adapter(LoraConfig(r=16, lora_alpha=32, init_lora_weights=False,
#        target_modules=["to_qkv","to_out.0","linear_1","linear_2"]))
#    - load_state_dict(state_dict, strict=False) + set_adapters("default", weights=scale)
# 3. 生成（重要：prompt/references/height/width/num_frames）
pipe(prompt="<Picture 1> She smiles joyfully. Keep the same anime character.",
     references=[MiniMaxH3Reference(image="/path/to/ref.png")],
     height=768, width=768, num_frames=124, ...)   # num_frames 自动对齐 17n+5
```
**注意**：
- 权重盘上命名是旧式(to_q/to_k/to_v/ff.net.0.proj)，pinned diffusers 加载时映射成新式(to_qkv/linear_1/linear_2)
  —— LoRA 必须用 h3ft 的 pinned diffusers 加载，**不能**用 minimax_h3_vxf（用户推理脚本那个 env 是新旧不匹配）。
- 单卡放不下（33B transformer + 32B text encoder ≈ 130GB）：走**多卡 FSDP2**（官方脚本
  `inference_minimax_h3.py --fsdp2` 需要 torch≥2.6，h3ft=2.11 满足；但它的 `minimax_h3_ref2va_pipeline.py`
  与 pinned diffusers 不兼容，需改用上面 `MiniMaxH3Ref2VABlocks().init_pipeline` 的方式重建）。
- 或者：文本先单独用 Qwen3VL-32B 编码好（复用 prepare_cache 的 text 编码逻辑），
  再在 8 卡上只放 transformer 做 diffusion —— 这样 8×H100 每卡 ~9GB，调度最省。
- 生成完 → 抽帧 → 跑 `/gluster_hdd_cv/user/jinzili/tmp/pilot/judge_emotion.py` 判情绪 → 出对比表。

## 本 round 关键文件
- `/gluster_hdd_cv/user/jinzili/tmp/h3ft/` — 训练器 repo（已打补丁：prepare_cache.py 2 处、train.py 2 处）
- `/gluster_hdd_cv/user/jinzili/tmp/pilot/runs/lora/checkpoint-300.pt` — 8×H100 训练的 LoRA 主 checkpoint
- `/gluster_hdd_cv/user/jinzili/tmp/pilot/traindata/` — 9 条 manifest + cache
- `/gluster_hdd_cv/user/jinzili/env/h3ft/` — 训练环境（可直接复用）
