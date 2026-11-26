# M3 训练计划：情绪忠实 LoRA + 表情自锚定（v1，Round 91）

## 目标
让音频驱动人像底座（EchoMimic V1）的生成脸**响应语音情绪**（SER↔FER 一致性、表现力提升），并通过 3 变体消融定位增益来源。

## 监督信号来源（关键决策）
- **首选：RAVDESS 视频版**（Audio_Video_Actors_01-24.zip，zenodo 1188976，~2GB）
  - 真实演员表情视频 + 同源音频 = 天然 (audio, expressive_video) 配对监督
  - 参照帧：从同一视频抽第一帧（中性偏多）或固定 actor 中性帧
  - 备选：CREMA-D（91 actors，VideoFlash）
- 训练目标：扩散 SFT（MSE），重建真实表情动态；**SER↔FER 一致性作为可验证奖励**用于 (a) 训练数据过滤 (b) checkpoint 选择 (c) reward-weighted 采样（可选第三阶段）

## 数据流水线
1. 视频切 24fps 窗口：12 帧/窗（context_frames），重叠 0，每 clip ~24-32 窗
2. 参照帧 = 每窗首帧偏移 -1s（保证非当前帧，防止恒等映射）
3. 512→256×256 训练（显存），推理 512
4. 过滤：FER(参考帧)=neutral 且 FER(窗中帧) expr<0.3 的窗丢弃（防止学中性）；GT 情绪按 manifest
5. 验证集：actor 23-24 留出（speaker-disjoint）

## LoRA 配置
- 目标模块：denoising_unet + reference_unet 的 to_q/to_k/to_v（r=32, alpha=64, dropout 0.05）
- 冻结：VAE、image_encoder、motion_module、face_locator
- 优化器：AdamW lr=1e-4，bf16，batch 4 窗 × 4 GPU（DDP 或单卡逐窗）
- 步数：~8k 步（1440 clips × ~26 窗 / batch16 ≈ 2.3k step/epoch，3-4 epochs）

## 3 变体消融
| 变体 | 配置 | 验证假设 |
|---|---|---|
| V-A | 纯 LoRA SFT | SFT 重建即可注入情绪响应？|
| V-B | LoRA + 表情锚定 token（GT 窗的 FER 嵌入作为额外 cross-attn 条件）| 显式情绪条件是否必要 |
| V-C | V-B + 表现力方差正则（batch 内 FER 方差惩罚）| 正则是否进一步稳定 |

## 评测协议（与 M2 同裁判，无泄漏）
- 主指标：SER↔FER agree_js、argmax_match、expressiveness（生成脸 vs 音频）
- 对照：Arm1 baseline 曲线（expr 0.079）
- 身份保持：ArcFace(ref, generated) 余弦
- 唇同步：SyncNet confidence
- 泛化：actor-disjoint 验证 + 后期加 MEAD/CREMA-D 外部集

## 算力
- GPU 4-7（4×H100，现空闲）足够；无需额外申请
- 时间预估：数据管线 0.5 天 + 训练 1 天（8k 步）+ 评测 0.5 天
