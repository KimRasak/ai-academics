# C2-Full 训练计划：Control-Preserving Distillation（8×H100 档位）

**"What Does Distillation Forget? Measuring and Preserving Control Fidelity in Few-Step Video Generators"**

## 一句话

系统性测量少步蒸馏对相机/轨迹控制保真度的损失 + 训练控制保持蒸馏损失把它找回——**用户点名的两大方向（蒸馏加速 × 可控生成）的正交缺口**，且复用我们已有的 Wan-Fun-Control 管线与 seed-resolved 测量纪律。

## 查重状态（2026-09-13）

- 系统类：MotionStream(62)/FlashMotion/RealCam 在**造系统**，不做测量与损失设计
- "Data-Forcing Distillation"(2026) 恢复的是 diversity/fidelity，非控制依从
- **空白确认**：控制保真退化的系统测量 + 控制保持蒸馏损失，0 命中
- 附加空白（独家）：**控制依从性本身的 seed 噪声**——所有控制论文报单 seed 结果，无人测控制误差的跨 seed 方差（我们的协议直接上）

## 为什么 8×H100（同 S6-Full 逻辑）

1. 机制消融需自训学生（控制条件化的 Self-Forcing 式学生+critic），非修补现成检查点
2. 消融矩阵：{vanilla 学生 / +控制保持损失} × {相机 / 轨迹} × 步数 {1,2,4,8} = 6-8 个训练 run
3. 教师控制条件 rollout 数据 + 大规模评测吞吐（每检查点 × 每 seed 数 × 控制信号扫描）

## 数据与模型（全部开源/已有）

- **教师**：Wan2.1-Fun-Control 1.3B（管线已有！50 步全步控制生成）
- **控制信号**：相机轨迹（RealEstate10K pose 序列，公开）+ 2D 点轨迹（DragNUWA 式，公开数据）
- **学生配方**：Self-Forcing/CausVid 公开配方（学生+伪评分 critic+对抗）
- **控制误差度量**：相机 pose 误差（已知轨迹 vs 估计轨迹）、轨迹端点/路径误差——确定性可算

## 8×H100 分相计划（1 个月）

| 周 | 内容 | 卡位 |
|---|---|---|
| 1 | 教师基线：控制依从性的 seed 噪声测量（50 步 × 12 seeds × 轨迹扫描）= 第一个结果；LoRA-student 预验证蒸馏收敛 | 4 卡生成 + 4 卡 LoRA |
| 2-3 | 完整蒸馏：vanilla 学生 vs +控制保持损失（各 2-3 run，FSDP 4-6 卡）| 6 卡训练 + 2 卡滚动评测 |
| 4 | 评测矩阵（步数×控制类型×seed）+ 消融 + 写作 | 8 卡评测 |

## 算力核算

- 显存：学生+critic FSDP ≈ 25-35GB/GPU，49 帧起步 → 单卡 batch 1-2 可行
- 教师 rollout ~2,500 控制视频 ≈ 3 GPU-天；6-8 训练 run × 3-4 GPU-天（4-6 卡并行≈1.5 周）；评测 ≈ 8 GPU-天（8 卡 1 天）
- 合计 ≈ 3 周训练实验 + 1 周写作

## 风险与对策

- 蒸馏收敛 → 周 1 LoRA 预验证；49 帧起步
- 控制保持损失与质量冲突 → 权重扫描即核心消融（预期 trade-off 曲线本身是贡献）
- 系统类论文抢发 → 我们的差异化是"测量协议+损失+机制"，且第一周即出测量结果占位

## 设备申请

- **主申请：8×H100 × 4 周**（理由同上：自训学生 + 消融矩阵 + 评测吞吐）
- **可选 +4（非必需）**：训练与教师 rollout 并行，省约 1 周
