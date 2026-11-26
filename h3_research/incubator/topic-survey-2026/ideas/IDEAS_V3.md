# Idea V3（Round 97，基于 220 篇前沿调查；用户点名方向：蒸馏加速/可控生成）

调查规模：12 组查询 × 20 条 = 240 条 → 去重 **220 篇**（19,150 总被引），加旧库 115 篇，累计 **335 篇**。文件：`refs/survey_v2/merged.jsonl` + `SURVEY_V2.md`。

## 各区前沿速览（被引最多的锚点）

- **蒸馏加速**：ADD(1101)、Self-Forcing(475, NeurIPS)、Rolling/Self-forcing++/Causal-forcing(2026 集中爆发)——实时 AR 视频扩散是当下最热
- **缓存加速**：TeaCache(247)、token-wise cache(167)、learning-to-cache(153)、FasterCache(100)
- **推理时 scaling**：beyond-denoising-steps(216)、best-of-N 分析(99)、Video-T1(48)、beam search(43)、evolutionary search(32)——新热点，方法都在 LLM 侧成熟后迁移
- **DPO 后训练**：VideoDPO(122)、DenseDPO(35)、PhyGDPO/RDPO(物理一致性 DPO，2025-2026 新)
- **可控生成**：MotionCtrl(627)、CameraCtrl(452)、DragNUWA(288)——成熟；**MotionStream(62)/FlashMotion(1)/RealCam(2026)** 已在做"实时/少步+可控"
- **世界模型/物理**：物理视角(204)、物理常识基准(169)、WorldSimBench(123)
- **奖励模型**：VisionReward(150)、UnifiedReward(196)——开源可用

## 候选 idea（每个都过了查重，含训练）

### C1 ★主线推荐：视频生成的计算最优推理时 scaling——Seeds × Samples × Steps
- **问题**：现有推理 scaling（Video-T1 等）都是 per-prompt 均匀预算。但质量-算力曲线**因 prompt 剧烈变化**（我们已证明：seed 方差因 prompt 差 10×）。该花多少 seed/样本/步数没人回答。
- **做法**：①测质量-算力曲面：{50 步 vs 开源少步蒸馏检查点（Self-Forcing/CausVid 用 Wan1.3B，开源）} × {N samples} × {K seeds} × reward model（VisionReward/VideoScore 开源）②**训练 per-prompt 自适应分配器**（预测 cheapest config 达到目标 reward）③用我们 seed-resolved 协议保证测量可靠。
- **贡献**：T2V 版"compute-optimal inference"（LLM 侧 455 引的姊妹篇）；蒸馏模型让 best-of-N 经济学改变（4 步模型 1 次采样≈50 步的 1/12 算力——N 该开多大？）——没人算过这笔账。训练实在（分配器），蒸馏/搜索/奖励三热点交叉，GPU 可行。
- **查重**：LLM 侧 "What If We Allocate Test-Time Compute Adaptively?"(2026)；T2V 侧无自适应 per-prompt 预算工作。
- **风险**：分配器泛化（prompt 分布外）；reward model 偏差。

### C2：控制保真度 × 蒸馏的系统量化 + 保真感知蒸馏损失
- "蒸馏掉不掉相机/轨迹控制"的系统研究 + 控制保真感知蒸馏损失（train）。
- **查重警讯**：MotionStream(62)/FlashMotion/RealCam/4DStreamCtrl 全是 2026——空白正被快速填补，方向本身热但窗口窄。若做，须以"测量协议+损失"差异化。

### C3：加速技术的隐藏代价——caching 家族是否放大 seed 噪声
- TeaCache/FasterCache 都报"质量不变"，但 caching 改变去噪轨迹→可能放大对初始噪声的敏感度。用 seed-resolved 协议量化。若成立=所有用 TeaCache 报的 benchmark 数字都需复核——影响面大。
- 训练弱（可加 learnable cache 校准作修复）。适合作 C1 的一节。

### C4：物理一致性 DPO 的评估审计
- PhyGDPO/RDPO(2025-2026 新) 训练物理一致性，但其增益是否超过评估噪声（DD 噪声载波！）没人检验。我们的协议直接适用。训练在别人模型上。副线。

### C5：相机控制感知 DPO（CameraCtrl 保真为偏好信号）
- CameraCtrl(452 引) 成熟 + DPO 热的交叉；查重未见。数据构造（相机轨迹偏好对）是瓶颈，风险中。

## 推荐：**C1 主线**（C3 作为其一节，C2 备选观察窗口）

标题方向："Compute-Optimal Test-Time Scaling for Text-to-Video Generation: How Many Seeds, Samples, and Steps Does a Prompt Need?"
- 三个热点（推理 scaling / 蒸馏加速 / 奖励模型）交叉 + 我们的独家资产（seed 方差数据、测量协议、GPU）
- 训练：分配器（监督/RL 皆可）；可选加 LoRA reward
- 对比 LLM 侧 455 引的 inference scaling laws——T2V 版本空白明确
