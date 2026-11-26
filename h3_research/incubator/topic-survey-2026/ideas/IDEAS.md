# 候选 Idea 清单与新颖度对比（Round 92，响应用户要求：≥5 个 idea → 查重 → 对比 → 选优）

前置约束：必须含训练（新要求 #10）；引用 ≥30（#9）；数据资产 = 576 seed-resolved 视频（3 模型）+ 2,392 人类分数 + RAFT 连续运动分数（可提取）+ 已量化的噪声底。

---

## I1. Seed-Consistent Soft-DD：可微校准替代硬阈值
- **机制**：VBench DD = RAFT 逐对 max_rad 硬阈值 + 计数（代码已核实）。提取连续分数，训练单调软映射（learned temperature/scale + soft-count），目标 = 跨 seed 一致性（最小化 within-prompt 方差）+ 排序保真（约束 vs 原 DD）
- **训练**：真实梯度训练（校准参数，train seeds 0–5 / test 6–11）
- **新颖度查重**：无 "binarization calibration for benchmark metrics" 命中；uncertainty-aware VQA（如 action quality score-distribution）是不同问题。**较新但单薄**
- **风险**：增益可能小（DD 已近双峰）；轻量训练可能被审稿人认为"训练不足"

## I2. 人类对齐可学习聚合器
- **机制**：6 指标 → LPF 的回归（ridge/GBM/MLP，2,392 人类分数，5-fold CV），替代无权平均
- **新颖度查重**：metric fusion / learning-to-rank IQA 成熟（QualiVision 2025、GraphIQA 2022 等）——**不新**
- **风险**：高。仅作组件不单独立 idea

## I3. Seed-Variance Predictor：预测"这条视频需要几个 seed"
- **机制**：训练回归器（输入视频帧特征/指标向量）→ 预测该视频的跨 seed 指标方差；据此做自适应 seed 分配（方差大的视频多采、小的少采）+ 方差加权 leaderboard
- **训练**：真实回归训练；**我们 576 视频天然自带标签（within-prompt SD）**——直接监督，无需新标注
- **新颖度查重**：无 "predicting cross-seed evaluation variance" 命中；uncertainty-aware VQA 预测的是人类评分不确定性，非 generator-seed 评估方差。**强新颖**；且是"评估效率"实际工具
- **风险**：16 prompts 的方差标签噪声大；需 cross-model 泛化验证

## I4. T2V 集成蒸馏（生成侧训练）
- **机制**：K seeds 生成 → LoRA 微调 Wan 1.3B 蒸馏集成，降低输出质量方差
- **训练**：重（1.3B LoRA 可行但周期长）
- **新颖度查重**：图像域 ensemble/distillation 常见；T2V seed-variance 蒸馏未见，但"降方差 vs 多样性"权衡根本未解决，且不解决 evaluator 噪声（我们已证明噪声主要在 metric 侧 DD）——**方向错配**
- **风险**：高（算力+论证不自洽：我们的数据显示 metric 侧才是瓶颈）

## I5. Pairwise Rank-Tester：训练的配对显著性检验器
- **机制**：学习成对比较器（两模型各 k seeds 的分数分布 → 判优 + 校准 p 值）
- **训练**：真实但数据极小（16 prompts × 3 模型的配对），**过拟合风险致命**
- **新颖度查重**：learning-to-rank + 假设检验组合未见，但数据撑不起训练。**淘汰**

## I6. 运动分数集成自蒸馏（I1 深化）
- **机制**：跨 seed RAFT 分数集成 → 蒸馏进小型时序 motion head（3D conv/temporal pooling），替代 RAFT+阈值整条链
- **训练**：重于 I1，端到端可微
- **新颖度查重**：RAFT 蒸馏有先例（flow distillation），但目标函数是"跨生成 seed 一致性"——未见
- **风险**：工程量大；增益需先由 I1 验证才值得

---

## 对比矩阵

| | 新颖度 | 训练实在度 | 数据适配 | 算力 | 与诊断衔接 | 风险 |
|---|---|---|---|---|---|---|
| I1 软校准 | 中+ | 中 | ★★★ | 分钟级 | 直接 | 增益小 |
| I2 聚合器 | 低 | 中 | ★★ | 分钟级 | 部分 | 不新 |
| I3 方差预测 | **高** | 中高 | ★★★ | 分钟–小时 | 直接 | 标签噪声 |
| I4 生成蒸馏 | 中 | 高 | ★ | 天级 | 错配 | 算力+逻辑 |
| I5 检验器 | 中 | 低 | ★ | 分钟 | 部分 | 过拟合 |
| I6 蒸馏头 | 中高 | 高 | ★★★ | 小时级 | 直接 | 工程量 |

## 选定：**I3 + I1 组合**（一篇论文的两个训练组件）

**"Learning to Predict and Reduce Seed Noise in Composite Video Metrics"**
1. 诊断（已有，四轮审稿通过）：seed 噪声底 5.3× 于 leaderboard gap；DD 是载波
2. **训练组件 A（I1）**：seed-consistent 软 DD 校准 → 直接降 flip 率/方差
3. **训练组件 B（I3）**：seed-variance 预测器 → 自适应 seed 分配 + 方差加权排名
4. 保真约束：人类相关性不降（2,392 分数）；跨模型验证（H3 训练 → Wan/LTX 测试或各自训练）
5. 引用：现有 12 + I1–I6 查重新增（uncertainty-aware VQA、score fusion、RAFT 蒸馏、adaptive evaluation、statistics）≥30 可达

**对最近竞品的差异**："Evaluating the Evaluators"（2026）类工作是 metric reliability 的**分析**；我们做**训练修复**（learned calibration + learned variance predictor）——从"测量噪声"到"预测并降低噪声"。

## 下一步（验证优先，不急写论文）
1. 提取 RAFT 连续分数（576 视频，脚本已部署，bug 修复中）
2. I1 训练 + 消融（train/test seed 划分）
3. I3 训练 + 跨模型验证
4. 增益确认后才写论文 v2
