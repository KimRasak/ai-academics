# 候选 Idea V2（Round 95，响应：别光新颖度，要对学术界有实质贡献 + 必须含训练）

选优标准升级：①改变领域做法（方法/标准）②提供公共资源（数据/工具）③回答领域关心的问题（归因/机理）④训练必须实在。新颖度只是门槛。

---

## J1. Seed-Consistency Regularized Fine-Tuning（生成侧训练干预）
- **机制**：LoRA 微调 Wan（1.3B，空闲 GPU 可做），损失 = 跨 seed 指标一致性正则 + 质量保真。回答问题：**评估噪声有多少是生成器可训练掉的吗？**
- **贡献面**：无论结果正负都是发现——若 LoRA 后 within-prompt 方差大降→"评估噪声部分是生成器属性，可训练消除"，生成+评估两个社区都受用；若不降→归因坐实 metric 侧。方法本身（SC-LoRA）是新训练目标。
- **查重**：邻域有 "Prompt-Residual Seed Shaping"(2025, T2I)、"Noise consistency regularization for subject-driven synthesis"(2025)、seed-robust safety alignment（T2I）——**T2V + 评估噪声归因角度未见**。
- **训练**：实在（LoRA 微调，GPU 数小时）。

## J2. Metric 侧训练修复（原 I1+I3：软 DD 校准 + 方差预测器）
- **贡献面**：给 Sci-VBench 一族 benchmark 打补丁——**贡献面窄**（上次被否的主因），保留作 J1 的对照/组件。

## J3. Seed-Resolved 评估数据集 + 报告标准（扩规模 + 公共资源）
- **机制**：扩到 ~50 prompts × 3–4 模型 × 8 seeds（≈1,600 视频，GPU 空闲正好），发布逐视频指标 + 连续运动分数 + 人类分数 join；附统计报告指南（配对 SE、ICC、flip 率）。
- **贡献面**：社区资源 + 改变 leaderboard 报告习惯（引用我们统计框架）。数据集论文是实打实的学术贡献。
- **查重**：无 multi-seed T2V 评估数据集命中；"Beyond statistical significance"(2025) 是 NLP 侧不确定性报告——支持需求存在、T2V 空白。
- **训练**：弱（除非并入方差预测器作配套工具）。

## J4. Seed-Stable Learned Evaluator（评估侧学习，训练以 seed 稳定性为设计准则）
- **机制**：训练轻量 learned evaluator（人类分数监督 + **跨 seed 一致性目标函数**）→ 同时人齐且 seed 稳定。
- **查重**：VideoScore2(2025)、AIGV-Assessor(2025)、Q-Align 系 crowded——**差异化只在 seed-stability 训练准则**，需证明 crowded 场 didn't consider it。
- **贡献面**：中——拥挤赛道里的增量。风险：与 SOTA 比拼会输资源。

## J5. Noise-Aware Leaderboard 聚合（分层贝叶斯/学习收缩 + 校准排名）
- **机制**：分层模型（prompt/seed/model 效应分解）输出不确定性校准的排名；训练=模型拟合。
- **贡献面**：方法论标准，但"贝叶斯拟合"训练感弱、审稿人可能归为 stats。

## J6. 运动表示跨 seed 一致性预训练（原 I6）
- 贡献面窄（单指标），淘汰理由同 J2。

---

## 对比（贡献导向）

| | 学术贡献面 | 训练实在度 | 可行性(空闲GPU) | 查重空隙 | 风险 |
|---|---|---|---|---|---|
| J1 SC-LoRA | **大**（两社区+归因问题） | 高（LoRA） | 高 | 有（T2V 空白） | 训练不收敛 |
| J2 metric 修复 | 小 | 高 | 高 | 有 | 已验证混合 |
| J3 数据集+标准 | 大（资源+惯例） | 弱 | 高（生成任务本身） | 有 | 规模仍偏小 |
| J4 stable evaluator | 中 | 高 | 中 | 拥挤 | 竞争劣势 |
| J5 贝叶斯聚合 | 中 | 弱感 | 高 | 部分 | 归类风险 |

## 选定：**J1 为主线 + J3 为资源层 + J2 为对照** 的三贡献论文

**"Where Does Evaluation Noise Live? Seed-Resolved Attribution and Trainable Reductions for Text-to-Video Evaluation"**
1. **数据集**（J3 扩规模版）：seed-resolved T2V 评估测试床（公开）
2. **归因科学**：方差分解 + 生成侧干预（J1 SC-LoRA）vs metric 侧干预（J2）→ 回答"噪声在哪、能否训练掉"
3. **报告标准**：基于分解的 leaderboard 报告建议

贡献组合 = 资源（数据集）+ 机理（归因）+ 方法（SC-LoRA 新训练目标）+ 实践（报告标准）——四个维度都是学术界可用的东西。训练两处实在：LoRA 微调（生成侧）+ 校准训练（metric 侧）。

## 下一步
1. 扩数据集生成（50 prompts × 3 模型 × 8 seeds，GPU 空闲批量跑，~1,600 视频）
2. SC-LoRA 训练管线搭建（Wan 1.3B，cross-seed 一致性损失设计）
3. 查重 jsonl 已存 refs/（novelty_j1/j4j5/j3）


---

# V3 修订（Round 96，用户否掉"跨 seed 一致性"目标）

**否决理由（成立）**：教生成器抹平 seed 差异 = 抹掉多样性，与 seed 的用途相悖；且"一致性正则"作为贡献故事牵强。

## K5（新主线）：学习型运动质量度量——人类监督 + seed 稳定协议
- **机制**：诊断已证明 DD = 二值化 RAFT 启发式，是噪声载波且与人类不相关（ρ≈0）。训练一个**学习型运动质量回归器**（视频/光流编码器 + 头，人类 LPF 分数监督 + 公开 VQA 数据预训练），产出连续、人齐、可微的运动度量；在我们 seed-resolved 协议下对比 DD：seed 方差 ↓、人类相关 ↑、可分辨 gap 缩小 ×N。
- **学术贡献**：运动是每个 T2V benchmark 都要评的维度（VBench/EvalCrafter/AIGV-Assessor），而现有运动度量是启发式——给它一个"有人类监督、带不确定性协议"的继任者；同时 VT↔LPF ρ≈0 的人齐问题顺带被修。
- **训练**：实在（编码器+回归头，监督学习，1 GPU 可训）。
- **查重**（novelty_k5.jsonl）：VideoScore(198 引)做整体细粒度评分、VM-Bench 做感知对齐运动生成、Video-Bench 人齐 benchmark——**"专攻运动质量、以 seed 稳定性为一等评估协议的学习度量"未见**；"Objective QA of AIGC videos with transformation consistency"(2026) 最近但角度不同。
- **风险**：人类监督数据量（2,392 分/150 视频）偏小 → 需公开 VQA 数据增广或代理预训练；与 VideoScore 系的差异要讲清（专维度 vs 全维度大模型）。

## K2（备选）：自适应评估算力分配（test-time scaling for evaluation）
- 训练每 prompt 的"需要几个 seed"预测器（LOO-CV），热点框架（LLM 侧 2026 已有 3 篇），T2V 评估侧空白（novelty_k2.jsonl）。
- 贡献面：效率+严谨，但体量偏小——适合作 K5 论文的一节，不适合作主线。

## 结论：主线换为 **K5**，K2 降为其一节；J3 数据集与 seed-resolved 协议仍为资源层；J2 保留为"被替换者"的基线。
**新标题方向**："Learning Motion Quality for Text-to-Video Evaluation: Human-Supervised, Seed-Stable, and What the Binary Heuristic Misses"
