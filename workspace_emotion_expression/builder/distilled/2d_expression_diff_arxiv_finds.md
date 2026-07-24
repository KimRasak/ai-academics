---
author: builder
stage: Survey
status: draft
updated: 2026-07-24
---

# 2D 表情差分一致性 — 近期 arxiv 检索实据

> 来源：arxiv API (`export.arxiv.org/api/query`)，2026-07-24 检索。
> 仅 arxiv 预印本，**venue 均待核实**（未确认是否被 CVPR/ICCV/ECCV/NeurIPS/ICML/AAAI/SIGGRAPH 接收）。
> arxiv ID 前缀 YYMM：2602–2607 = 2026 年 2–7 月。

## 最接近本课题的 2D 工作（按相关度）

1. **FED-Bench** (arXiv:2603.29697) — "A Cross-Granular Benchmark for Disentangled Evaluation of Facial Expression Editing"。747 triplets（原图 / 编辑指令 / GT），FED-Score 评 Alignment/Fidelity(含 identity preservation)/Relative Expression Gain；评了 18 个编辑模型；附 20k+ 训练集。**最接近的 benchmark**，但面向"编辑指令→图"，无"风格轴×身份×情绪"配对，无风格一致性度量。
2. **PixelSmile / FFE-Bench** (arXiv:2603.25728) — 细粒度表情编辑，FFE 数据集含连续 affective 标注；diffusion + intensity supervision + contrastive learning；主打 disentanglement + identity preservation。连续强度轴而非风格轴。
3. **RegGAN** (arXiv:2603.15648) — GAN 表情合成泛化，含 identity preservation 人工评（26% 提升）。方法旧但 identity 度量可参考。
4. **ICEF** (arXiv:2607.16287) — NeRF few-shot 表情合成，identity-expression 解耦。**borderline**：NeRF 属 2D 图像合成但用 3D 表征，D-003 下需 critic 拍是否算 3D。
5. **MindAU / E-CAFE** (arXiv:2607.00410) — EEG-conditioned AU 编辑。AU 级，数据源偏门。
6. **FG-Portrait** (arXiv:2603.23381) — 3D flow 引导的可编辑 portrait animation。**borderline**（用 parametric 3D head）。

## 检索结论（白色空间）

- 未见任何论文提供"同一身份、同一情绪、仅变表达风格"的配对数据 + 风格标签 + 风格一致性度量。
- FED-Bench / FFE-Bench 是最接近的评测资源，但都落在"编辑指令/连续强度"框架，未触及 P1(可重复)/P2(风格可区分)/P4(身份+情绪双保持) 的联合检验。
- 支持"白色空间真实"判断，但需 rigor 逐篇核实是否占据了 P1/P2/P4 任一条。
- "expression style 与 expression entangled"在 SIGGRAPH ASIA 2023 (arXiv:2401.15414，3D，已排除) 中被明确承认为开放问题——间接支持课题动机。

## 不确定性

- 上述全部为 arxiv 预印本，**未被顶会官网核实**；WebSearch 工具本轮返回的是模拟结果（不可信），仅 arxiv API 实据可用。
- 训练数据中的 LivePortrait / Face0 / InstantID / PhotoMaker / PuLID / ConsistentID 等命名在本轮未能经 arxiv API 复核，仅作线索，不作已验证事实。
