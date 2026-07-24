---
author: visionary
stage: Survey
status: draft
updated: 2026-07-24
---

# 纯 2D 设定下"表情差分一致性"的战略重评（D-003 后）

> 一句话主旨：剔除 3D 后，2D 的"身份保持表情编辑"赛道比 3D 更拥挤（MagicFace/InstaFace/RigFace/Face2Diffusion/DiffusionRig/LaTo/PixelSmile 等 2024-2026 密集涌现），但它们全押 P2/P3、无人系统检验 P4（身份+情绪保持下的风格差分一致性）；白色空间从"方法空白"收窄为"评测/度量空白"，战略价值仍在但课题形态须从"生成方法"转向"一致性度量+受控配对"。

## 战略判断

1. **2D 赛道更拥挤，但拥挤在错的地方**：近 2-3 年 2D 表情编辑/AU 控制/身份保持论文密集（arXiv 2025 一茬：MagicFace AU 控制、InstaFace 单图、RigFace 解耦、LaTo landmark-token、PixelSmile 细粒度、Face2Diffusion CVPR'24、PortraitBooth、DiffFAE、DiffusionRig）。它们都在回答"能不能编辑/控制表情"，几乎无人回答"同一角色同一情绪下，表达风格差分是否可稳定复现与识别"。P4（身份保持+情绪保持）仍是无人独立检验的空白。
2. **白色空间从"方法"转移到"评测与配对"**：方法层红海，但"差分一致性度量 + 受控配对协议 + 风格轴 vs arousal 解耦检验"仍是无主之地。课题应定位为 benchmark/metric + 受控研究，而非又一个编辑方法。
3. **3D 剔除的净效应**：失去 3DMM/FLAME 这条"最干净的 φ 实例化"可控来源（Q-006），但换回更大的受众面（2D 是主流）和更少的许可泥潭（3D 资产 CC BY-NC 普遍）。代价是配对数据必须靠 2D 合成/编辑管线构建，可控性下降。

## 2D 候选方向排序（visionary 视角）

- **最值得做**：表情差分一致性 benchmark + 度量（P4 独立检验）。做成了=为整个 2D 表情编辑赛道提供缺失的评测基础设施，影响面最大，且天然避开"又一个编辑方法"的 incremental 陷阱。
- 次优：AU 级差分一致性（B1）——差分有数学定义，与 arousal 解耦更干净，但受众窄、故事偏技术。
- 谨慎：character-specific expression 生成方法——赛道已拥挤，除非有强 P4 论证否则 incremental。

## 最可能失败的一句话（仍有效）

"表达风格轴与 arousal 不可解耦，差分概念坍缩回强度轴。" 2D 下该风险未降反升——2D 缺乏 3D 的干净解耦先验，更难证明风格轴独立性。Q-005 是命门。

## 不确定性声明

- arXiv 检索返回的若干论文（MindAU/Edit3DGS/PixelSmile 的 arXiv ID 形如 2603.xxxxx / 2607.xxxxx）疑似检索工具合成或 ID 不准，未逐篇打开核实；具体论文存在性与归属待 rigor/builder 系统检索确认。
- "P4 无人独立检验"是基于 arXiv 摘要印象的战略判断，非系统检索结论。
- 3DMM 作为 2D 编辑的内部条件（RigFace/InstaFace/DiffusionRig 用 3DMM 驱动 2D 输出）是否算"3D 话题"——边界模糊，需 rigor 按检索标准裁定。
