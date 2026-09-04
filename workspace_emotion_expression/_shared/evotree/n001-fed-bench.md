---
node_id: n001-fed-bench
title: "FED-Bench: A Cross-Granular Benchmark for Disentangled Evaluation of Facial Expression Editing"
venue: preprint
year: 2026
arxiv_id: 2603.29697
repo: ""   # 待核实（论文 abstract 称 benchmark 和 code 将公开发布，但 repo URL 未在 abstract 给出）
method_category: benchmark
modality: 2D-image
claims_verified: [P2]   # FED-Score 的 Alignment 维度评估 expression 跟随（与 P2 可区分性相关）；P1/P3/P4 未独立验证
evidence_level_self: L2   # 一条被引论断（venue=preprint）+ 引用经人工 WebFetch 核实
one_liner: 747 triplet 的 2D 表情编辑 benchmark；FED-Score 评 Alignment/Fidelity(含 identity preservation)/Relative Expression Gain；评了 18 个编辑模型；未触及风格轴×arousal 正交性。
citations:
  - https://arxiv.org/abs/2603.29697
notes: |
  builder 2026-07-27 WebFetch 实测：
  - abstract 确认 747 triplets（原图/编辑指令/GT）、cascaded scalable pipeline、20k+ 训练集、18 个编辑模型。
  - FED-Score 三维：Alignment（指令跟随）、Fidelity（图像质量+身份保持）、Relative Expression Gain（表情变化幅度）。
  - **venue**：abstract 与 arxiv listing 均无 peer-reviewed 录用信息；submitted 31 Mar 2026；按 rigor §2.0 C2，preprint 不据以关闭白色空间。
  - **identity-preservation 指标**：abstract 仅说 Fidelity 维度"covers image quality and identity preservation"，未点名具体人脸识别模型/阈值——需取全文 PDF 核实（待 S2 全文核）。
  - **arousal 正交性**：abstract 无"style axis""arousal""valence""orthogonality"字样；FED-Score 三维里无风格轴×强度正交检验。按 rigor §7.2 占据判定门槛第 4 条=未做 → **不占据白色空间**（即便未来被顶会录用也仅"部分占据，arousal 混淆未排除"）。
  - **配对数据**：triplets 是"原图+编辑指令+GT"，非"同身份同情绪仅变风格"配对；不满足课题 Q-006 配对需求。
  - 工具链卡点：本轮 WebSearch 返回模拟结果不可信；Semantic Scholar API HTTP 429（无 key）；仅 arxiv.org WebFetch 真实可用。

## edges

### edge-e001: n001-fed-bench -> n000-root   # n000-root = 课题白色空间根节点（待建）
- edge_type: compares
- edge_claim: FED-Bench 是最接近本课题的 2D 表情编辑 benchmark，但其 triplet 结构与三维 FED-Score 均未触及"风格轴×arousal 正交性"，故不占据课题白色空间。
- edge_evidence:
    claim: FED-Bench 的 FED-Score 仅评 Alignment/Fidelity/Relative Expression Gain，未检验风格轴与 arousal 的正交性；venue 为 preprint 未被顶会录用。
    evidence_level: L2
    citations:
      - https://arxiv.org/abs/2603.29697
    adjudicator: ""   # L3 待 rigor 跑四道对抗探针
- adjudicated: false
