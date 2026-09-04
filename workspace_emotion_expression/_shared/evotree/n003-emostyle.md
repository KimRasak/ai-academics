---
node_id: n003-emostyle
title: EmoStyle: One-Shot Facial Expression Editing Using Continuous Emotion Parameters
venue: WACV（IEEE Winter Conference on Applications of Computer Vision）
year: 2024
arxiv_id:
repo:
method_category: editor
modality: 2D-image
claims_verified: [P2, P3]
evidence_level_self: L2
one_liner: 用连续情绪参数（摘要含 arousal/valence）直接驱动单次表情编辑——"强度轴即风格轴"混淆的典型代表；未报告固定强度后残差风格判别
citations:
  - https://doi.org/10.1109/wacv57701.2024.00626
notes: |
  OpenAlex/WACV 摘要级核实（L2）。摘要出现 arousal/valence/style 词——其编辑轴直接用连续情绪参数，恰是 Q-005 要检验的混淆（风格=强度）在现有方法层的默认假设。未做 VA 残差风格判别，无配对数据集。它是主线判别式最值得评估的 baseline（方法层部分占据，不在白空间中）。

## edges

### edge-e003: n003-emostyle -> n001-fed-bench
- edge_type: compares
- edge_claim: EmoStyle 这类"连续情绪参数编辑"恰是 FED-Bench 强度轴（Relative Expression Gain）覆盖的方法；两者共构"强度-风格混淆带"，nobody 做正交性
- edge_evidence:
    claim: EmoStyle 以连续情绪参数为编辑控制（摘要），FED-Bench 评估相对强度增益维度，两者均无"残差风格可区分"检验
    evidence_level: L2
    citations:
      - https://doi.org/10.1109/wacv57701.2024.00626
      - https://arxiv.org/abs/2603.29697
    adjudicator: coordinator@2026-08-03
- adjudicated: false