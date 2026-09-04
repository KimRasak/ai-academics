---
node_id: n002-pixelsmile-ffebench
title: PixelSmile: Toward Fine-Grained Facial Expression Editing（FFE-Bench）
venue: preprint
year: 2026
arxiv_id: 2603.25728
repo:
method_category: benchmark
modality: 2D-image
claims_verified: [P2]
evidence_level_self: L2
one_liner: 细粒度表情编辑 + FFE 数据集（连续情感标注）与 FFE-Bench（结构混淆/编辑精度/线性可控性）；强度轴连续可控但无风格轴×arousal 正交检验
citations:
  - https://arxiv.org/abs/2603.25728
notes: |
  2026-03 预印本，未正式录用。摘要级核实：continuous affective annotations（连续情感标注）、linear controllability（线性可控性）——它们评的是"编辑对强度的线性响应"，不是"固定强度后风格可分"。不提供同身份同情绪仅变风格配对。不占据 P4/正交白空间。

## edges

### edge-e002: n001-fed-bench <-> n002-pixelsmile-ffebench
- edge_type: compares
- edge_claim: FFE-Bench 与 FED-Bench 同为表情编辑评测，前者主打连续强度与线性可控，后者打对齐/保真/相对增益；两者都未做风格×arousal 正交
- edge_evidence:
    claim: 两个 2026 预印本均无 valence/arousal 正交检验；均未提供"同身份同情绪仅变风格"配对
    evidence_level: L2
    citations:
      - https://arxiv.org/abs/2603.25728（PixelSmile）
      - https://arxiv.org/abs/2603.29697（FED-Bench）
    adjudicator: coordinator@2026-08-03
- adjudicated: false