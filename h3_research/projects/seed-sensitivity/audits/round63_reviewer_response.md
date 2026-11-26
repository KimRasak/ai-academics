# Round 63：第三方审稿（subagent）与修改落实

## 审稿结论（adversarial reviewer, fresh context）

**Borderline**。3 个 must-fix + 8 个 soundness 点 + 6 个 presentation 点。关键意见：

- S1 头条数字混用单位（per-prompt SD vs 均值 gap）→ 应以 paired-SE（±0.036, 5.5×）为准
- S3 ICC 未说明估计量、无 CI
- S4 flip rate 未做边际归一/无 CI（但 Wilson CI 不重叠 → 跨模型差异成立）
- S5 human 节自相矛盾（"无推断"却给 video-level CI 且排除 0）；LPF 未定义
- S6 "bit-level"过强 → "numerical (~1e-6)"
- S7 两模型泛化 + TeaCache 未声明；S8 分辨率分析同方差假设未声明

## 落实（全部完成）

1. **统计重算**：
   - Wilson 95% CI：H3 flip [24.9,36.5]%、Wan [11.8,21.0]%（不重叠）；any-flip [38.6,81.5]/[23.1,66.8]
   - ICC 改为 balanced one-way ANOVA variance components + prompt-level bootstrap（B=2000）：H3 0.56 [0.19,0.78]、Wan 0.48 [0.04,0.71]（`data/wan_cross/icc_anova_bootstrap.json`）
   - human join 改 **cluster bootstrap（150 prompt topics）**：ρ=−0.0423，CI **[−0.088,+0.005]**（含 0 → "indistinguishable from zero"，与 descriptive 立场一致且更诚实）（`data/scivbench/e1_cluster_bootstrap.json`）
2. **正文修改**（`paper/main.tex`）：
   - Abstract/E3/E4 头条重锚定到 paired-SE；flip rate 加 CI；ICC 表更新为 ANOVA+CI+bootstrap 说明；human 节重写（定义 LPF 为 released expert dimension、cluster CI、去除因果措辞）
   - "bit-level"→"numerical (~10⁻⁶)"；Limitations 增补 two-models-only、TeaCache、homoscedastic 假设
   - 图引用切换为 PDF；stale 数字（0.0049/0.007/0.505/0.670）清零，结构校验通过
3. 遗留（下轮）：fig1 参考线说明文字仍写 "released adjacent-model gap"（已是 median，表述微调即可）；P4 术语首次定义（LPF/ICC）可在 Setup 补一句。

## 教训

human join 的 video-level bootstrap 曾给出"排除 0"的 CI（−0.0423 [−0.082,−0.001]），换成 prompt-topic cluster bootstrap 后含 0——聚类结构在这个数据里是真实的，早先的显著性是聚类忽略的伪影。这正是审稿人 S5 抓住的点。
