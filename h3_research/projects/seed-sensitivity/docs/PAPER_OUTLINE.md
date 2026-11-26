# 论文骨架（E1–E4 汇总）——v0.1

## 目标投稿

- 首选：WVAQ @ WACV 2027（workshop，B 级以下可接受）
- 备选：ReScience C / TMLR（可验证复现类）

## 题目（工作稿）

Seed Sensitivity of Composite Video-Quality Metrics: A Reproducibility Study of MiniMax-H3 under Sci-VBench VT

## 一句话主张

在 Sci-VBench VT 协议下，复合指标 VT 继承了其最不稳定分量（Dynamic Degree 阈值化）的 seed 敏感性：单 seed 的 VT 分数与相邻模型间差距同量级甚至更大，跨模型（H3 → Wan2.1-Fun）复现表明这是 metric-pipeline 性质而非单一模型现象。

## 证据链（E1–E4，全部可复现、有审计）

### E1 human–VT join（描述性）

- 2,392/2,392 join 成功；pooled Spearman(VT, expert LPF) = −0.0423
- 结论：VT 与人类专家分在 pooled 层面基本无相关（不 claim 因果/聚类推断）
- 审计：`audits/round40_human_vt_audio.md`

### E2 released VT 复现（kill-switch gate）

- n=150，中位 |Δ|：SC 1.45e-08、BC 2.44e-05、MS 1.82e-06、DD 0.0（149/150 精确一致）、AQ 6.17e-06、IQ 1.64e-05、VT 6.15e-06
- 本地 aggregate VT=0.735706；gate PASSED
- 审计：`audits/round41_e2_reproduction.md`

### E3 H3 seed-power（Go gate）

- 16 prompts × 12 seeds（192 视频；H3/Wan/LTX 各一套，共 576）
- DD pairwise flip 73/240 = 30.42%；any-flip 10/16 = 62.5%
- within-prompt VT SD 0.051963（max 0.109205）；DD SD 0.306386
- released 相邻模型 VT gap ≈ 0.0049 → seed 噪声 ≈ 10× 模型间 gap
- k-seed 平均的 SE：0.021061(k=1)→0.016803(k=6)，多 seed 平均不能消除
- 审计：`audits/round45_e3_seed_power.md`

### E4 跨模型 replication（Wan2.1-Fun，官方协议）

- 16×6，96 唯一视频，0 失败；官方 81 frames/16FPS/50 steps/832×480（协议差异显式声明）
- DD flip 38/240 = 15.83%；any-flip 7/16 = 43.75%
- within-prompt VT SD 0.038396（max 0.081148）；DD SD 0.169217
- 方差分解：H3 ICC(prompt)=0.670；Wan ICC(prompt)=0.505（prompt/seed 各半）
- 三模型 within-prompt(seed) SD 同量级（0.051/0.041/0.040）→ metric 管线性质
- 审计：`audits/round54_e4_wan_replication.md`

## 章节结构

1. Introduction — 复合 video metrics 用于模型排行榜；单 seed 分数的普遍做法
2. Related Work — VBench/Sci-VBench、reproducibility in generative evals（用 scholar-crawler 补引文）
3. Setup — Sci-VBench VT、H3 官方权重、Wan2.1-Fun 官方入口、协议表
4. E2 Reproduction — 复现精度表（kill-switch 证据）
5. E3 Seed-power — flip/SD/SE 表 + DD 阈值机制分析
6. E4 Cross-model — 方差分解表（ICC）、协议差异声明
7. Human scores — E1 join（描述性，不做强因果）
8. Discussion — 对排行榜做法的建议（k-seed 平均 + 报告 within-prompt SD；DD 单独呈现或去掉阈值）
9. Limitations — 单 evaluator 栈、1.3B/16×6 规模、跨模型仅协议对齐时长非同配置

## 可复现资产

- 全部 runner/分析脚本：`../src/` 与 `../runners/`（`e1_human_join.py`、`analyze_e3_seed_power.py`、`analyze_e4_wan.py`、`wan/run_wan_batch.py` 等）
- 远程数据：/ssd2/jinzili/{scivbench_*, e3_*, wan_batch, wan_vt_batch2}
- 每步审计：`audits/round40–54`

## 待办

- [x] Related Work 种子文献（scholar-crawler，78 条 → `refs/merged.jsonl`、`refs/report.md`、`refs/shortlist.md`）：锚点引文齐备
  - 单 seed 排行榜不可靠的先例：Henderson et al. 2018 "Deep RL that matters"（3,655 cites）
  - 评测套件：VBench（1,910）、VBench++、VBench-2.0、EvalCrafter、VideoPhy
  - 本基准：Sci-VBench（arXiv:2608.09873）
  - 运动感知 QA 谱系：Motion-tuned spatio-temporal QA（2009）
  - human feedback：VideoFeedback（NeurIPS 2025）
- [x] 画图：`paper/assets/fig1_within_prompt_sd.pdf`（H3/Wan 各 prompt 的 12-seed VT SD vs median gap 0.0066 参考线）、`paper/assets/fig2_dd_flip.pdf`（flip 率 27.46%/17.23%/11.93% + per-prompt DD 散点）；H3 数据用 `src/analyze_e3_seed_power.py` 的 canonical join（e3_new_vt + pilot_full→vt_full），与审计数字一致（脚本输出 30.42%/15.83% 校验通过）
- [x] Tab1 复现精度表（写入正文 Sec. E2 文字内联）、Tab2 方差分解表（`paper/main.tex` Table 1, ICC）
- [x] 8 页正文初稿 v0.1：`paper/main.tex`（abstract/intro/related/setup/E2–E4/human/discussion/limitations + 2 图 1 表 9 引文）；本地无 TeX 工具链，结构校验通过（环境配对 + 图路径存在），可在 Overleaf 编译

---

**v0.5.2 状态（Round 81）**：四轮审查（Borderline→Weak Accept→一致性→Weak Accept 维持，S7 关闭）。实验 3 模型 × 16 × 12 = 576 视频全评分。待办：CFP/模板排版、编译验证、co-author。
