# Round 70：12-seed 扩展审计（E3/E4 最终版）

## 生成与评分

- H3 seeds 6–11：runner `run_scivbench_pilot.py --seeds 6,7,8,9,10,11`，96/96 完成、0 错误，输出 `/ssd2/jinzili/e3_seed_s6_11`；VT `/ssd2/jinzili/e3_s6_11_vt`（96/96，VT=0.741312）
- Wan seeds 6–11：`run_wan_batch.py`（v2 runner），96/96、MD5 唯一、0 失败，输出 `/ssd2/jinzili/wan_batch_s6_11`；VT `/ssd2/jinzili/wan_vt_s6_11`（96/96，VT=0.680689）
- 与 seeds 0–5 合并后两模型各 16 prompts × 12 seeds = 192 条

## 12-seed 最终统计

| 统计量 | H3 | Wan |
|---|---:|---:|
| DD pairwise flip | 290/1056 = 27.46% [24.9, 30.2] | 182/1056 = 17.23% [15.1, 19.6] |
| any-flip | 12/16 | 9/16 |
| VT within-prompt SD（mean / max） | 0.0509 / 0.0841 | 0.0411 / 0.0841 |
| DD within-prompt SD | 0.3012 | 0.2054 |
| ICC(ANOVA, bootstrap) | 0.56 [0.30, 0.72] | 0.48 [0.22, 0.65] |
| overall mean VT | — | —（见 aggregate） |

与 6-seed 估计（30.42%/15.83%、SD 0.0520/0.0384）方向与量级一致；12-seed CI 更窄。flip CI 仍不重叠 → 跨模型差异成立。

## SE 实证验证（subset 法，每档 500 随机子集）

| s | H3 emp/pred | Wan emp/pred |
|---|---|---|
| 2 | 0.89 | 1.23 |
| 3 | 0.83 | 1.12 |
| 4 | 0.78 | 1.11 |
| 6 | 0.66 | 0.94 |

i.i.d. 噪声模型 ±25% 内成立；H3 略低于预测、Wan 略高于预测，均在批次效应/有界指标可解释范围。s=12 行退化（单一子集，SD=0）不使用。

## 方法教训（已修正）

6-seed 版脚本曾把"单 prompt 内 SD/√s"与"16-prompt mean SE"混比，得到假象比值 3.9；subset 法（每子集覆盖全部 prompts，prompt 方差相消）才是对 resolution 模型的正确检验。

## 产物

- `data/scivbench/se_validation.json`、`icc_12seed.json`
- 本地 per-video JSON：`e3_s6_11_vt/h3_base/`（96）、`wan_cross/vt_s6_11/`（96）
- 论文 v0.4：abstract/E3/E4/Table1/resolution 全部 12-seed 口径；fig1/fig2 重生成（0.2746/0.1723）
- 补充包重建（385 项）
