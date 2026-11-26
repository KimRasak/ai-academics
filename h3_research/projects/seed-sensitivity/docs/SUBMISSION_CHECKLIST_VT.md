# SUBMISSION_CHECKLIST_VT.md — VT seed-sensitivity 论文投稿清单

（注：`SUBMISSION_CHECKLIST.md` 是早前 EmoH3/ICASSP 轨道的清单，本文件只管当前论文。）

## 已完成

- [x] 实验：E1（human join + bootstrap CI）、E2（released 复现）、E3（H3 seed-power）、E4（Wan 跨模型）、leaderboard 间距、rank stability
- [x] 数字 QA：`paper/main.tex` 全部统计量与 artifacts 逐项核对（Round 58/60）
- [x] 图：fig1（within-prompt SD vs median gap 0.0066）、fig2（DD flip 30.42%/15.83%）、fig3（human join + per-metric ρ）
- [x] 表：Tab1（ICC 分解）；E2 精度内联正文
- [x] 引文：9 篇核心锚点 + `refs/merged.jsonl` 备选池
- [x] 复现指南：`REPRODUCE.md`
- [x] 审计留痕：`audits/round40–60`（含 v1 作废记录、off-by-one 修复记录）

## 投稿前必做

- [ ] 选定目标：WVAQ@WACV 2027（查 CFP：页数、模板、deadline）；备选 ReScience C / TMLR
- [ ] 模板排版：迁移 `paper/main.tex` 到官方 LaTeX 模板
- [ ] 匿名化：作者、致谢、路径中的用户名（`jinzili`）→ 泛化描述；补充材料同步
- [ ] 补充材料：打包 `data/wan_cross/`、`e3_seed_power.json`、`leaderboard_spacing.json`、`seed_rank_stability.json`、全部 runner 脚本 + `REPRODUCE.md`
- [ ] 图矢量化：matplotlib 存 PDF 版本（当前 PNG 150dpi，正式版需 PDF/600dpi）
- [ ] 语言润色 + co-author 通读
- [ ] Claims 复查：不声称"DD 错误"，只声称 seed 敏感性与负人类相关；E1 保持 descriptive；Wan 为 metric-behaviour replication

## 风险与应对

| 风险 | 应对 |
|---|---|
| deadline 提前 | 初稿已可编译，优先排版与匿名化 |
| 16×6 规模被质疑 | k-seed SE 曲线（0.0211→0.0168）佐证规模合理性；Limitations 已声明 |
| "只有两个模型" | E4 的价值在 metric-pipeline 性质（within-prompt SD 跨模型一致 0.047/0.049）；必要时评估第三模型 |
| DD 批评被误读为"指标无效" | 全文只讨论阈值化传播 seed 噪声 + 负人类相关，未断言 metric 无效 |

## 状态

- 距目标 deadline 约 4 周；实验与写作主体完成，进入打磨期。

- [x] 论文 v0.5.2（四轮审查）（2026-09-13）
- [x] E5 第三模型 LTX-Video 2B（2026-09-13）
- [x] 12-seed 扩展（2026-09-13）
- [x] DD 阈值不变性（2026-09-13）
- [x] 补充材料包（2026-09-13）
## ICASSP 2027（发现于 Round 84，deadline 2026-09-16）

- [x] 4+1 页 IEEEtran 版本 `paper/icassp-build/main.tex`（tectonic 编译，3 页含图）
- [x] 内容：v0.5.2 全部结果压缩（E2/E3/E4/E5/human/resolution/threshold-invariance）
- [x] 匿名（Anonymous）；图 3 张矢量复用
- [ ] **用户动作**：ICASSP 注册/OpenReview 账号、作者信息（投稿系统内填写）
- [ ] 提交前最终人工通读（co-author）
