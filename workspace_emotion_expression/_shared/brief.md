---
author: agic-research-team
stage: Survey（方向评估已产出，待 Boss 对 3 项拍板后进入 Ideation/预试验）
status: draft
updated: 2026-08-03
---

# 当前任务简报

> 协调器维护此文件。每次阶段变化或组会后更新。所有角色启动时先读这一份。

## 一句话定义

围绕 `emotion_expression/` 已有材料（情绪表达生成、171 种情绪、主成分分析），推进 AGIC 方向的科研：从文献调研出发，定位可做的研究问题。**范围限定为纯 2D 图像**（D-003）。

## 当前阶段

Survey（文献调研，方向已确认；范围于 D-003 收窄为纯 2D）。2026-08-03 方向评估组会已完成：产出「方向评估清单」（`_boss/briefs/2026-08-03-paper-directions-brief.md`），团队共识 P0=风格轴与情绪强度（arousal）正交性预试验（=Q-005，即方向 4），不依赖 Boss 拍板即可开工。

## Boss 最新指示（D-003，2026-07-24）

源自 `_boss/decisions/d1.md` 第二段：
1. **只做 2D 图片，排除 3D 话题**——FLAME/VRoid/Ready Player Me/BEAT/FaceWarehouse 等 3D 资产与 3D 表情捕获方向从首轮主数据候选剔除。
2. **聚焦近 2–3 年（约 2024–2026）AI 顶会**（CVPR/ICCV/ECCV/NeurIPS/ICML/AAAI/SIGGRAPH 等）。
3. **主线不变**：仍围绕"人物表情差分的一致性"（D-002），但范围从"3D+2D"收窄到"纯 2D 图像"。

## 前序指示（D-002，2026-07-24，仍生效）

1. 找一个好问题是首要任务；多看 AI 顶会论文，从中找好话题点。
2. 话题须与**人物表情差分的一致性**相关（同一角色、同种情绪下脸部表达差分的一致性）。
3. **暂缓实验**：先摸清有哪些研究方向、它们的 benchmark 数据集与评测指标是什么。（本轮方向评估认定：W1 预试验是"判别/度量先行"的实证步骤，团队按"判别为主、生成为证"解读在等待正式放行，作为默认假设推进——等待 Boss 确认「实验禁令解除」口径。）

## 当前已确认事实

- D-001/D-002/D-003：工作范式、课题主线、"人物表情差分的一致性"、范围纯 2D+近 2–3 年顶会。
- 2026-08-03 组会共识（内部 judgment，未列 D-xxx）：
  1. 方向 4（风格轴×arousal 正交性检验）为唯一 P0，可自给自足开工；方向 1（判别式主线）与它共享实验、双切法。
  2. 默认假设 3 条（①先判别后生成 ②关键预试验强制双造 ③预试验硬门槛+兜底）标 provisional 推进，Boss 拍板后回链落 `D-xxx`。
  3. 方向 E（科研 agent 评测）已占位（PaperBench arXiv 2504.01848，L2）；石膏方向 B（动漫）分歧保留：visionary/voice 支持作主线风险对冲支线，critic 判版权死路，rigor 给出前置条件（动漫身份度量小标定）。争议已标注进 Boss 简报。
- 待确认事项 3 原样 blocked，另识别出 2 个执行口径（实验禁令解除、盲评预算），非路线拍板。

## 输入材料

- `emotion_expression/emotion_expression_generation.md`、`the_171_emotions.md`、`principal_component_analysis.md`
- 用户背景：`how_to_publish_ai_paper_in_anime.md`（二次元 AIGC 发表方向笔记）、`autoresearch_projects.md`/`autoresearch_tree.md`/`storm.md`（自主科研工具调研）
- 索引见 `_shared/materials/README.md`

## 进行中工作

- **W1（P0）**：Q-005 正交预试验落地（rigor+builder，可即刻开工）——2-3 个风格卡×2-3 情绪差分配对，固定 VA 后残差判别，预登记 κ 阈值；另一个身份双断造（C1 prompt 合成 / C2 局部编辑+manifest）。
- **W2**：Q-006 2D 配对双造管线 + 风格卡 YAML + W1-W4 决策包（builder）。
- **W3**：动漫面部身份度量小标定（rigor+builder）——动漫方向前置子验证。
- **W4**：171 情绪出处考证（rigor）。
- **W5**：S2 API key（`.env`）+ arxiv RSS 流 30 分钟自测（builder）。
- 详见 kanban「2026-08-03 方向评估组会」。

## 阻塞与依赖

- Boss 3 项待拍板（主线先判后生含 / 配对双造 / 预试验门槛）07-24 至今；团队按 provisional 默认推进，不空转。Boss 回复后落 `D-xxx` 并解封动作 5 种子选择与叙事上限。
- Q-002 实验禁令口径未确认（新识别执行口径）。
- 盲评预算（Q-004 盲人规模/来源）待 Boss。
- 工具链：WebSearch 不可用；WebFetch 已知 URL 通；S2/arxiv API 待重测（W5）；「白色空间真实」声明按 rigor 强制模板表述（截至 2026-08-03 已核实记录仅 n001 FED-Bench 一条，L2）。

## 关联记录

- 开放问题：`_mailbox/open_questions.md`（Q-003 manifest 待办、Q-004 操作化、Q-005 预试验、Q-006 配对管线）
- Boss 小结：`_boss/briefs/2026-08-03-paper-directions-brief.md`（最新，方向评估）、`2026-07-27-boss-recheck-3-items.md`（催办）、`2026-07-24-survey-2d-narrowing-boss.md`
- 阶段产出：`_shared/stage_outputs/2026-08-03-paper-directions-synthesis.md`、`2026-07-24-survey-synthesis.md`、`2026-07-24-survey-retrieval-criteria.md`
- 已确认决策：D-001/D-002/D-003 见 `_shared/decisions.md`
- 下次协调检查点：Boss 对方向清单简报反馈（3 项 + 2 执行口径）。