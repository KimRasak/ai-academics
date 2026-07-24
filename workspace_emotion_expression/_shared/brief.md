---
author: agic-research-team
stage: Survey
status: draft
updated: 2026-07-24
---

# 当前任务简报

> 协调器维护此文件。每次阶段变化或组会后更新。所有角色启动时先读这一份。

## 一句话定义

围绕 `emotion_expression/` 已有材料（情绪表达生成、171 种情绪、主成分分析），推进 AGIC 方向的科研：从文献调研出发，定位可做的研究问题。**范围限定为纯 2D 图像**（D-003）。

## 当前阶段

Survey（文献调研，方向已确认；范围于 D-003 收窄为纯 2D）

## Boss 最新指示（D-003，2026-07-24）

源自 `_boss/decisions/d1.md` 第二段：
1. **只做 2D 图片，排除 3D 话题**——FLAME/VRoid/Ready Player Me/BEAT/FaceWarehouse 等 3D 资产与 3D 表情捕获方向从首轮主数据候选剔除。
2. **聚焦近 2–3 年（约 2024–2026）AI 顶会**（CVPR/ICCV/ECCV/NeurIPS/ICML/AAAI/SIGGRAPH 等）。
3. **主线不变**：仍围绕"人物表情差分的一致性"（D-002），但范围从"3D+2D"收窄到"纯 2D 图像"。

## 前序指示（D-002，2026-07-24，仍生效）

1. 找一个好问题是首要任务；多看 AI 顶会论文，从中找好话题点。
2. 话题须与**人物表情差分的一致性**相关（同一角色、同种情绪下脸部表达差分的一致性）。
3. **暂缓实验**：先摸清有哪些研究方向、它们的 benchmark 数据集与评测指标是什么。

## 当前已确认事实

- D-002：课题主线锁定为"人物表情差分的一致性"，排除"真实人格生成/还原"主张。
- D-003：范围收窄为纯 2D 图像；3D 资产/方向剔除；检索收紧到近 2–3 年顶会。
- Q-001 已关闭（Boss 确认方向边界）；Q-002 标记 deferred（暂缓实验与试点）。
- 待确认事项 3（FaceWare 名称澄清）因 3D 被排除而**失效**，已降级关闭。
- `_shared/` 与 `_mailbox/` 仅供内部使用；面向 Boss 的内容只发布到 `_boss/briefs/`。

## 输入材料

- `emotion_expression/emotion_expression_generation.md`
- `emotion_expression/the_171_emotions.md`
- `emotion_expression/principal_component_analysis.md`
（索引见 `_shared/materials/README.md`）

## 进行中工作

- 首轮全组调研已合成（`_shared/stage_outputs/2026-07-24-survey-synthesis.md`），但其中"3D 表情捕获契合度最高"结论已被 D-003 推翻。
- D-003 收窄后启动第二轮全组讨论：纯 2D 设定下白色空间是否仍真实存在、候选 2D 方向、近 2–3 年顶会缺口、2D 数据配对瓶颈、Q-005 在 2D 下的可验证性。
- 检索标准需新增"仅 2D 图像""近 2–3 年顶会"两条约束并重新核实白色空间。
- 方向×benchmark×指标矩阵需剔除 3D 行（BEAT/FLAME/MEAD/DISFA 等）或标"已排除（D-003）"。

## 阻塞与依赖

- 实验与 baseline 复现暂缓（D-002）；待 Survey 产出后由 Boss 决定是否重启 Q-002。
- Q-003（首轮可公开资产）manifest 剔除 3D 资产分支，聚焦 2D 图像数据集与合成管线；主流真实人脸表情 benchmark（AffectNet/RAF-DB/CelebA/FFHQ 等）许可审计仍是重点。
- **新瓶颈**：剔除 3D 后，"自建可控资产"失去 3D 渲染这条可控来源，2D 下如何获得"同一角色、同一情绪、仅变表达风格"的配对数据成为关键依赖（critic 重点审）。
- 课题新颖性押在 Q-005（风格轴与 arousal 解耦）；在纯 2D 设定下该假设是否仍成立、预试验如何用 2D 资产做，需 rigor/builder 重新评估。

## 关联记录

- 开放问题：`workspace_emotion_expression/_mailbox/open_questions.md`（`Q-001` 已关闭，`Q-002` deferred，`Q-003`/`Q-004` 范围已按 D-003 更新，`Q-005` 待 2D 下重评，FaceWare 事项降级关闭）
- Boss 决策：`workspace_emotion_expression/_boss/decisions/d1.md`
- Boss 小结：`workspace_emotion_expression/_boss/briefs/2026-07-24-survey-2d-narrowing.md`（最新，待确认）+ `2026-07-24-survey-team-synthesis.md` + `2026-07-24-survey-kickoff-direction-calibration.md`
- 阶段产出：`workspace_emotion_expression/_shared/stage_outputs/2026-07-24-survey-synthesis.md`、`2026-07-24-survey-retrieval-criteria.md`（均需按 D-003 增补 2D 收窄说明）
- 已确认决策：`D-001`、`D-002`、`D-003`（见 `_shared/decisions.md`）
- 下次协调检查点：Boss 对 2D 收窄小结反馈后。
