---
author: agic-research-team
stage: Survey
status: needs-confirmation
updated: 2026-07-24
meeting: 全组并行调研（visionary/rigor/builder/critic/voice）
trigger: Boss `decisions/d1.md`（D-002）
---

# Survey 全组调研小结：差分一致性的空白与头号威胁

## 一句话结论

"人物表情差分的一致性"在版图上落在**身份一致性生成 × 情绪表情可控生成**的交叉空白，现有 benchmark 无一提供"同一角色、同一情绪、仅变表达风格"的配对与标签——白色空间真实，但课题新颖性押在一个尚未被验证的前提上：**风格轴能和情绪强度解耦**。

## 本次范围与证据边界

本轮是 Survey 调研合成，**不是实验报告**。团队未跑任何实验、未复现 baseline、未下载任何资产。以下分两类：

- **可作事实陈述**：现有 benchmark/指标清单（基于领域知识，许可证凡未联网核实者均标"待核实"）；课题主张已由 rigor 形式化为四条子命题；Boss D-002 已落地、Q-001 关闭、Q-002 deferred。
- **仍是假设/判断**："白色空间真实无人占据"是战略判断而非检索核实结论（待按检索标准系统核实）；"风格轴可与 arousal 解耦"是**待验证假设**，决定课题成立与否。

## 进展（3）

1. **方向版图与缺口定位**：voice 给出演进线（表情识别→生成→身份保持编辑→**表达差分一致性**）与对比表；builder 给出方向×benchmark×指标×契合度矩阵——结论是 3D 表情捕获（BEAT/FLAME 系）契合度最高但许可受阻，"自建可控资产 + 公开风格标签集"是首轮现实主路径。详见 [stage_outputs/2026-07-24-survey-synthesis.md](../../_shared/stage_outputs/2026-07-24-survey-synthesis.md)。
2. **主张边界与检索标准已固化**：rigor 把"一致性"拆为四条可证伪子命题（P1 可重复 / P2 可区分 / P3 可控 / P4 解耦身份+情绪），指出多数论文只报 P2、默认 P4，而 P4 从未被独立检验。检索纳入/排除标准与 Q-004 操作化判据已写入 [stage_outputs/2026-07-24-survey-retrieval-criteria.md](../../_shared/stage_outputs/2026-07-24-survey-retrieval-criteria.md) 与 [glossary.md](../../_shared/glossary.md)。
3. **数据合规结论**：critic 审计 + builder 盘点一致——主流真实人脸表情 benchmark（AffectNet/RAF-DB/CelebA/FFHQ/MEAD/DISFA…）在"可公开审计"约束下几乎全部不可再发布；唯一可行方向是合成/3D 资产（FLAME/VRoid/Ready Player Me）+ 自标注，且须逐项核许可。这把 Q-003 的资产 manifest 工作前置到 Survey 末期。

## 待确认事项（3，需 Boss 拍板）

### 1. 首轮研究范围：先做"判别/度量"还是同时含"生成控制"？
- **需要确认什么**：D-002 把课题写成"能否被稳定**控制、识别与保持一致**"——这其实是三个任务（控制=生成、识别=判别、一致=度量），一年内同时做风险大。
- **选项与团队推荐**：(a) 先收敛到"识别/度量"单一任务——先证"差分可被稳定识别与度量"是好问题，再上生成控制（critic 推荐方案 B2，与 Boss"暂缓实验、先找好问题"一致）；(b) 保持三任务并进（visionary/主线）。**团队推荐 (a)**。
- **为何现在确认**：范围决定检索重点、自建数据集规模与后续 Q-002 重启门槛；越晚改成本越高。
- **确认后影响**：选 (a) 则首轮不碰生成器，主线为"差分可识别性基准 + 一致性指标组"；选 (b) 则需在 Q-002 重启时同步准备生成实验。

### 2. "风格轴存在性预试验"是否作为重启实验的前置门槛？
- **需要确认什么**：候选风格轴（克制/外放、眉眼主导/嘴部主导、回避/直视）能否被盲评稳定识别、且与 arousal 解耦——这是课题新颖性的命门。现有 benchmark 无法回答，只能靠自建小规模预试验。
- **选项与团队推荐**：把"风格轴存在性预试验（含遮蔽检查 + arousal 正交性检验）"设为 Q-002 重启的硬前置（rigor/critic/builder 一致推荐）。**团队推荐设为前置门槛**。
- **为何现在确认**：若不前置，团队可能在风格轴尚未证成立时进入生成实验，事后发现坍缩则全部返工。
- **确认后影响**：设为前置则 Q-002 在预试验通过前不重启；预试验失败则切替代方案 B1（收窄到 AU 级差分）或 B2（先判别）。

### 3. "FaceWare" 名称澄清
- **需要确认什么**：Boss 在 d1.md 未提 FaceWare，但前置材料提及"FaceWare"类 3D 资产；builder 未能定位到名为 FaceWare 的标准公开数据集，疑似与 FaceWarehouse 混淆或指某商用 3D 捕获系统。
- **选项与团队推荐**：(a) 指 FaceWarehouse（历史项目，许可不明，不推荐作主数据）；(b) 指某商用捕获系统；(c) BEAT 系列别称。**团队请 Boss 澄清**。
- **为何现在确认**：影响 3D 数据源选型与 Q-003 manifest。
- **确认后影响**：澄清后 builder 在资产清单中定位/排除该项。

## 风险（1）

**风格轴坍缩回情绪强度轴**——若预试验显示"克制/外放型愤怒"在盲评者眼中无法与 arousal 解耦，整个"表达差分"概念就坍缩为已有的强度控制，课题新颖性归零。这是四视角一致识别的头号威胁（visionary 的坍缩风险、rigor 的 arousal 混淆 Critical、critic 的 S4 捷径、builder 的 MEAD confound），也是 Q-004 关闭判据与"切 B1"触发器的直接依赖。

## 确认后的内部动作

- Boss 回复后，协调器把每项确认落为 `D-xxx`（追加 [decisions.md](../../_shared/decisions.md)），同步更新 [kanban.md](../../_shared/kanban.md) 与 [brief.md](../../_shared/brief.md)，回链关闭或追踪关联 `Q-xxx`。
- 无论 Boss 选哪项，本轮已新增 `Q-005`（风格轴 vs arousal 正交性验证）进 [open_questions.md](../../_mailbox/open_questions.md)，作为预试验的待验证问题追踪。
- 下一轮 Survey 优先：按检索标准系统核实"白色空间是否真无人占据"+ 收口 benchmark 许可"待核实"项。

## 内部追溯

- 全组产出合成：`workspace_emotion_expression/_shared/stage_outputs/2026-07-24-survey-synthesis.md`
- 检索标准/主张边界/Q-004 操作化/混淆表：`workspace_emotion_expression/_shared/stage_outputs/2026-07-24-survey-retrieval-criteria.md`
- 术语定义草案：`workspace_emotion_expression/_shared/glossary.md`
- 各角色精华：`visionary/`、`rigor/`、`builder/`、`critic/`、`voice/` 下 distilled
- 开放问题：`workspace_emotion_expression/_mailbox/open_questions.md`（Q-001 已关闭，Q-002 deferred，Q-003/Q-004 进行中，Q-005 新增）
- 已确认决策：D-001、D-002（见 `_shared/decisions.md`）
