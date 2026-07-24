---
author: agic-research-team
stage: Survey
status: needs-confirmation
updated: 2026-07-24
meeting: 全组第二轮并行讨论（visionary/rigor/builder/critic/voice）
trigger: Boss `decisions/d1.md` 第二段（D-003：纯 2D、近 2–3 年顶会、表情差分）
---

# Survey 第二轮小结：收窄到纯 2D 后的白色空间重核与数据瓶颈

## 一句话结论

剔除 3D 后，"人物表情差分一致性"的白色空间**从"方法空白"收窄为"评测/度量空白"**——近 2–3 年 2D 身份保持表情编辑（LivePortrait/EMO/IP-Adapter 类）已占据"能控制表情且保身份"的方法层，但 **P4（身份保持+情绪保持下的风格差分一致性）仍无人独立检验**；课题若仍主张"可控+一致"将与这些工作正面撞车，团队建议把主线形态转向"判别/度量优先"，生成降为后续。同时，2D 下"配对数据全是构造的"，构造过程蕴含课题命题，成为剔除 3D 后最隐蔽的新风险。

## 本次范围与证据边界

本轮是 Survey 调研合成，**不是实验报告**。团队未跑实验、未复现 baseline、未下载任何资产。

- **可作事实陈述**：D-003 已落档；3D 资产/方向已剔除；检索标准新增"仅 2D（C1）""近 2–3 年顶会（C2）"两条硬约束；φ 限定到 2D 表征后 Δ 定义与 P1/P2/P4 可证伪性已重写；Q-006（2D 配对数据瓶颈）四路径已审计；FaceWare 事项因 3D 排除已关闭。
- **仍是假设/判断（不应表述为定论）**：
  - "白色空间在评测/指标层仍真实、方法层已被占据"——本轮 WebSearch 工具未返回可核验论文引用，判断依赖问题内在逻辑 + builder 经 arxiv API 取得的预印本线索（FED-Bench/PixelSmile/RegGAN 等，**venue 均未核实**）。
  - "P4 从未被独立检验"是基于 arxiv 摘要印象的战略判断，非系统检索结论。
  - 风格轴与 arousal 解耦（Q-005）在 2D 下仍可证伪但更易被污染，是**待验证假设**。
  - arxiv 预印本的存在性、归属、是否纯 2D（含 ICEF/FG-Portrait 等 borderline 用 3D 表征的 2D 工作）均待 rigor+builder 逐篇核实。

## 进展（3）

1. **白色空间重核：从"方法空白"收窄为"评测/度量空白"**。剔除 3D 后，2D 身份保持表情编辑赛道比 3D 更拥挤（MagicFace/InstaFace/RigFace/Face2Diffusion/DiffusionRig/LivePortrait/EMO/IP-Adapter 等 2024–2026 密集涌现），但它们全押 P2/P3，无人系统检验 P4；最接近的评测资源 FED-Bench（747 triplets，评 18 个编辑模型的 alignment/fidelity/identity）与 FFE-Bench 仍落在"编辑指令/连续强度"框架，未触及"同一身份×同一情绪×仅变风格"的配对与风格一致性度量。结论：课题形态应从"又一个编辑方法"转向"差分一致性度量 + 受控配对协议 + 风格轴 vs arousal 解耦检验"。详见 [stage_outputs/2026-07-24-survey-synthesis.md](../../_shared/stage_outputs/2026-07-24-survey-synthesis.md)（即将增补 2D 收窄段）与各角色 distilled。
2. **2D 下的形式化边界与检索标准已更新**。rigor 把 φ 限定到 2D 表征（AU 向量 / 表情通道 2D 嵌入），3DMM/3D 表情系数路径作废；检索标准新增 C1（仅 2D）、C2（2024–2026 顶会正式录用，preprint 不据以关闭白色空间）+ §7 白色空间再核实协议 + §8 φ 在 2D 下的边界与补救（固定 P+残差化、φ_2Demb 须过"变身份保 Δ 方向"测试、双 φ 交叉验证）。Q-005 在 2D 下仍可证伪，但预试验资产须用 2D 配对，且 arousal 从"通道系数"变"回归器输出"，固定 arousal 只能取残差，可信度下降一档。见 [stage_outputs/2026-07-24-survey-retrieval-criteria.md](../../_shared/stage_outputs/2026-07-24-survey-retrieval-criteria.md)。
3. **2D 配对数据瓶颈（Q-006）四路径已审计 + 风险重评**。critic 审出 2D 下"配对全是构造的"，构造过程蕴含课题命题：路径 4（编辑生成，循环论证，最危险）> 路径 3（自标注，arousal 在标注期注入）≈ 路径 1（2D 合成，无正交通道）> 路径 2（已有编辑数据集，姿态协变不可消）。新增 F6（数据构造循环性，2D 专属最隐蔽）、F7（姿态/背景/光照协变失控）、S7（编辑模型 identity leakage）、S8（φ-选择自由度）、S9（姿态/背景搭便车，masking check 升为强制）。critic 据此建议把 B2（先判别后生成）从替代方案**上调为 2D 首选主线候选**，B1（AU 级收窄）设为兜底——判别路径绕开配对四路径全部循环性，且契合"暂缓实验、先找好问题"。

## 待确认事项（3，需 Boss 拍板）

### 1. 首轮主线形态：转向"判别/度量优先（B2）"还是仍含"生成控制"？
- **需要确认什么**：2D 下方法层白色空间很可能已被身份保持表情编辑工作占据，生成路径又受 F6 循环性污染，在 Survey 阶段无法干净论证。是否把主线从"可控+识别+一致"三任务收窄为"判别/度量优先"——先证"差分可被稳定识别与度量"是好问题，生成控制降为后续？
- **选项与团队推荐**：(a) B2 先判别后生成（critic 强烈推荐为 2D 首选；visionary/voice 倾向 benchmark+metric+受控配对，与 B2 方向一致；契合 Boss"暂缓实验、先找好问题"）；(b) 仍保持三任务并进（主线原貌，但 2D 下生成主张风险显著高于 3D 设定）。**团队推荐 (a)**。
- **为何现在确认**：范围决定检索重点、自建数据集规模与 Q-002 重启门槛；2D 下越晚改，F6 循环性返工成本越高。
- **确认后影响**：选 (a) 则首轮不碰生成器，主线为"差分可识别性基准 + 一致性指标组 + 受控配对协议"；选 (b) 则需在 Q-002 重启时同步准备生成实验并接受 F6 风险。

### 2. 2D 配对数据原则：是否采用"多构造路径一致性（方案 C）"作为硬约束？
- **需要确认什么**：Q-006 四路径均有循环性或协变问题，单一配对来源不可信。是否强制要求配对数据跨 ≥2 条独立构造路径（如 2D 合成 + 自标注，或 合成 + 已有编辑数据集），结论一致才采信，且 manifest 每条路径标注"构造器"（哪个编辑模型/哪个标注协议）？
- **选项与团队推荐**：(a) 采纳方案 C（critic 提出，对冲 F6/S7；劣势：成本翻倍，两路径可能共享隐含假设）；(b) 单路径 + 事后多验证器交叉（成本低，但无法对冲构造期循环性）。**团队推荐 (a)**，至少在 Q-005 预试验阶段强制。
- **为何现在确认**：决定 Q-003 资产 manifest 的形态与 Q-005 预试验工作量；直接影响后续可公开审计承诺。
- **确认后影响**：选 (a) 则 builder 在 manifest 中为每条配对路径标注构造器，Q-005 预试验须跨 ≥2 路径；选 (b) 则接受单路径局限并在论文中明确标注证据边界。

### 3. Q-005 预试验前置门槛 + 兜底切换：2D 加固版是否设为 Q-002 重启硬前置？
- **需要确认什么**：风格轴与 arousal 解耦在 2D 下更易被污染（S2+S4 双重）。是否把"2D 版风格轴存在性预试验（≥2 异质 arousal 回归器取一致性 + ≥2 独立配对构造路径 + 预登记 κ 阈值并报 CI + masking check 强制）"设为 Q-002 重启的硬前置，且任一路径跌破 κ 阈值即切 B1（AU 级收窄）？
- **选项与团队推荐**：(a) 设为硬前置 + B1 兜底（rigor/critic/builder 一致推荐；2D 下触发概率高于 3D）；(b) 不设前置，直接进入生成实验（不推荐，坍缩则全部返工）。**团队推荐 (a)**。
- **为何现在确认**：若不前置，团队可能在风格轴尚未证成立时进入实验，事后发现与 arousal 不可分则全部返工；2D 下该风险更高。
- **确认后影响**：选 (a) 则 Q-002 在预试验通过前不重启，预试验失败则切 B1；选 (b) 则接受坍缩返工风险。

## 风险（1）

**2D 数据构造循环性（F6）**——剔除 3D 后，2D 下"同一角色、同一情绪、仅变表达风格"的配对数据没有任何自然来源，全部由编辑模型/标注者/合成管线构造，而构造过程的隐含假设（编辑模型保身份、标注者能区分风格、提示词只改表情）正是课题要证的命题。这使得"身份保持"与"风格差分"的测量沦为对构造器先验的测量，且与 arousal 坍缩（Q-005，2D 下更难排除）叠加。这是 2D 专属、最隐蔽的失败机制，也是 critic 把 B2 上调为首选、把 masking check 升为强制的直接原因。

## 确认后的内部动作

- Boss 回复后，协调器把每项确认落为 `D-xxx`（追加 [decisions.md](../../_shared/decisions.md)），同步更新 [kanban.md](../../_shared/kanban.md) 与 [brief.md](../../_shared/brief.md)，回链关闭或追踪关联 `Q-003`/`Q-005`/`Q-006`。
- 无论 Boss 选哪项，本轮已新增 `Q-006`（2D 配对数据瓶颈）进 [open_questions.md](../../_mailbox/open_questions.md)，并把 B2/B1 作为条件切换路径记录。
- 下一轮 Survey 优先（硬依赖）：按更新后检索标准（C1/C2）由 rigor+builder **逐篇核实 2024–2026 顶会**，确认方法层白色空间是否真被 LivePortrait/EMO/IP-Adapter/FED-Bench/FFE-Bench 类工作占据、P4 是否真无人独立检验；同时收口 2D benchmark 许可"待核实"项为 manifest（剔除 3D 分支）。
- voice 据确认结果更新 pitch 与演进表；critic 把 F6/F7/S7/S8/S9 并入正式风险登记册。

## 内部追溯

- 全组产出合成：`workspace_emotion_expression/_shared/stage_outputs/2026-07-24-survey-synthesis.md`（待增补 2D 收窄段）
- 检索标准/主张边界/Q-004 操作化/§7 白色空间再核实/§8 φ 在 2D 下边界：`workspace_emotion_expression/_shared/stage_outputs/2026-07-24-survey-retrieval-criteria.md`
- 术语定义：`workspace_emotion_expression/_shared/glossary.md`
- 各角色精华：`visionary/distilled/2d_narrowing_strategic_reassessment.md`、`rigor/distilled/expression-difference-consistency.md`、`builder/distilled/2d_expression_diff_arxiv_finds.md`、`critic/distilled/d003-2d-risk-reeval.md`、`voice/distilled/method_evolution_table.md`
- 开放问题：`workspace_emotion_expression/_mailbox/open_questions.md`（Q-001 关闭，Q-002 deferred，Q-003/Q-005 范围已按 D-003 更新，Q-006 新增）
- 已确认决策：D-001、D-002、D-003（见 `_shared/decisions.md`）
