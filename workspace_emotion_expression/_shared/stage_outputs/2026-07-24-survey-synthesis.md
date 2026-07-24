# Survey 阶段团队讨论结果

> 协调器合成。触发：Boss `decisions/d1.md`（D-002）。日期 2026-07-24。
> 五视角并行调研：visionary / rigor / builder / critic / voice。
> 证据不足处标注；推测不表述为已验证事实。

## 共识

1. **白色空间真实**：课题"人物表情差分一致性"落在身份一致性生成 × 情绪表情可控生成的交叉空白；现有 benchmark 无一提供"同一角色、同一情绪、仅变表达风格"的配对与风格标签（builder 三层缺口、rigor 举证缺口、critic F3、visionary 战略判断一致）。
2. **头号威胁=arousal 混淆**：候选风格轴（克制/外放等）可能只是低↔高唤醒的再包装；若不做"回归出 arousal 后残差风格仍可区分"的正交性检验，课题新颖性坍缩（visionary 坍缩风险、rigor #2 Critical、critic S4、builder MEAD confound）。
3. **主张须拆分**：rigor 把"一致性"拆为四条可证伪子命题 P1 可重复 / P2 可区分 / P3 可控 / P4 解耦（身份保持+情绪保持）；多数论文只报 P2 默认 P4，P4 从未被独立检验——这是检索纳入门槛与评测严谨性核心。
4. **数据合规瓶颈**：主流真实人脸表情 benchmark（AffectNet/RAF-DB/CelebA/FFHQ/MEAD/DISFA…）在"可公开审计"约束下几乎全部不可再发布；唯一可行路径=合成/3D 资产（FLAME/VRoid/Ready Player Me）+ 自标注，逐项核许可（critic 审计 + builder Q-003 一致）。
5. **实验暂缓**：D-002/Q-002 deferred，本轮只产标准、判据、盘点，不启动 baseline 复现与试点。

## 分歧与冲突

**冲突 1｜首轮范围：判别/度量 vs 同时含生成控制**
- visionary 最看好方向 2（角色一致性生成）下"表达风格差分一致性"子问题，含"控制"主张。
- critic 指出 D-002 把"控制+识别+一致"写成三个任务太散，建议先收敛到"识别/度量"单一任务（方案 B2 先判别后生成），与 Boss"暂缓实验、先找好问题"一致。
- **消解**：未在本轮消解，保留为待 Boss 拍板（见 Boss 小结待确认 1）。Boss 倾向（暂缓实验）与 critic 推荐方向一致，但需 Boss 明确。

**冲突 2｜主线 vs AU 级收窄（B1）**
- critic 给替代方案 B1：收窄到 AU 级别表情差分一致性，差分有数学定义（AU 向量差）、与身份较解耦、F1 风险下降；触发判据=Q-004 风格卡 α<0.6 两次修订不闭。
- rigor 的 Q-004 操作化判据已给出"最小可验证"定义，任一轴不通过即记"不适用"。
- **消解**：这是条件切换而非立场冲突——若主线风格轴预试验失败则切 B1，本轮作为切换判据记录，不现在决断。

## 关键发现（5）

1. **白色空间真实但押注未验证前提**：方向空白存在，但课题新颖性押在"风格轴可与 arousal 解耦"这一未验证假设上；该假设是 Q-004/Q-005 关闭判据的直接依赖。
2. **三层 contribution 切入点**（builder）：风格标签集 + 差分配对协议（含遮蔽检查+情绪/身份保持双约束）+ 差分一致性指标组（P1 `Var_ξ[φ]` / P2 留置风格分类器 acc / P4 身份余弦+VA 漂移），均无公开 leaderboard 报过。
3. **"一致性"≠ ID-sim+FID**：现有评测把 ID-sim/FID 误用为一致性代理，但二者都不反映风格稳定性也不反映可分性；P4（身份保持+情绪保持）必须独立度量，不能默认。
4. **3D 表情通道是最干净的 φ 实例化**（builder/rigor）：FLAME blendshape / 3DMM 表情通道天然身份/姿态/表情解耦，差分 Δ 可直接定义；但许可（BEAT CC BY-NC、FaceWarehouse 状态不明）与"FaceWare 名称待澄清"阻碍直接用作公开主数据。
5. **六条评测捷径**（critic）：identity-expression 隐性 trade-off、FER 分类器当 ground truth 的循环性、人类盲评协议缺陷、风格被强度偷换、随机性伪装成风格、案例图 cherry-picking——均须在预试验与预注册阈值（τ_id/σ_repeat/chance+δ/κ₀）中前置防御。

## 阶段产出

- 主张边界与四子命题：`_shared/glossary.md`（expression difference / consistency / identity preservation / emotion preservation / expression style / arousal confound / masking check / style card）
- 检索纳入/排除标准 + Q-004 操作化判据 + 混淆表：`_shared/stage_outputs/2026-07-24-survey-retrieval-criteria.md`（rigor）
- 方向×benchmark×指标×契合度矩阵 + 三层缺口：`builder/distilled/`（builder）
- benchmark 许可审计表 + pre-mortem 5 条 + 6 条捷径 + 替代方案 B1/B2：`critic/distilled/survey-risk-register.md`
- 方法演进对比表 + pitch + Boss 小结框架：`voice/distilled/method_evolution_table.md`
- 战略落点与 5 候选方向：`visionary/distilled/`

## 下一步行动

| 行动 | 负责视角 | 时间估算 | 依赖 |
|---|---|---|---|
| 按检索标准系统核实"白色空间是否真无人占据"（逐篇记录验证了 P1/P2/P3/P4 哪几条） | rigor + builder | 下一轮 Survey | Boss 范围确认（待确认 1）后定检索重点 |
| 收口 benchmark 许可"待核实"项为 manifest（来源/许可证/允许用途/再发布限制/替代） | builder + critic | 下一轮 Survey | 无 |
| 风格轴存在性预试验草案（含遮蔽检查 + arousal 正交性检验 + κ 预登记） | rigor + builder | 待 Q-005/Q-004 | Boss 确认前置门槛（待确认 2） |
| 澄清 FaceWare 名称 | builder→Boss | 即刻 | Boss 回复 |
| 预试验失败则评估切 B1(AU 级) 或 B2(先判别) | critic + rigor | 预试验后 | 预试验结果 |

## 不确定性声明

- "白色空间无人占据"是战略判断，未经系统检索核实；不应在对外材料表述为定论。
- benchmark 规模与许可证凡未联网核实者均标"待核实"；不作为已验证事实。
- 风格轴与 arousal 解耦、身份/情绪保持等均为**待验证假设**，本轮无 stable 项。
