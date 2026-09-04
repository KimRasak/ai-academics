# critic 工作日志

> 倒序时间线（最新在最上）。Append-only，低密度：每次工作结束追加一行。格式：`日期 | 阶段 | 做了什么 | 产出指向`。

| 日期 | 阶段 | 做了什么 | 产出 |
|---|---|---|---|
| 2026-08-03 | 方向评估组会 | 对 8 方向逐个做 pre-mortem/审稿预判/合规快筛（4 唯一 P0，2/5 DNP，3 并入 4）+ 默认假设审（双造须异构否则假双造、默认须标 provisional 待 Boss）+ 优先排序与合成预期差分 | 返回消息给协调器 |
| 2026-07-24 | Survey | 对"表情差分一致性"主线做 pre-mortem（5 失败机制）；审计主流表情 benchmark 许可（结论：可公开再发布者≈0，仅合成/3D 可走）；列 6 条评测捷径；提 AU 级收窄与先判别后生成两条替代路径 | notes/survey-pre-mortem-and-benchmark-audit.md；distilled/survey-risk-register.md |
| 2026-07-24 | Survey（2D 收窄第二轮） | 审 Q-006 配对数据四路径(编辑生成最危险/循环论证)；3D 剔除后风险重评(可控性/可证伪性受损,新增 F6 构造循环性+F7 协变)；6 捷径 2D 变化+新增 S7/S8/S9；白色空间判断(高不确定性,WebSearch 未核验)；B2 上调为 2D 首选；Q-005 更易污染加固建议 | notes/d003-2d-narrowing-risk-reeval.md；distilled/d003-2d-risk-reeval.md |
| 2026-07-27 | 组会(meta 主题) | 抽象到通用调研层面做方法论 pre-mortem：列 8 个调研失败模式(F-S1..F-S8)+早期信号+规避；定义"真跑通"三档验收；过度依赖 LLM 调研工具的退回手工判据；4 类假资产；选题空泛根因(命题未操作化)+强制收敛三步；现有演化树+workspace+graveyard 体系 6 个薄弱点；替代路(单篇反向扩展/反例搜索/可证伪命题驱动)。结论：调研 SOP 必含 9 条防呆机制清单 | 本次组会讨论产出，未落盘公共区 |
| 2026-07-27 | Survey（组会议题审） | 审 7 候选动作优先级与依赖：判定 1/2/3/6/7 为"调研变目的本身"元风险高发动作(在 Boss 拍板前做抽象/链路/固化均制造可返工中间品)；判定 5(A+B+C 试点)是唯一能改变 Boss 决策的动作但须 Boss 确认主线后再启；判定"等 Boss 拍板"应列为显式阻塞挂 kanban blocked。补 graveyard 三条(AI-Scientist/RD-Agent/karpathy-autoresearch，阶段错配否决，复活条件为阶段触发事件而非永久封印) | 本次组会产出；graveyard/README.md 已落盘 |
