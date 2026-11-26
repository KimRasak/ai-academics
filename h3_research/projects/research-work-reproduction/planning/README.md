# 规划与协议

此目录解释“为什么这样验证、证据能升级到什么程度、下一步为何做或不做”。它不覆盖
`../dashboard/` 的当前状态结论。

| 文件 | 用途 |
|---|---|
| [REPRODUCTION_BLUEPRINT.md](REPRODUCTION_BLUEPRINT.md) | C0–C5/X/U 证据等级、完整训练复现定义、benchmark 和方法谱系 |
| [REPRODUCTION_EXECUTION_MATRIX.md](REPRODUCTION_EXECUTION_MATRIX.md) | 当前优先级、每条路径的停止条件、串行安排 |
| [CURRENT_GOAL.md](CURRENT_GOAL.md) | 当前运行中的 MiniMax-H3 benchmark：冻结输入、等待 gate 和完成后的验证顺序 |
| [REPRODUCTION_FIRST_PLAN.md](REPRODUCTION_FIRST_PLAN.md) | 首轮候选、公开资源审计和 ≤8×H100-class 的训练准入逻辑 |

新实验先在此处确认问题、公开数据、对照和评测是否闭环；只有通过后，才在
`../reproductions/` 创建论文目录或运行任务。
