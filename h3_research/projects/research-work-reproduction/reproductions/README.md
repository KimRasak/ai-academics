# 已入队复现

每个子目录对应一个已完成资源审计或实际运行的工作。目录存在不代表论文已复现；
当前结论以 [展示总表](../dashboard/PAPER_REPRODUCTION_TRACKER.md) 为准。

| 目录 | 对象 | 当前作用 |
|---|---|---|
| `minimax-h3-official-768p/` | MiniMax-H3 Base | 冻结官方 768p 请求、部署失败证据与运行时审计 |
| `mm-diffusion/` | MM-Diffusion | Landscape checkpoint 指标复现与媒体审计 |
| `t2v-turbo-vc2/` | T2V-Turbo | VC2 LoRA 的 VBench checkpoint 验证 |
| `vader-videocrafter/` | VADER | 发布 LoRA 的 reward/motion effect 审计 |
| `videodpo-vc2/` | VideoDPO | 偏好数据完整性与训练门禁审计 |
| `motionctrl-videocrafter/` | MotionCtrl | checkpoint smoke 与控制协议缺口审计 |
| `vabench/` | VABench | evaluator / I2AV 物料与缺口审计 |

最低可审查结构为 `REPRODUCTION_CARD.md`、`source_manifest.jsonl` 和冻结 protocol。
运行结束后还必须保留结果对照、输出媒体 inventory、失败日志和 evaluator 证据。
