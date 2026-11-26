# Research-work reproduction

科研论文、公开权重和训练流程的复现与证据工程。当前聚焦通用/可控视频生成与
MiniMax-H3 的可验证边界；不使用私有或玩具数据开展探索训练。

## 先看哪里

1. [论文复现结果总表](dashboard/PAPER_REPRODUCTION_TRACKER.md)：每篇论文是否公开
   权重、在何数据集/指标上验证、公开权重是否复现论文指标、训练为何能或不能复现。
2. [复现过程总表](dashboard/REPRODUCTION_PROCESS_TRACKER.md)：代码仓库、权重/数据
   下载链接、加工步骤、特征提取与 evaluator 流程。
3. [当前执行矩阵](planning/REPRODUCTION_EXECUTION_MATRIX.md)：正在做什么、停止条件
   是什么、哪些工作禁止启动训练。
4. [当前目标](planning/CURRENT_GOAL.md)：正在运行的 MiniMax-H3 benchmark 的冻结协议、
   等待 gate 与生成完成后的唯一推进顺序。

这三份文档按顺序回答“结论是什么、过程是什么、下一步是什么”。展示总表是当前
状态的唯一入口；JSONL 是同一内容的机器可读版本。

## 目录

| 目录 / 文件 | 用途 |
|---|---|
| [dashboard/](dashboard/README.md) | 给导师和协作者审阅的两张展示表 |
| [planning/](planning/README.md) | 证据等级、benchmark 蓝图、执行顺序和停止条件 |
| [reproductions/](reproductions/README.md) | 每篇已入队论文的原始 manifest、命令、运行记录和结果 |
| [governance/](governance/README.md) | 工作区边界、文件职责和状态更新规则 |
| [refs/](refs/README.md) | 文献检索结果与候选引用；不是最终 bibliography |
| [archive/](archive/README.md) | 已被替代的历史摘要，不用于更新当前结论 |
| `claim_ledger.jsonl` | 原子主张及其证据等级 |
| `reproduction_registry.jsonl` | 代码、权重、数据、算力等工件库存 |

## 当前执行

- `reproductions/mm-diffusion/`：Landscape 官方 checkpoint 的 2,048 样本
  DDPM/DDIM 指标复现。第 1/6 与预注册第 2/6 次均已完成媒体审计和原始/SR 独立
  评测；原始 64×64 的逐次 FVD/KVD/FAD×10k 为
  `249.0004/15.4862/10.7107`、`231.8951/14.3733/10.9198`。二次 C3 证据
  不是六次聚合、稳定方差或训练复现；预注册 run 3 正在运行，终态以前不得报告指标。
- 其余路径的可复现性、证据强度和训练准入条件见
  `planning/REPRODUCTION_EXECUTION_MATRIX.md`。没有同时满足数据、协议、权重与
  训练配置准入条件的路径，不启动“论文训练复现”。

## 文献边界

- 已核验的核心主源位于
  `../h3-audio-visual-emotion-consistency/literature/CORE_PRIMARY_SOURCE_LEDGER.md`。
- 31 条候选引用位于同目录的 `REFERENCE_CANDIDATE_AUDIT.md`；其 BibTeX key
  已对共享文献池作机械核验，但它们仍是待逐篇核对的候选，不能充当已验证证据或
  最终 bibliography。

## 项目边界

本项目负责：

- VideoCrafter2、T2V-Turbo、VADER、VideoDPO、MotionCtrl 等公开工作的
  权重和训练复现；
- VBench、EvalCrafter、T2V-CompBench、PhyGenBench、ChronoMagic-Bench
  等公开评测协议；
- 区分论文主张、公开权重实测和完整训练复现；
- MiniMax-H3 官方权重部署及独立公开 benchmark 验证；
- 维护可迁移到 H3 或其他视频模型的可靠结论。

本项目不负责：

- H3 proposal 的最终研究问题与方法设计；
- 私有数据上的探索训练。

H3 Audio-Visual Emotion Consistency 课题内容保留在
`../h3-audio-visual-emotion-consistency/`。只有达到明确证据状态的结论才被
引用回其 proposal。
