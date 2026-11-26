# 复现工作区约定

**更新：** 2026-09-20  
**目的：** 让论文复现的状态、工件和运行记录各有唯一归属，避免“有代码目录”
被误读为“已复现”，或“有公开权重”被误读为“可训练复现”。

## 入口与权威顺序

1. `dashboard/PAPER_REPRODUCTION_TRACKER.md` 是导师汇报和日常状态的唯一入口；
   `dashboard/PAPER_REPRODUCTION_TRACKER.jsonl` 是同一结论的机器可读版本。
2. `dashboard/REPRODUCTION_PROCESS_TRACKER.md` 是代码、权重、数据、加工、特征和 evaluator
   过程的唯一汇总入口；其 JSONL 记录结构化 provenance。
3. `claim_ledger.jsonl` 记录可原子核验的论文主张与证据等级。
4. `../reproduction_registry.jsonl` 只登记工件可用性、算力和历史资源，不单独定义
   “能否复现”的结论。
5. `../reproductions/<paper>/` 保存该论文的原始证据、协议、命令、运行 manifest 和结果。
6. `../planning/REPRODUCTION_BLUEPRINT.md` 与
   `../planning/REPRODUCTION_EXECUTION_MATRIX.md` 是解释/排程文档；它们不能覆盖
   总表状态。

若文档冲突，按上述顺序回溯至运行 manifest、原始媒体审计、日志或 source manifest，
然后同步修正低优先级说明。历史日志可以保留旧时态，但必须有 supersession 记录。

## 两类论文对象

| 状态 | 是否建立 `reproductions/<paper>/` | 允许的内容 | 例子 |
|---|---|---|---|
| **执行/审计中** | 是，且一篇论文一个目录 | release/source manifest、协议、复现卡、命令、运行 manifest、报告值与本地结果 | MiniMax-H3、MM-Diffusion、T2V-Turbo、VADER、VideoDPO、MotionCtrl、VABench |
| **候选/未入队** | 否 | 只在总表和文献/资源池中登记；不得虚建空目录或宣称已开始复现 | VideoCrafter2、AnimateDiff |

VideoCrafter2 虽被下游工作用作固定 base，仍不是一项已启动的独立论文指标复现；
AnimateDiff 同理。二者只有在论文表格行、权重身份、正式数据/评测协议和成本审计完成后
才创建独立目录。

## 已执行论文目录的最小结构

每个 `reproductions/<paper>/` 至少必须具有：

```text
reproductions/<paper>/
├── REPRODUCTION_CARD.md       # 范围、论文主张、禁止升级的结论
├── source_manifest.jsonl      # 代码/权重/媒体/数据/评测来源、revision、hash
└── protocol.json 或等价冻结协议
```

在执行权重验证后，额外必须保存 `reported_vs_reproduced.json` 或等价的按指标对照；
开始/计划训练复现时，必须增加 `training_reproduction_gate.json` 或等价的训练准入与
停止条件。纯 evaluator 项目可用 `protocol_gate.json`，但必须解释为何不适用训练字段。

所有缓存、`__pycache__`、`.pyc`、`.DS_Store` 和可再生中间物放入忽略规则，不作为证据。
原始媒体、标注和媒体到 latent/cache 的 provenance manifest 不能以缓存替代。

**当前清理注记（2026-09-23）：** 已发现若干本地 `reproductions/**/__pycache__/`
目录。它们是 Python 可再生缓存，不是证据、不得被链接为复现工件，也不得影响任何
状态判断；在允许的工作区清理操作中移除。当前执行环境拒绝了删除命令，故本次没有
删除任何文件。

## 状态更新事务

一次状态变更至少同步修改：

1. `dashboard/PAPER_REPRODUCTION_TRACKER.md` 与 `.jsonl` 的权重验证/训练复现列；
2. `dashboard/REPRODUCTION_PROCESS_TRACKER.md` 与 `.jsonl` 的下载、加工、特征/evaluator 字段；
3. 该论文目录的运行/结果 manifest；
4. `claim_ledger.jsonl` 的原子主张（若形成新证据）；
5. 若影响排程或 proposal 准入，再更新 execution matrix / evidence gate。

对于运行中的任务，总表只能写“进行中”；只有 closeout、媒体审计和 evaluator 全部终态后
才能写入本地指标。失败同样必须留存日志、配置和停止理由。
