# 展示台

此目录只放便于人阅读、比较和决策的当前台账；不放运行日志、临时命令或原始媒体。

| 文件 | 先回答什么问题 | 更新时机 |
|---|---|---|
| [PAPER_REPRODUCTION_TRACKER.md](PAPER_REPRODUCTION_TRACKER.md) | 哪篇论文有无公开权重？在什么数据集和指标上验证？权重/训练能否复现？ | 权重验证、训练准入或训练终态改变时 |
| [REPRODUCTION_PROCESS_TRACKER.md](REPRODUCTION_PROCESS_TRACKER.md) | 代码、权重、测试对象从哪里获得？怎样处理和评测？ | 下载、加工、evaluator 或 provenance 改变时 |
| 同名 `.jsonl` | 上述内容的机器可读版本 | 必须与 Markdown 同步 |

阅读顺序：先读“结果总表”，再读“过程总表”。任何结论都必须能回到
`../reproductions/<paper>/` 的 source manifest、protocol、运行 manifest 和媒体审计。
