# 当前目标：MiniMax-H3 Sci-VBench testmini checkpoint 验证

**更新：** 2026-09-23  
**状态：** 同一 8×H100 Base 服务已恢复，既有 runner 正在续跑；不启动并行 H3 benchmark、训练或调参。

## 目标

在公开的 Sci-VBench `testmini` 150 条测试样本上，使用公开 MiniMax-H3 Base
checkpoint、作者逐条发布的生成设置和固定 six-VT evaluator，得到可审计的本地
checkpoint 结果，并与作者发布值分开比较。这是公开权重 benchmark 验证，不是完整
论文指标复现，也不是训练复现。

## 已冻结的输入和生成协议

- 测试集：官方 `Sci-VBench/Sci-VBench` 的 `testmini.jsonl`，150 条，SHA-256
  `09f2cb83c8e3dde74e56360f41968f2d6c9cd867cb276f30087dbfff6a066414`。
- 条件设置：官方 `Sci-VBench/Sci-VBench-Videos` 发布的 MiniMax-H3 sidecar；与测试集
  150/150 的 ID 和 prompt 精确匹配。
- 生成：T2VA、FL2VA variant、768p 16:9、15 秒、24 FPS、50 steps、seed `20260803`。
- 服务：项目专用 `osa00-h100-034` 的 8×H100 SGLang MiniMax-H3 Base 服务；无 LoRA。
- 预注册与完整 manifest：
  [SCIVBENCH_TESTMINI_VT_PREREGISTRATION_20260922.json](../reproductions/minimax-h3-official-768p/SCIVBENCH_TESTMINI_VT_PREREGISTRATION_20260922.json)。

## 当前 gate：续跑完成后再验证

150 条顺序生成由既有 resume-safe runner（PID `2824942`）继续执行。服务恢复后仍只允许
使用相同 released Base 权重、TP=2、Ulysses=4、Ring=1、无 LoRA 和冻结 manifest；不得
改变生成设置、另开 benchmark、训练或调参，也不得对部分样本计算或报告 aggregate VT。

**最近已核验观测（2026-09-23）：** 已恢复至 `146/150` 条 `runner_summary.json`、`0` 条
`error.json`。先前中断的第 129 条 `e719ea4d-…` 现为 `completed`（15.083333 秒；
`inference_time_s=211.9226`）；runner 当前正轮询下一条 queued request `0844a629-…`。
`/health` 返回 `ok`，`/v1/models` 报告 MiniMax-H3、`num_gpus=8`、`MiniMaxH3Pipeline`，
`/v1/list_loras` 显示无 loaded adapter。GPU 0–3 的外部作业不受干预；不得以四卡、不同
分片或部分样本替代。只有 `150/150` 条 summary、`0` 条 error 且 runner 已退出/无
in-flight request 三项同时满足，才允许进入下述验证阶段。

**后续服务端观察（同日）：** 第 135 条 `0844a629-…` 已完成；第 136 条
`1e38bd43-…` 虽一度在 denoise `42/49` 后仍显示 `queued, progress=0`，现已明确
`completed`，MP4 已写入，`inference_time_s=213.6078`、`peak_memory_mb=54878`。
此前 API 状态滞后不构成失败证据；保留既有 runner，不重启/重投，直到满足全量 gate。

生成完整后，严格按以下顺序推进：

1. 审计 150/150 条输出：逐条确认 request、status history、最终状态、MP4、SHA-256、
   ffprobe、prompt 与 settings provenance；检查缺件、解码失败和字节完全重复媒体。
   审计器已冻结为
   [`audit_scivbench_testmini_run.py`](../reproductions/minimax-h3-official-768p/tools/audit_scivbench_testmini_run.py)，
   仅在生成终态后运行；它不写入或改动媒体。
2. 审计通过后，以冻结的 Sci-VBench evaluator revision 运行 six-VT：subject consistency、
   background consistency、motion smoothness、dynamic degree、aesthetic quality、imaging
   quality；验证 150/150 条 evaluator 输出和聚合过程。
3. 分开对照三个值：作者发布 H3 `VT=0.734596`、作者发布媒体的本地 evaluator 回放
   `VT=0.735706`、本地 checkpoint 的新 VT。不得把相近 VT 写成完整论文指标复现。
4. 将媒体审计、配置、运行记录、聚合结果和差异原因更新至两张展示表：
   [论文复现结果总表](../dashboard/PAPER_REPRODUCTION_TRACKER.md) 与
   [复现过程总表](../dashboard/REPRODUCTION_PROCESS_TRACKER.md)。

PG、SCC、SC、LPF 依赖论文所用 Qwen3.5-397B judge，目前未闭环；不能以其它 VLM
替代并声称复现。H3 训练程序、训练数据和论文匹配训练协议也未公开，因此本轮不得
称为训练复现。
