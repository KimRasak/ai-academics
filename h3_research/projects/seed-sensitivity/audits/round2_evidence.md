# Round 2 证据审计

日期：2026-09-07

## 1. 官方请求归档

从远程权重快照的 `scripts/readme/` 原样复制了三条官方 768p 请求，且通过 `bash -n`：

| 请求 | SHA-256 |
|---|---|
| `reproducible-768p-t2va-request.sh` | `78bfe72365f27a53d145d0ddbf1c422de3cfe075e6558998cc7e01d7d41cdab1` |
| `reproducible-768p-fl2va-request.sh` | `82d1c06897da37e5976ae8f58cc1833ed04e92559aebe9f9578e0a890d2a8cb5` |
| `reproducible-768p-ref2va-request.sh` | `73ea0e5cc94440b2e8955dc0703a07e3c691d62967c481f5d9b3e92391fc882a` |

远程模型目录不是 Git checkout，因此没有可记录的 repository commit；以文件 manifest、模型卡 hash 和请求脚本 hash 固定当前快照。`official_checkpoint_manifest.json` 记录 611 个文件、总计 498,573,221,865 bytes；小于 20 MB 的文件另记 SHA-256，大权重暂以路径和字节数固定，后续在 GPU 宿主机后台计算 shard hash，避免 SSH 长时占用。

三条请求的硬事实：

- T2VA：10 秒、16:9、seed 0、无 condition，使用完整 Context-IR 展开 prompt。
- FL2VA：8 秒、auto ratio、seed 0、首帧图像 condition。
- Ref2VA：5 秒、auto ratio、seed 0、视频和音频 reference。
- 官方 shell 在提交后只查询一次状态便下载；为可靠复现需包装成轮询至 terminal status 的 runner，不能直接假设一次 GET 后已经 completed。

## 2. 当前服务身份

`osa00-h100-021:30011/v1/models` 报告：

- model/root：`/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3`
- `num_gpus=8`
- `dit_precision=bf16`
- pipeline：`MiniMaxH3Pipeline`

API 无法报告 variant 或是否使用了启动时 LoRA。launcher PID 98590 是启动器跟踪的 shell PID，不是可从 SSH 容器直接看到的宿主机进程。第一次 `/run` 审计因 shell quoting 错误未抓到 cmdline，已记录为失败证据，不能据此补推参数。

## 3. 通过 gpu_launcher 启动官方 Base

在完全空闲的 `osa00-h100-034` 上通过 `/run` 提交两个 detached resident 服务：

- GPUs 0–3：FL2VA，TP=2、Ulysses=2，port 30011；
- GPUs 4–7：Ref2VA，TP=2、Ulysses=2，port 30020；
- model path 均为 `/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3`；
- `lora_path=null`，无 CPU/layerwise offload、无 FSDP。

launcher 外层任务很快结束是预期 detach 行为；宿主机进程审计确认真正的两个 `sglang serve` 及子 rank 仍在运行。启动期间生成 GPU P2P cache，日志有 `ibv_open_device` warning，但没有 traceback/OOM。最终 Ref2VA 于 11:35:17、FL2VA 于 11:36:34 完成 warmup 并打印 `ready to roll`；两个 `/health` 均返回 `{"status":"ok"}`。可以进入官方请求复现，不能重复启动。

## 4. 历史 150-case 完整性

以 index 中每条 `video` 路径和 `sglang310/bin/ffprobe` 审计：

- 150 records；
- 150 status=`completed`；
- 150/150 文件存在；
- 150/150 ffprobe 成功；
- 0 missing，0 decode metadata failure；
- 150 条人工评分的四轴均为 null。

因此可确认“生成产物完整”，不能确认“感知质量通过”。结构化结果见 `history_150_integrity.json`。

## 5. Sci-VBench 代码与公开分数复现

下载并固定官方 evaluator：

- repo：`Sci-VBench/Sci-VBench`
- commit：`e8cb3988631d0c22d31c02fc5b1cc3693bfb7031`
- `python -m compileall` 通过；
- package import 通过；
- judge 与 process CLI `--help` 通过；
- 官方 `load_rubric()` 对下载的 150 条 `testmini.jsonl` 完整校验通过。

下载的数据与 hash：

| 文件 | SHA-256 |
|---|---|
| `testmini.jsonl` | `09f2cb83c8e3dde74e56360f41968f2d6c9cd867cb276f30087dbfff6a066414` |
| dataset card | `ddcad09aac02357fe60a7b83c00a36c24609365bff09569fc501577cd11f0dd4` |
| `judge_scores.jsonl` | `365da12b3a7545213d06af01ceb94286c2de1a5cbf443c28fdbe7fb7ed63c533` |
| `vt_scores.jsonl` | `0f930c045f3cb826524c00a257acd0cf0172835af89626ce54bb07bbdc1f0d96` |

公开结果覆盖验证：MiniMax-H3 在 testmini 上有 150 个视频 × 3 judge rounds = 450 行，并有 150 条 VT。按 dataset card 协议重新聚合得到：

| Model | PG | SCC | SC | Judge LPF | VT |
|---|---:|---:|---:|---:|---:|
| MiniMax-H3 | 2.5178 | 1.6311 | 2.6156 | 3.6311 | 0.73460 |

四项 judge 分数是 1–5 标尺。VT 是六个 VBench Video Quality 指标的非加权均值。该结果是对作者公开逐视频分数的离线再聚合，不是我们重新调用 judge；结构化输出在 `data/scivbench/released_testmini_aggregate.json`。它验证了论文结果处理协议，也给出下一步本地 H3 重生成的目标基线。

## 6. 当前研究判断

Sci-VBench 给出一个可直接验证的缺口：H3 的低层感知分数不差，但公开 testmini 的科学与因果正确性 SCC 只有约 1.63/5。现阶段只能把它作为待复现 failure mode，不能立即提出或训练方法。先做以下排查：

1. 用原始 `posted_prompt` 而非 Context-IR 展开 prompt，重生成固定小子集；
2. 复用作者相同 judge、相同 3 rounds；
3. 检查视频时长差异和 H3 15 秒上限是否使长实验 prompt 天生不可完成；
4. 按 domain、prompt 长度、阶段数和关键因果事件数分层；
5. 若新生成 H3 仍稳定复现低 SCC，再讨论 prompt planning、结构条件或后训练。
