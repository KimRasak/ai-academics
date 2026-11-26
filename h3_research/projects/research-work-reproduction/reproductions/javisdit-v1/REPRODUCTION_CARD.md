# JavisDiT++ / JavisDiT-v1.0：公开权重验证卡

**状态：** 已完成公开权重 C2 生成与四条 JavisBench-mini 媒体/evaluator 接口 smoke；
官方 evaluator 已成功读取四对媒体。尚未启动 1,000 条 JavisBench-mini benchmark，故无
任何可报告的 JavisBench-mini 或论文分数。  
**范围：** 仅验证官方 JavisDiT-v1.0 checkpoint 在官方 JavisBench-mini 协议下的生成与
评测。它是 JavisDiT++（ICLR 2026）的官方模型页所指向的 v1.0 权重，不是早期
JavisDiT-v0.1 preview 权重。

## 论文主张与这次运行的边界

官方仓库的主结果表（当前 release README）给出 JavisDiT++ 的 JavisBench 指标，例如
FVD `141.5`、FAD `5.5`、CLIP `0.316`、AV-IB `0.198`、DeSync `0.832`。本次计划先在
**JavisBench-mini（1,000 prompts）**跑官方 240p、4 秒协议；它不是完整 JavisBench
的论文表格复现。因此无论结果如何，不能声称重现或否定上述完整 benchmark 数值。

## 已核验资源

- 官方代码：[JavisVerse/JavisDiT](https://github.com/JavisVerse/JavisDiT)，审计 revision
  `6821b8d161e9c5b195a58799859f08731a1471f9`。
- 官方权重：[JavisVerse/JavisDiT-v1.0-jav](https://huggingface.co/JavisVerse/JavisDiT-v1.0-jav)。
  官方客户端 dry-run 显示 5 个模型 shard、合计 4,539.84 MiB，另有 255.8 MiB LoRA；
  已下载并完成 76 个非缓存文件的 SHA-256 inventory（inventory SHA-256
  `98060f…f93a5`）；逐文件清单在远端 provenance manifest，链接和本地证据见
  `source_manifest.jsonl`。
- 依赖权重：官方命令还需
  [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) 与
  [AudioLDM2](https://huggingface.co/cvssp/audioldm2)。它们必须分别记录版本、许可和 hash，
  不能仅记录 JavisDiT adapter shard。
- 官方评测：[JavisBench](https://huggingface.co/datasets/JavisVerse/JavisBench)，已取得 mini
  CSV 与 FVD/FAD cache 的小型审计副本；SHA-256 见 source manifest。

## 数据与证据边界

JavisBench-mini 的 1,000 条 prompt/metadata 是公开的；模型生成的 MP4/WAV 必须保留，
同时保存每行 prompt、seed、生成命令、媒体 hash、decode 审计和 evaluator 输入映射。

但 benchmark 作者明确说明其 YouTube 原始参考音视频因版权限制没有发布，只发布
FVD/KVD/FAD 的预提取 reference feature cache。因此：

- 可执行作者公开的 checkpoint 评测协议；
- 不能自行审计真值原媒体或从其重建 cache；
- 不能把该评测用作训练集、C4 训练复现或原媒体完整 provenance 的证据；
- 除 FVD/KVD/FAD 外的生成侧指标只依赖公开 prompt + 本地生成媒体，但 evaluator weights
  和版本仍须锁定。

## 预注册的权重验证流程

1. 下载所有官方依赖，固定文件 hash、Hub revision、许可和本地路径；任何 gated/变更/下载
   错误都记录为资源状态，不替代为其他模型。
2. 用官方 `configs/javisdit-v1-0/inference/sample.py` 生成 JavisBench-mini 的 240p、4 秒、
   9:16 音视频；记录实际 world size、seed 行为、每 prompt 样本数、采样参数和失败项。
3. 保留每个生成 MP4/WAV，执行完整 decode/hash inventory；不以特征 cache 取代生成媒体。
4. 用固定 revision 的 `eval.javisbench.main` 和官方 mini FVD/FAD cache 单独运行指定指标；
   evaluator 自动下载的第三方权重必须在运行后补入 manifest。
5. 仅将结果报告为“官方 JavisDiT-v1.0 权重在 JavisBench-mini 官方评测协议下的 C3
   checkpoint evidence”，并与完整 JavisBench 论文数值分列。

### 当前 smoke 状态

首次 smoke 在 CUDA 前因缺 `soundfile==0.12.1` 停止；第二次 smoke 成功加载官方
JavisDiT-v1.0 checkpoint、合并官方 LoRA 并实际运行采样，但远端嵌套 shell 把传给
`--prompt` 的一条带空格文本拆成多个词级 prompt。检测到第 4 个词级输出后仅终止该进程。
该 run 的 4 个 MP4 与日志已全量 hash/ffprobe，保留为**失效的命令传递证据**，不能评分或
展示为模型样例。

第三次 run 改用一行 prompt 文件经 `--prompt-path` 传递，完成了有效的单样本 smoke：
prompt 文件仅一行且 SHA-256 为 `54bd…47bcc`；官方日志报告一条完整 prompt、50/50 steps、
`Inference finished` 和 `Saved 1 samples`。它输出且只输出一个可解码 MP4：H.264 432×240、
16 kHz 双声道 AAC、4.1265 秒、SHA-256 `15da…f0528`。该证据仅证明官方 released checkpoint
能够在声明配置下本地加载和生成；它不是 JavisBench-mini 结果、论文表格复现或训练复现。

随后，JavisBench-mini 发布 CSV 的前四条 `text` 被按原始顺序导出为四行 prompt 文件，逐行
保留 CSV 行号、ID 与文本 hash。官方 checkpoint 对四条各生成一个连续命名的 MP4；再使用
官方 `tools.datasets.datautil.extract_audio(..., dst_sr=16000)` 提取同名 WAV。四个 MP4、四个
WAV、四行 MP4→WAV→CSV 映射均已 hash/ffprobe。这只证明生成媒体可以满足官方 evaluator
的命名与音频接口，下一步仍需独立 evaluator 环境真正读取这四对媒体；四条样本不能报告
为 JavisBench 分数。

### evaluator 环境门槛（当前卡点）

生成环境与 evaluator 环境严格隔离。官方 evaluator 即使只请求 `fvd+kvd+fad`，也会在
参数解析前顶层导入 Synchformer、VideoAlign、AudioBox 与 ImageBind；因此缺失的非本次
指标模块也会阻止入口启动。2026-09-20 的逐项依赖审计已补齐官方明确 pin 的
`decord==0.6.0`、`trl==0.8.6`、主 requirements 的 `peft==0.13.2`，以及运行时所需
`rich`。仓库内部对 `timm` 的声明冲突：ImageBind 子目录写 `0.6.7`，但 evaluator 的
Synchformer 直接导入新版才有的 `timm.layers`；实际采用仓库主 requirements 的
`timm==0.9.16`，故这是记录在案的兼容层，并非严格 solver-resolved 官方环境。

固定 PyTorchVideo revision 还需 README 所附 augmentation patch；evaluator 环境已应用，
source/target SHA-256 都为
`b96a5306689fcc52acc6b911f5433351d4fc3b40d29b65565e84920553b78c79`。随后入口推进至
`matplotlib` 缺失。该次 `--help` 还证明 evaluator 在打印帮助前就会初始化 CUDA 并打开
所有 GPU device handle；进程在不增加显存时陷入 I/O 等待，已仅终止本工作启动的 PID，
未触碰任何外部任务。因此，后续不再把 `--help` 作为 CPU-only gate，必须在专属空闲 GPU
预检通过后以预注册的实际四样本接口运行完成剩余环境验证。该运行随后在物理 GPU 4
（`CUDA_VISIBLE_DEVICES=4`）退出码 0：官方 evaluator 读取了 4 对本地 MP4/WAV、发布的
mini feature cache、自动下载的 I3D 与 AudioCLIP，输出 `FVD 3324.9404 / KVD −11.0270 /
FAD 11.0482`。这只说明 evaluator 输入链路可运行；四条样本和分布特征 cache 都不支持
mini benchmark、论文指标或质量结论。详见
`run_manifests/javisbench_evaluator_environment_gate_20260920.json` 与
`run_manifests/javisbench_mini_batch4_evaluator_interface_smoke_20260920.json`。

### 生成环境门槛

官方安装目标是 Python 3.10、PyTorch 2.5.1 CUDA 12.1、torchvision 0.21.0、xFormers
0.0.28.post3 和 FlashAttention。当前共享远端默认环境是 Python 3.11 / PyTorch 2.8 CUDA
12.8，xFormers CUDA extension 不可加载，且缺 FlashAttention 与 decord；**不得直接使用**。
必须在工件 hash 审计完成后创建独立 Python 3.10 环境，先完成 CPU import/version gate，
才允许申请 GPU。详见 [`environment_gate.json`](environment_gate.json)。

## 训练复现门槛与停止条件

不启动训练。JavisDiT++ 的 Stage 2 使用不可再发布的 TAVGBench YouTube 媒体，官方只给出
video ID list；JavisBench raw reference media 也不发布。虽然 Stage 3 的
[AV-DPO](https://huggingface.co/datasets/JavisVerse/AV-DPO) 提供了 generated-only 子集，
它不闭合 Stage 1/2 的论文训练流程，也不能使 JavisDiT++ 成为 C4。

只有原始媒体、许可证、固定 split、完整各阶段配置/预算和论文匹配评测全部可审查时，才
重新评估训练准入。任何 mini benchmark 结果、released checkpoint 成功或 AV-DPO 单阶段
训练，都不得升级为 JavisDiT++ 完整训练复现。
