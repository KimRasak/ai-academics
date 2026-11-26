# JavisBench-mini 前四条接口闭环（非 benchmark）

**状态：** 预注册，等待生成。  
**目的：** 验证官方 JavisDiT-v1 生成、MP4→16 kHz WAV 提取、`sample_0000` 命名映射和
JavisBench 官方 evaluator 的 `fvd+kvd+fad` 输入接口能闭环。

这不是论文指标复现，也不是 JavisBench-mini 的有效统计。四条样本的任何数值都只用于
验证评测接口、缓存加载与失败处理，不能写入结果总表的论文数值列。

## 固定输入与映射

- 条件来源：官方 `JavisBench-mini.csv` 前四行，保持发布顺序。
- Prompt 文件：从 CSV 的 `text` 字段导出为四个非空行；文件 SHA-256、行号、原始 CSV
  行号、文本 hash 都必须留存。
- 生成命名：官方 `sample.py` 配合 `--prompt-path`，输出预期为
  `sample_0000.mp4` 至 `sample_0003.mp4`。
- 音频提取：调用官方 `tools.datasets.datautil.extract_audio` 或 README 规定的
  `datautil --extract-audio --audio-sr 16000` 等价路径，要求生成同名 `.wav`；保留命令、
  ffprobe、SHA-256 与 MP4→WAV 映射。
- evaluator：固定代码 revision `6821b8d` 的 `eval.javisbench.main`，传入
  `--eval-num 4 --metrics fvd+kvd+fad`、官方 mini reference cache 与独立 output JSON。

## 固定生成配置

- JavisDiT-v1.0、Wan2.1-T2V-1.3B、AudioLDM2 的官方已 hash 权重；
- 240p、9:16、65 帧、16 FPS、16 kHz、50 rflow steps、bf16、seed 42、batch size 1；
- 使用物理 GPU 4，只有启动前占用不超过 128 MiB 时才允许启动；
- 每个 prompt 均独立生成，失败不得静默跳过、重命名或补生成。

## 成功与停止条件

成功的“接口闭环”要求：恰好 4 个 MP4、恰好 4 个 WAV、命名连续、每个文件可由 ffprobe
解码、所有媒体和映射有 hash，evaluator 至少完成对 4 条 fake media 的
FVD/KVD/FAD 输入读取。若 evaluator 缺权重/依赖/实现，保存失败证据，不能替代实现。

即使全部成功，仍仅为 **C2 小批接口 smoke**：reference 原媒体没有公开，样本数为 4，
不具备 JavisBench-mini 或论文主张的数值含义。只有小批媒体和 evaluator 接口均闭环后，
才可考虑预注册 1,000 条正式 checkpoint 评测。
