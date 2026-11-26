# Round 46：跨模型复现选择

## 盘点结果

远程确认 `Wan2.1-Fun-V1.1-1.3B-InP` checkpoint 约 19GB，且代码盘实际存在官方 VideoX-Fun 入口：

`model_demo/minimax_h3/tests/ecosystem/VideoX-Fun/examples/wan2.1_fun/predict_t2v.py`

该入口明确支持 Wan2.1-Fun text-to-video，默认 `video_length=81`、`fps=16`、`sample_size=[480,832]`，单 GPU 配置 `ulysses_degree=ring_degree=1`，使用 sequential CPU offload。它不是 H3 的 768p/24fps/5s 同协议，外部模型结果应明确作为 cross-model metric sensitivity replication，而非严格同配置对比。

另有 `HunyuanImage-3-Instruct-Distil`，是图像模型，不用于视频 VT。

## 下一步

1. 复制官方 `predict_t2v.py` 到工作目录，仅改模型路径、输出目录、prompt/seed，并先做单 prompt smoke；
2. 核查 Wan 环境的 diffusers/transformers 依赖以及 GPU 显存；
3. 生成后做 ffprobe 与 VT 单视频 smoke；
4. 只有上述通过，才考虑 16 prompts × 多 seeds。

不训练。
