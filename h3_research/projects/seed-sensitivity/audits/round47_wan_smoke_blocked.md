# Round 47：Wan cross-model smoke 环境检查

## 已确认

- Wan2.1-Fun-V1.1-1.3B-InP checkpoint 完整，约 19GB。
- VideoX-Fun 官方 `examples/wan2.1_fun/predict_t2v.py` 已定位，明确是 text-to-video 入口。
- 034 GPU 3–7 空闲，H3 30011/30020 healthy。

## Smoke 前阻塞

尝试在 evaluator venv 中补装 Wan 依赖时，发现该 venv 复用 `sglang310` system-site-packages；安装 transformers 5.16.1 后，导入 `diffusers` 失败：

`ImportError: cannot import name 'is_offline_mode' from huggingface_hub`

这说明不能继续在 VT evaluator 环境中装 Wan 依赖，否则会污染已验证的 evaluator/H3 运行环境。尚未启动 Wan 推理，也没有产生跨模型结果。

## 决策

跨模型实验暂缓。下一步创建完全独立的 Wan venv（不使用 system-site-packages），仅安装 VideoX-Fun/ Wan 所需依赖，然后做单 prompt/seed smoke。H3 服务和 Sci-VBench venv 不改动。
