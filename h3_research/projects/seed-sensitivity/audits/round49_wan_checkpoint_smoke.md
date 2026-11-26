# Round 49：Wan checkpoint load smoke

## 环境

- `/ssd2/jinzili/envs/wan_fun_clean`，Python 3.9，独立环境
- torch 2.8.0+cu128，CUDA available
- diffusers 0.36.0，transformers 4.57.6
- VideoX-Fun 最小源码：`/ssd2/jinzili/wan_fun_src`
- OpenCV 已替换为 headless build
- VideoX-Fun model/pipeline exports 已缩减到 Wan 必需模块，避免非 Wan optional modules

## 结果

使用官方配置 `config/wan2.1/wan_civitai.yaml` 加载：

`/gluster_hdd_cv/user/jinzili/weights/Wan2.1-Fun-V1.1-1.3B-InP`

transformer checkpoint 成功加载：

```text
CHECKPOINT_LOAD_OK
parameter count = 1,564,428,608
```

尚未运行完整 T2V 生成；下一步是使用官方 `predict_t2v.py` 的最小修改版，设置较短 5 秒/低步数 smoke，记录与官方默认 81 frames/16 FPS 协议的差异。

后续首次 generation wrapper 多次在同一 import 处失败：官方脚本无条件导入 `CLIPModel`，而最小化的 Wan-only exports 未导出该可选 image encoder。wrapper 每次重新从原脚本生成临时文件，导致仅修改临时文件没有生效。未进入 checkpoint load/denoise，GPU 未被占用。下一轮应在 wrapper 内显式删除 `CLIPModel` import 或将对应 class 添加回 Wan-only exports，再启动一次；不要继续修改一次性 `/tmp` 文件。
