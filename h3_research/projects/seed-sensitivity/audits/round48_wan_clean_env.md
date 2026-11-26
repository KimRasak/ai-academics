# Round 48：Wan 独立环境配置

## 进展

已创建不继承 system-site-packages 的 Wan 环境：

```text
/ssd2/jinzili/envs/wan_fun_clean
```

该环境 Python 3.9，单独安装了 CUDA torch 2.8.0+cu128，以及 diffusers 0.36.0、transformers 4.57.6、omegaconf、sentencepiece、accelerate、einops、safetensors、timm、ftfy、torchdiffeq、torchsde、imageio、opencv-python、decord、librosa、onnxruntime 等依赖。

为避免复制 gluster 上不支持的特殊目录，已只复制 VideoX-Fun 所需子树到：

```text
/ssd2/jinzili/wan_fun_src
```

并保留原始 checkpoint 不动。

## 当前阻塞

VideoX-Fun 的宽泛 `videox_fun.models` import 会导入 Ernie/FantasyTalking 等非 Wan 模块；在 Python 3.9 上触发了仓库代码的现代类型注解/可选依赖问题，随后 `cv2` 缺失 `libGL.so.1`。尚未运行 checkpoint load 或生成。

这不是 Wan checkpoint 失败，也没有污染 H3 或已验证 VT venv。下一步将针对 Wan 入口做最小 import patch（绕过非 Wan model exports，或仅提供 `libGL` headless workaround），再做实际 checkpoint load smoke。
