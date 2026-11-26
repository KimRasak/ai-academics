# Round 37：辅助 Qwen2-VL judge smoke test

## 目的

尝试在空闲 GPU 7 上运行已有的 Qwen2-VL-7B-Instruct checkpoint，作为**独立辅助 judge**，不是 Sci-VBench judge。该模型仅用于一个 32-case pilot 视频的 4-frame coherence label，不能替代 Sci-VBench 的 Qwen3.5 rubric judge。

## 结果

- Checkpoint：`/gluster_hdd_cv/user/jinzili/weights/reward_models/Qwen2-VL-7B-Instruct`，约 16 GB
- Environment：`/gluster_hdd_cv/user/jinzili/env/qwen_vl_3_5`
- GPU：GPU 7
- 输入：一个 fresh H3-Base pilot MP4 的 4 帧
- 进程启动并保持约 4 分钟，但没有产生 stdout、result JSON 或 traceback；最终处于不可推进的 `D` 状态，已停止，避免占用 GPU。

## 诊断边界

这是初始化/共享文件系统或依赖导入阶段的 stall，尚未证明 checkpoint 或推理逻辑失败。已知同一类模型在历史 emotion judge 脚本中有可用产物，但本次环境的 `transformers` import 也在 bounded probe 中超时，因此不能把此次结果解释为模型不兼容。

## 决策

辅助 judge smoke 未通过，且没有写入任何分数。Qwen2-VL 不替代 Sci-VBench judge；fresh VT 结果保持当前主评测。后续若需要 auxiliary judge，应使用已验证的历史环境/脚本并先做最小 import-time profiling，不能继续让 GPU 被无输出进程占用。
