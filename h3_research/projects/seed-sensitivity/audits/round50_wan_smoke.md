# Round 50：Wan2.1-Fun 跨模型 smoke 通过

## 环境

- 独立 venv：`/ssd2/jinzili/envs/wan_fun_clean`（Python 3.9，无 system-site-packages，torch 2.8.0+cu128）
- 最小 VideoX-Fun 源码：`/ssd2/jinzili/wan_fun_src`
- checkpoint：`/gluster_hdd_cv/user/jinzili/weights/Wan2.1-Fun-V1.1-1.3B-InP`（1.56B 参数，missing/unexpected keys=0）
- GPU：034 physical GPU 3（进程内 cuda）

## Smoke 设置（相对官方默认的差异，全部记录在案）

- video_length：81 → 17 frames
- num_inference_steps：50 → 4
- num_skip_start_steps（TeaCache）：5 → 2
- GPU_memory_mode：sequential_cpu_offload → model_cpu_offload
- seed 0，官方默认 prompt，480×832，16 FPS

## 结果

- 4/4 denoise steps，TeaCache enabled
- 输出：`/ssd2/jinzili/wan_smoke/00000001.mp4`
- ffprobe：H.264，832×480，16 FPS，1.0625 s
- VT 六指标 smoke（GPU 5，evaluator 与 H3 栈一致）：
  - background_consistency 0.9169
  - motion_smoothness 0.9935
  - dynamic_degree 0.0
  - aesthetic_quality 0.3746
  - imaging_quality 0.2002
  - composite VT = 0.552052，1/1 complete，0 errors

## 边界

这是**跨模型 pipeline 连通性 smoke**，不是 benchmark 复现：步数（4 vs 50）、帧数（17 vs 81）、时长（1.06s vs 5s）都是缩减的，VT 分数不能与 H3 的 5s 结果直接比较。它证明了：

1. Wan2.1-Fun checkpoint 完整可加载；
2. 官方 VideoX-Fun 入口在独立环境中可运行；
3. 输出可被同一个 VT evaluator 正常打分。

## 下一步（按 Gate 顺序）

1. 以官方默认 81 frames / 16 FPS（约 5 秒）重跑 seed 0，使时长与 H3 pilot 对齐；
2. 16 prompts × seeds 0–5 只在此通过后启动；
3. 所有 Wan 结果在论文中标注为 cross-model replication，协议差异显式列出。
