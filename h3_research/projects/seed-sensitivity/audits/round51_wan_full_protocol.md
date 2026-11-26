# Round 51：Wan2.1-Fun 官方协议 5s 视频生成与 VT 评分

## 协议（官方默认，未缩减）

- video_length 81 frames @ 16 FPS → 5.0625 s（ffprobe 实测），与 H3 pilot 的 5.175 s 时长对齐
- num_inference_steps 50，TeaCache threshold 0.10 / skip 5（官方推荐范围）
- seed 0，官方默认 prompt
- 分辨率 832×480（Wan 官方默认；与 H3 的 1344×768 不同，论文中作为 protocol difference 报告）
- GPU 3，model_cpu_offload，50/50 steps 完成约 40 s

## VT 结果（同一 evaluator/GPU 栈）

| Metric | Wan full seed0 | H3 pilot mean (32) |
|---|---:|---:|
| subject_consistency | 0.9560 | 0.9222 |
| background_consistency | 0.9579 | 0.9508 |
| motion_smoothness | 0.9951 | 0.9917 |
| dynamic_degree | 1.0000 | 0.5938 |
| aesthetic_quality | 0.5320 | 0.4773 |
| imaging_quality | 0.6943 | 0.6583 |
| VT | 0.8559 | 0.7657 |

单视频对单模型的均值没有统计意义；该表只作为 sanity 数据。要点：Wan seed0 的 DD=1.0，而 H3 的 32 条 pilot 中 DD 有 0/1 波动——DD 的 seed敏感性是跨模型共有的候选现象，正是 E4 想验证的。

## Gate 状态

- Wan 官方协议单视频生成：PASS
- 同一 evaluator 六指标评分：PASS
- 跨模型 seed-power replication（16 prompts × seeds 0–5）：待启动，预计 16 prompts × 6 seeds × ~75s ≈ 2 小时生成 + ~20 分钟 VT

注意：16:9 时长对齐了，但分辨率与 sampler 均为 Wan 官方默认；对比结论限定为 metric behavior replication，不是同配置质量比较。
