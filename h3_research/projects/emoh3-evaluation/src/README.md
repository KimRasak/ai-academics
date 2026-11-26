# Source layout

- `analysis/`：离线统计、judge agreement、身份和人评分析。
- `judges/`：情绪与身份 VLM 判官。
- `generation/`：H3 生成、抽帧和输出验证。
- `training/`：数据 manifest、cache 编码和 LoRA 训练器。
- `paper/`：论文附录、图和盲标工具生成器。

远程实验脚本通常仍包含 `/gluster_hdd_cv/.../tmp/pilot` 的历史工作路径；本地分析入口则从本项目根目录解析 `results/`。

