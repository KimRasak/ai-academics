# Runner layout

- `generation/`：基座、LoRA 和多 seed 推理。
- `training/`：cache 编码、LoRA 训练和环境安装。
- `evaluation/`：抽帧、judge 重跑与稳健性控制。
- `ops/`：GPU、内存、IPv6 和分布式诊断。

这些脚本记录的是远程 `/gluster_hdd_cv/.../tmp/pilot` 工作区的可执行配方，移动本地文件不会改变其远程绝对路径。

