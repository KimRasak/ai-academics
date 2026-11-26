# AnimateDiff 候选准入审计

**日期：** 2026-09-20  
**范围：** 只读审计；未下载模型权重、未下载训练数据、未启动推理或训练。  
**结论：** 保持“候选/未入队”。公开代码和下载入口可以核验，但论文匹配训练、
评测和成本路径尚未闭环，不能创建 `reproductions/animatediff/` 或声称开始复现。

## 已核验的官方主源

- 官方代码：[guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff)，远端
  `HEAD`：`e92bd5671ba62c0d774a32951453e328018b7c5b`（2024-07-17）。
- 官方论文入口：[arXiv:2307.04725](https://arxiv.org/abs/2307.04725)。仓库 README
  将其标为 ICLR 2024 Spotlight 的官方实现。
- 官方模型入口：[guoyww/animatediff on Hugging Face](https://huggingface.co/guoyww/animatediff)。
  README 明确列出 v2 motion module `mm_sd_v15_v2.ckpt`、八个 MotionLoRA，以及
  v3 SD 1.5 motion module `v3_sd15_mm.ckpt`；这些是可下载对象的入口，**尚未下载或
  hash**，也尚未确认它们分别对应论文中的哪一行。

## 训练路径审计

冻结仓库的 `train.py` 将数据集直接命名为 `WebVid10M`。其数据加载器要求：

- 一个 CSV，至少有 `videoid`、`name`、`page_dir` 三列；
- 对应的原始 `<videoid>.mp4` 媒体目录；
- 随机起点抽取 16 帧，stride 4，并做随机水平翻转、resize、中心裁剪。

但官方 `configs/training/v1/training.yaml` 只给出
`csv_path: path_to_csv_file` 与 `video_folder: path_to_video_foler` 占位符；示例的
`max_train_steps: 100`、`train_batch_size: 4` 是管线示例，不能当作论文训练预算。
仓库没有固定论文训练 CSV、原始视频 inventory、许可证、过滤规则、train/validation
split、完整步数、world size 或可映射到论文最终 checkpoint 的训练日志。

因此不能以任意 WebVid 子集、重建下载或短示例训练代替论文训练复现。

## 评测与权重验证路径审计

现有官方仓库可直接识别的是推理 demo / MotionLoRA 控制配置，不是论文表格的冻结
生成协议。尚缺：

1. 每个论文数值对应的 motion-module checkpoint 身份和 hash；
2. 论文匹配 prompt / condition / seed / 每 prompt 样本数；
3. FVD reference media、CLIP evaluator revision/weights、人工评测样本及聚合规则；
4. 生成分辨率、帧数、FPS、采样器和失败样本处理；
5. 在不超过 8×H100 的范围内复现完整 motion-module 训练所需的已报告或可推导预算。

在上述信息闭环前，权重只能标为“有公开权重但尚未验证论文指标”，训练只能标为
“不能训练复现（当前公开物料不足）”，不是“训练结果较差”。

## 准入条件

以下材料到位后，才可以为 AnimateDiff 建立独立 reproduction 目录并启动权重验证：

1. 论文精确表格行与对应 checkpoint 的官方身份；
2. 可下载且保留 hash 的论文匹配 motion module 与 Stable Diffusion base；
3. 冻结的公开评测输入、reference media 与 evaluator；
4. 可审查的原始训练媒体、CSV、许可证、筛选/split 和预处理映射；
5. 明确训练配置与不超过 8×H100 的成本估计。

本审计不应被用于把 demo 成功、权重可下载或任意 MotionLoRA 小训练写成论文指标或
完整 AnimateDiff 训练复现。
