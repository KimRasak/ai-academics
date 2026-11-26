# Stable Video Diffusion 候选准入审计

**日期：** 2026-09-20  
**范围：** 只读官方代码审计；未下载 SVD/SVD-XT 权重、未运行推理、未启动训练。  
**结论：** 可保留为公开 image-to-video checkpoint 的候选，但不能进入“论文指标复现”或
“训练复现”队列。当前可验证的是发布版代码、权重入口、推理配置和许可证；论文匹配的
评测/训练链路尚未公开闭环。

## 官方主源和发布身份

- 官方代码：[Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)。
  当前远端 HEAD 是 `e8cd657656fa5d61688191730d0e03242bf4ed44`；这不是 SVD 发布版本。
- 历史中明确标有 `Stable Video Diffusion` 的发布 commit 为
  `059d8e9cd9c55aea1ef2ece39abf605efb8b7cc9`（2023-11-21）。本审计以该 commit
  审查 SVD 文件，避免以后来的通用仓库状态替代论文时代代码。
- 官方权重入口：[SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) 与
  [SVD-XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)。发布说明称：
  SVD 从一张 576×1024 context frame 生成 14 帧；SVD-XT 同架构并针对 25 帧微调。
  本工作尚未下载这些权重，因此没有本地 SHA-256，也没有权重运行结论。
- 发布仓库附带 [SVD non-commercial community license](https://github.com/Stability-AI/generative-models/blob/main/model_licenses/LICENSE-SVD)；
  后续若下载或派生模型必须再次核验当时权重页的许可与访问条件。

## 已核验的推理路径

发布 commit 包含 `scripts/sampling/simple_video_sample.py` 与
`scripts/sampling/configs/svd.yaml` / `svd_xt.yaml`。`svd.yaml` 指向
`checkpoints/svd.safetensors`，固定 VideoUNet 与 Euler EDM sampler 配置；它说明
“权重如何接入推理”，但不包含论文 benchmark 的输入集或 evaluator。

## 论文主张复现的缺口

发布快照中没有找到以下论文级物料：

1. 论文每个报告值对应的 checkpoint 文件/hash；
2. 评测条件图像、prompt、seed、样本数、帧率、重编码及失败样本规则；
3. FVD/FID 或其他指标的 reference videos、evaluator 代码/权重/revision、逐样本分数和聚合；
4. 训练视频的原始媒体、许可证、筛选、split、caption/condition 映射；
5. SVD/SVD-XT 完整训练配置、优化轨迹、world size、总步数与可核验计算预算。

仓库在“Training”部分只提供通用 diffusion 训练示例，并明确要求用户为非玩具数据自行
修改配置、以 WebDataset tar 格式准备数据；这不能作为 SVD 论文训练程序或数据证明。

## 当前允许与禁止的工作

允许在后续取得权重并先冻结一个**独立公开 I2V 协议**后，做已标注为 checkpoint-only 的
生成与媒体审计。禁止把 demo 成功、任意自选图片，或后来外部 benchmark 的分数称为 SVD
论文指标复现；禁止把通用训练框架或自建 WebDataset 训练称为 SVD 训练复现。

只有同时获得论文行级 checkpoint 身份、公开评测协议/媒体/evaluator，以及公开可审查的
训练数据和训练预算后，才应创建 `reproductions/stable-video-diffusion/` 目录。
