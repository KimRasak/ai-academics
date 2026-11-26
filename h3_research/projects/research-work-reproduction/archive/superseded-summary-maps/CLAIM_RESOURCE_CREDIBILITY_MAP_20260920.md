# 论文主张—公开资源—复现结果—可信度图（已归档）

> 本文件曾是摘要展示层，现已由
> [`dashboard/PAPER_REPRODUCTION_TRACKER.md`](../../dashboard/PAPER_REPRODUCTION_TRACKER.md)
> 与 [`dashboard/REPRODUCTION_PROCESS_TRACKER.md`](../../dashboard/REPRODUCTION_PROCESS_TRACKER.md)
> 替代。保留只用于追溯 2026-09-20 的历史判断；不要据此更新当前状态。

**更新：** 2026-09-20  
**范围：** 通用与可控视频生成优先，MiniMax-H3 为目标模型。本图只汇总已保留在
本项目中的主源、manifest、媒体审计和运行日志；它不是论文排行榜，也不把缺失
工件用推测补齐。

## 如何阅读

一条论文主张只有在下列链路每一段可审计时才可升级。`C2` 只说明作者已发布媒体
的评测器可回放；`C3` 只说明本地以发布 checkpoint 新生成的媒体可按固定协议评测；
只有 `C4` 才是从公开原始媒体到训练、生成、评测、聚合的完整过程复现。

```text
论文主张
  ├─ 公开代码 / 权重 / 原始媒体 / 标注与切分
  ├─ 固定生成协议（checkpoint、prompt、seed、sampler）
  ├─ 原始输出与每次失败记录
  ├─ 固定 evaluator 与逐样本/聚合记录
  └─ 训练输入、缓存 provenance、优化过程与 checkpoint
       └─ 全部闭环才可称 C4；任一缺口保持 C0–C3 或 U
```

## 当前路径

| 路径与论文主张 | 已审计公开资源 | 本地复现结果 | 当前可信度 | 不能升级的原因 / 下一步 |
|---|---|---|---|---|
| **MiniMax-H3**：发布 Base checkpoint 可重跑官方 768p T2VA/FL2VA/Ref2VA | Base 权重、官方脚本、条件媒体、参考 MP4、冻结路由；无公开训练程序或训练数据 | Sci-VBench 对 150 个作者发布 H3 视频为 C2；授权 4×80GB FL2VA 两次在 DiT FSDP 初始化 OOM，未监听、未请求、未生成 | C2 evaluator；C1 deployment failure | 不是本地 H3 生成。H3 已测试 `memory` 模式不降低失败的 DiT 峰值；需有该模型支持的降峰值配置，或不同硬件/分片范围 |
| **MM-Diffusion**：Landscape `Ours*` DDPM-1000 的 FVD/KVD/FAD | 官方代码、Landscape retained media、base/SR checkpoints、I3D/AudioCLIP evaluator 均固定 | run 1/2 均为 2,048 组完整 C3；original FVD/KVD 未回到论文六次聚合，FAD 接近。run 3 正在产生 2,048 组 | C3 checkpoint（两次终态；第三次未终态） | 缺作者六次 sample sets/seeds、上游 YouTube 可追溯来源/许可与精确重建 manifest；不得训练或称论文主张已复现 |
| **T2V-Turbo**：四步 VideoCrafter2 的 VBench 改善 | 官方代码、发布 LoRA、固定 VBench 路径 | 4,720 本地视频、16 维 VBench checkpoint 评测完成 | C2/C3 checkpoint 证据，训练为 E | 论文 WebVid-10M 样本/分片与媒体不可得，且必要 reward checkpoint 当前受限；不能用替代数据训练冒充论文复现 |
| **VADER**：Table 1 reward-gradient 训练收益 | 官方代码、发布 LoRA、部分公共 prompts | 360 对固定提示下的 released-LoRA 效应审计 | C2 checkpoint effect；Table 1 为 C0/C1 | 发布 LoRA 不等于 Table 1 checkpoint；缺精确 base、seed、样本数、checkpoint 选择与聚合 |
| **VideoDPO**：OmniScore reweighted DPO 指标 | 官方 20,000 MP4 / 10,000 pair archive 完整解码 | 数据/配对来源审计完成 | C1 data audit | 每对 OmniScore 权重和论文匹配 evaluator 缺失；配置中的 `dupbeta: 0` 也不执行 headline reweighting |
| **MotionCtrl**：CamMC/ObjMC/FID/FVD/CLIPSIM | 官方代码与 example checkpoint | 单 camera case deployment smoke | C2 deployment smoke；论文表为 C0/C1 | 407/283 test manifest、1,000 reference 视频、评测代码及论文匹配训练流程缺失 |
| **VABench**：公开 AV generation benchmark score | 代码、I2AV 图像、部分 evaluator weights 均固定 | 工件/依赖缺口审计完成，未正式计分 | C1 protocol audit | image-to-annotation join 与 DNSMOS wrapper 缺失；不猜测对应关系或以替代实现评分 |

## 对一个月训练 idea 的含义

当前没有任何论文训练流程达到 C4。因而在满足下列前，不选定或启动自主训练：

1. 选定一个具有可审计原视频、标注、许可、固定 split 和 cache provenance 的公开数据集；
2. 用同一数据与固定 evaluator 跑通至少一个强公开 checkpoint baseline；
3. 将问题写成可证伪假设，并预先锁定对照、指标、重复 seed 与停止条件；
4. 证明最近方法未在同一协议下已解决该问题；
5. 将 H3 限定为公开可部署的 Base 边界，不能把 hosted Context-IR/2K 能力写成可训练或已复现能力。

在此之前，GPU 仅用于本图中定义的 checkpoint 复现、媒体审计、正式 evaluator 和必要
预处理，不能用于私有/玩具数据的探索训练。

## 关联记录

- 定义、证据等级和完整方法 lineage：`../../planning/REPRODUCTION_BLUEPRINT.md`
- 优先级、停止条件和当前队列：`../../planning/REPRODUCTION_EXECUTION_MATRIX.md`
- 逐主张结构化记录：`../../claim_ledger.jsonl`
- 每个工作工件可用性：`../../reproduction_registry.jsonl`
- H3 的 proposal 选择门禁：`../../../h3-audio-visual-emotion-consistency/PROPOSAL_EVIDENCE_GATE.md`
