# 最终引用准入门槛

**更新：** 2026-09-23  
**状态：** 最终已核验引用 `0/30`；不得把候选 BibTeX 或 Scholar 检索结果计入最终引用数。

## 当前可核验盘点

| 来源层 | 数量 | 现状 | 能否计入最终 ≥30 引用 |
|---|---:|---|---|
| `pool_v1.bib` / `merged_v1.jsonl` 候选池 | 31 个 proposal 候选（共 148 条原始去重记录） | 有本地 metadata；未完成逐篇主源、相关性和资源审查 | 否 |
| 当前复现已使用的核心对象 | 10 个 | 主源入口与局部工件证据已在复现卡中固定 | 尚否：仍须补齐规范作者、版本/venue、拟支持论断与 proposal 段落 |
| 已批准最终引用 | 0 个 | 没有任何条目已完成本页的全部字段 | 否，缺口为 30 |

候选池中 `refs/v2/control_benchmark.jsonl` 与 `video_alignment.jsonl` 当前均为
零记录；它们不是空领域结论。2026-09-23 的单页 Scholar 查询在第一个请求即收到
“unusual traffic”并停止，事件见
[`v2/SEARCH_LOG.md`](v2/SEARCH_LOG.md)。不重试 headless、不绕过 CAPTCHA、不使用代理轮换。

## 每条引用的批准条件

一条引用只有同时完成下列字段后，才可从 `pending` 改为 `approved` 并计入 30：

1. 规范题名、完整作者、年份、版本及 venue/DOI（或明确 arXiv 版本）；
2. 可核验主源 URL，以及其本地保存的证据、release checkout 或人工核验记录；
3. 它支持的**唯一具体论断**，而不是泛泛“相关工作”；
4. 与已选研究假设的相关性，以及不采用它会缺少哪一类比较；
5. 代码、权重、训练数据、测试数据、evaluator 的公开状态；
6. proposal 中将出现的段落；
7. 排除检查：是否仅谈数字人/说话头像、私有数据，或与最终通用/可控视频问题不相干。

不能因某论文已有公开权重、已被 Scholar 收录，或引用数很高，就跳过上述字段。

## 当前优先审查队列

先审查已经实际影响研究判断的核心对象，再审查候选池。核心对象的主源入口和已允许/禁止的
论断见下表；它们来自各自 `source_manifest`、复现卡和总台账。

| 对象 | 主源与当前证据 | 可支持的 proposal 边界 | 尚缺的最终引用审查 |
|---|---|---|---|
| MiniMax-H3 | [官方发布](https://github.com/MiniMaxAI/MiniMax-H3)；Base 部署/运行 evidence | 公开 Base 的部署边界与可评测 checkpoint 路径 | 论文作者、版本/venue及正式可引用主张 |
| Sci-VBench | [官方代码](https://github.com/Sci-VBench/Sci-VBench)；[发布媒体](https://huggingface.co/datasets/Sci-VBench/Sci-VBench-Videos) | 作者媒体 evaluator replay 与 testmini 评测门槛 | 规范论文 metadata、正式评分协议引用 |
| MM-Diffusion | [论文](https://arxiv.org/abs/2212.09478)；[官方代码](https://github.com/researchmm/MM-Diffusion) | 可审计的联合音视频 checkpoint baseline | canonical venue/作者与论文聚合主张的措辞审查 |
| VBench | [官方代码](https://github.com/Vchitect/VBench) | 固定视频生成 evaluator 的版本约束 | 规范论文与启用维度的引用范围 |
| T2V-Turbo | [论文](https://arxiv.org/abs/2405.18750)；[官方代码](https://github.com/Ji4chenLi/t2v-turbo) | released LoRA 的独立 VBench checkpoint observation | venue、Table 1 与独立协议的严格区分 |
| VADER | [论文](https://arxiv.org/abs/2407.08737)；[官方代码](https://github.com/mihirp1998/vader) | released LoRA 的 reward/dynamics trade-off | Table 1 不能由发布 LoRA 代替 |
| VideoDPO | [论文](https://arxiv.org/abs/2412.14167)；[官方代码](https://github.com/CIntellifusion/VideoDPO) | 偏好数据 provenance 与缺失 pair score 的限制 | headline reweighting 不可写成已复现 |
| MotionCtrl | [论文](https://arxiv.org/abs/2312.03641)；[官方代码](https://github.com/TencentARC/MotionCtrl) | 可控视频 checkpoint smoke 与评测缺口 | 论文指标不可升级为已复现 |
| JavisDiT++ | [官方代码](https://github.com/JavisVerse/JavisDiT)；[模型](https://huggingface.co/JavisVerse/JavisDiT-v1.0-jav) | joint AV checkpoint/evaluator 接口边界 | ICLR paper metadata；4 条 smoke 不可称 mini benchmark |
| VABench | [论文页面](https://openaccess.thecvf.com/content/CVPR2026/html/Hua_VABench_A_Comprehensive_Benchmark_for_Audio-Video_Generation_CVPR_2026_paper.html)；[代码](https://github.com/tanABCC/VABench) | 评测 release 的可执行性缺口 | pairing/DNSMOS 缺失条件下不可做正式 score |

## 候选池的分批规则

- **训练与对齐：** 先处理 DPOK、differentiable reward fine-tuning、PRDP、DiffPPO、
  reward-hacking 等候选；每篇必须明确它是否能迁移到视频，而非只因图像 diffusion 相似。
- **联合音视频：** 优先 JavisDiT/JavisDiT++、SyncFlow、MOVA、MM-Sonate、Syncphony、
  AudioGen-Omni 等与最终问题直接相邻的工作。
- **可控视频：** 以 controllable-video survey、CtrlVDiff 和公开 control benchmark 为背景，
  但排除只与数字人/说话头像有关、无法支持通用视频假设的条目。
- **数据与评测：** 只引用能说明公开数据、正式 split 或 evaluator 限制的工作；不能让
  feature cache 或 latent 代替原始媒体 provenance。

每批完成后，必须在本文件新增一个逐篇审批表，列出 `approved / excluded / pending` 与理由。
在明确研究问题、正式数据集、强基线和评价协议之前，保持所有候选 `pending`，不据此启动
自主训练。

## 已完成的逐篇主源审查（第一批）

下列工作已在当前复现中实际使用，因此先完成了“本地主源、release 和可支持论断”的
审查。它们仍不是 approved final citations：最终假设尚未选定，而且规范完整作者名单与
canonical venue metadata 尚未从保存 PDF 的前页/正式发表页逐项转录。

| 工作 | 状态 | 本地核验主源 | 已核验资源与可支持论断 | 仍缺字段 / 不允许的升级 |
|---|---|---|---|---|
| MM-Diffusion (2023) | `pending` | arXiv `2212.09478` 的历史 PDF hash `39c9…a9b3c`；[官方代码](https://github.com/researchmm/MM-Diffusion) | Landscape 原始 MP4、base/SR checkpoints、I3D/AudioCLIP 和 standalone evaluator 均留有 hash；可支持“存在可审计的联合音视频 checkpoint baseline 和 FVD/KVD/FAD 链路” | 当前 PDF 副本缺失，外加完整作者/venue、最终假设段落；不得称论文六次聚合或训练复现 |
| T2V-Turbo (2024) | `pending` | arXiv `2405.18750v2` PDF，SHA-256 `528b…325a4`；[官方代码](https://github.com/Ji4chenLi/t2v-turbo) | VC2 LoRA、VideoCrafter2 base、paper-day VBench prompts/evaluator 与全 16 维独立 protocol 均固定；可支持“公开视频 diffusion post-training checkpoint 在固定 VBench 协议下可被检验” | 完整作者/venue、最终假设段落；不得将独立 seeds 的 `82.2872` 写成论文 Table 1 复现 |
| VADER (2024) | `pending` | arXiv `2407.08737v1` PDF，SHA-256 `d08a…2111`；[官方代码](https://github.com/mihirp1998/vader) | 发布 PickScore LoRA、VideoCrafter2、72 prompt × 5 paired seed 的 720 MP4、PickScore 与 VBench temporal evaluator 已固定；可支持“在独立公共配对协议下 reward 增益可与动态性下降并存” | 完整作者/venue、最终假设段落；发布 LoRA 未证明是 Table 1 checkpoint，不能称训练或 Table 1 复现 |

这三条的原始证据分别位于
`../reproductions/mm-diffusion/source_manifest.jsonl`、
`../reproductions/t2v-turbo-vc2/source_manifest.jsonl` 与
`../reproductions/vader-videocrafter/source_manifest.jsonl`。它们提供可追溯主源，
但不会因已进入本表就增加 approved 计数。

### 第一批 metadata 阻断记录（2026-09-23）

T2V-Turbo 与 VADER 的保存 PDF SHA-256 已回读并与 manifest 一致；MM-Diffusion 的历史
PDF hash 已保留，但其 `/workspace/mm-diffusion-audit/paper-arxiv.pdf` 副本当前缺失，
已在 source manifest 追加可用性审计。远端环境没有 `pdftotext`、`pdfinfo`、`pypdf` 或
`PyPDF2`，故不能仅靠本地工具可靠提取首页完整作者名单。随后对
`export.arxiv.org/api/query` 的三个已知 arXiv ID（`2212.09478`、`2405.18750`、
`2407.08737`）的直接 metadata 查询均被环境策略拒绝：
`Blocked: host 'export.arxiv.org' is outside note.com.`

这不是主源不存在、作者不存在或论文不可引用的证据。允许的下一步是从已保存 PDF
以可审计工具提取首页，或由人工核验 arXiv/正式会议页面后补入完整作者和 canonical
venue；在此之前三条保持 `pending`，不进行网络重试或以截断 Scholar 作者字段替代。

## 与复现蓝图的关系

最终 bibliography 必须包括实际使用的模型、训练方法、数据集与 evaluator；它不能仅由
31 条候选构成，也不能用候选数量宣称调研充分。当前可执行的实验结论以
[论文复现总台账](../dashboard/PAPER_REPRODUCTION_TRACKER.md) 和
[复现过程总台账](../dashboard/REPRODUCTION_PROCESS_TRACKER.md) 为准；本页只管理文献准入。
