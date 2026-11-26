# MiniMax H3 可验证研究计划

> 状态：**论文冲刺阶段**——E1–E4 实验全部完成，初稿 v0.1 就绪  
> 最近更新：2026-09-13  
> 目标：约一个月后可投稿的 AI / 多媒体 / 视觉会议或 workshop（B 级或以下可接受，首选 WVAQ@WACV 2027，备选 ReScience C / TMLR）。

## 0. 当前进度速览（2026-09-13）

| 阶段 | 状态 | 关键产物 |
|---|---|---|
| E1 human join | 完成 | `data/scivbench/e1_human_join.json`、`e1_bootstrap.json`、fig3 |
| E2 released VT 复现 | 完成（gate PASSED） | `released_vt_comparison.json` |
| E3 H3 seed-power | 完成（gate PASSED，**12 seeds**） | `e3_seed_power.json`；12-seed flip 27.46%（`icc_12seed.json`） |
| E4 跨模型 Wan replication | 完成（v2 有效，**12 seeds**） | `e4_wan_seed_power.json`；12-seed flip 17.23%，ICC 0.48（`icc_12seed.json`） |
| Leaderboard 间距 | 完成 | `leaderboard_spacing.json`（median gap 0.0066，H3 rank 4/8） |
| Rank stability | 完成 | `seed_rank_stability.json`（12-seed：H3 0.54 / Wan 0.54 / LTX 0.46，位移 3.2–3.5/16） |
| 文献 | 种子库完成 | `literature/refs/`（78 条 + shortlist） |
| 图表 | 3 图 1 表 | `paper/assets/` |
| 初稿 | v0.1 | `paper/main/main.tex`（Overleaf 可编译，数字已 QA） |
| 复现指南 | 完成 | `docs/REPRODUCE.md` |

**剩余**：目标 workshop 模板排版、匿名化检查、co-author 通读、按 deadline 打磨。详细结论见 §"当前结论"与 `audits/round40–60`；写作结构见 `docs/PAPER_OUTLINE.md`。


## 1. 当前结论

MiniMax H3 是一个公开权重、但完整产品链路并未全部开源的 omni-modal 音视频生成系统。官方公开的 H3-Base 有 FL2VA 与 Ref2VA 两套 checkpoint；H3-Context-IR、H3-Regenerate-2K 以及原生稀疏注意力当前没有随首版权重完整开源。因此，现阶段不能把在线 2K 效果等同于本地 H3-Base，也不应在没有对照实验时声称复现完整 H3。

现有远程资产比预期成熟：基座权重、Turbo LoRA、合并权重、SGLang/Diffusers/vLLM-Omni 部署脚本、150 条 anime case、循环 LoRA 四格实验所需脚本、控制视频实验与自动指标代码均已存在。当前最重要的工作不是立刻训练，而是把这些资产整理成可信的基线：补官方案例复现、修正参数口径、补人工评测和关键对照，再据失败模式决定研究方向。

## 2. 可核验事实与来源

### 2.1 官方系统边界

来源以 [官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3) 和 [官方仓库](https://github.com/MiniMax-AI/MiniMax-H3) 为准：

- H3-Base 是 33B dense single-stream Omni-Transformer；约 13B 参数位于 AdaLN 相关分支。
- 文本/视觉语义编码器使用完整 Qwen3-VL-32B，并取第 50 层 hidden state。
- Visual VAE 为 `f16t4d24`，进入 Transformer 前再以 `1×2×2` patchify，有效空间下采样 32×、时间下采样 4×。
- Audio VAE 将 32 kHz 双声道音频分别压缩为 40 Hz latent token 序列。
- 本地开源输出为 768p、24 FPS、32 kHz stereo，时长 4–15 秒。
- FL2VA checkpoint 支持 T2VA / 首帧 / 尾帧 / 首尾帧；Ref2VA checkpoint 支持图像、视频和伴随视觉输入的音频参考。
- 发布 checkpoint 已经做过 CFG distillation。
- 官方说明模型训练时支持 native sparse attention，但首版只发布 full-attention inference；原生 sparse attention 尚待发布。
- H3-Context-IR 与 H3-Regenerate-2K 是托管组件，当前不能做纯本地完整复现。

### 2.2 本地资产盘点（远程共享盘）

远程入口：`ssh -T -o RemoteCommand=none jinzili_mv_260423`。该别名配置了 RemoteCommand，因此执行远程命令时必须显式加 `-T -o RemoteCommand=none`。

| 资产 | 路径 / 状态 |
|---|---|
| 官方/历史 H3 权重 | `/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3` |
| Turbo 4/8 step LoRA | `/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3-Turbo` |
| 合并权重 | `MiniMax-H3-Turbo8-merged-a8`、`MiniMax-H3-Turbo8-merged-a128` |
| 社区循环 LoRA | `/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3-LoRA` |
| H3 ControlNet | `/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3-Fun-Controlnet-Union` |
| 环境 | `/gluster_hdd_cv/user/jinzili/env/sglang310`、`minimax_h3_vxf` |
| 部署与实验代码 | `/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3` |
| 150 条 anime 结果 | `tests/anime_style/report/{index,anime_metrics,anime_ratings}.json` |
| 加速结果 | `tests/acceleration/report/*.json` |
| 生态实验 | `tests/ecosystem/` |

2026-09-07 实测状态：

- `osa00-h100-021:8659/list` 返回 launcher PID 98590、`running=true`。
- `osa00-h100-021:30011/health` 返回 `{"status":"ok"}`。
- 029 launcher/服务拒绝连接；034 launcher 可连接但登记进程已停止，30011 未监听。
- 当前 021 服务是 Turbo/合并权重链路，不应拿来宣称官方 base checkpoint 的精确复现。

### 2.3 已有实验中可以相信什么

1. **服务链路已经真实跑通。** 日志显示完整 text/visual/audio encode、8 次 denoise、decode、MP4 落盘；一次 768×768、约 5.17 秒请求的 pipeline 时间为 5.67 秒，峰值约 49.8 GB/GPU。
2. **8-GPU TP=2 × Ulysses=4 的已有时间证据：** 50-step 约 19.6–20.1 秒，8-step 约 3.27–3.39 秒（3 个 case）。这些是工程吞吐结果，不是质量复现。
3. **已有 150/150 anime case 标记 completed，自动指标也有 150 条。** 但 `anime_ratings.json` 的人工四轴字段均为 null，因此不能把它写成已完成的感知质量评测。
4. **循环 LoRA 已有 12/12 completed。** 同图首尾 pin 相比仅首帧，6 对样本的 `LOOPDIFF_RGB` 由 5.50 降到 2.08、`LOOP_PSNR` 由 25.41 dB 升到 35.48 dB；同时运动量下降。它证明 pin 有效，尚未证明 LoRA 有效，因为缺 base/no-LoRA 对照。
5. **ControlNet 跑出了 3 个视频。** 结果 JSON 没有 `status` 字段，但包含输出路径、帧数、耗时与显存；此前把它们统计为“失败”是不正确的。应先检查文件和视觉质量，再决定成功与否。
6. **Turbo LoRA 证据有高风险。** 历史记录显示直接 SGLang 加载曾出现 `applied to 0 layers`；`a8` 合并又可能因 bf16 舍入丢失多数 delta。`a128` 是工程 workaround，但不等于官方 LoRA 数学配置。所有旧 Turbo 质量结论都需以权重差异、命中层和固定 seed A/B 重新审计。
7. **采样步数存在 off-by-one 风险。** 当前 SGLang 路径中 `num_inference_steps` 表示 sigma 网格点数，实际 NFE 为 steps−1。旧报告把 50、8 等字段直接称为 NFE，必须回看请求与日志后统一口径。

## 3. 文献地图

Scholar 两轮共抓取 168 条原始记录，按 `cluster_id` 合并为 156 篇；第二轮针对 H3、加速、音视频同步评测与 reference identity 新增 40 条记录（8 条重复），5 个请求、0 次人工接管。完整产物：

- `literature/out/step1_keywords.jsonl`
- `literature/out/step2_targeted.jsonl`
- `literature/out/literature_merged.jsonl`
- `literature/out/literature_merged.csv`
- `literature/out/literature_report.md`

质量警告：87% venue 和 74% author 在 Scholar 列表页被截断；该集合适合检索与选读，不直接作为最终 BibTeX。最终引用必须回到论文、OpenReview/CVF/ACM 页面核验。

优先阅读簇：

| 簇 | 代表工作 | 与 H3 的关系 | 当前动作 |
|---|---|---|---|
| H3 直接扩展 | H3-World、SolarWM、Code World Model | 直接基于 H3 做 world control / world model | 核验代码、权重、训练数据与表格复现条件 |
| H3 专项评测 | Sci-VBench、VGI-Bench、PAWBench、HarnessEval-W | H3 已列入被测模型或 world-model 评测 | 优先复现公开 evaluator，而不是先造新 benchmark |
| 音视频统一评测 | TAVGBench、T2AV-Compass、PhyAVBench、LongAV-Compass | 对应 H3 原生音视频输出的核心卖点 | 选择开源、计算预算可控的子集做 baseline |
| reference consistency | Identity-Preserving Video Generation Challenge、MagicMirror、OpenS2VEval/RefAlign | 对应 Ref2VA 多参考能力 | 复用 identity / text / temporal 指标与 prompt |
| 加速 | Sparse VideoGen、Sparse-Distill、VORTA、Efficient-VDiT、SubBlock | 对应 H3 未发布 sparse attention 的空缺 | 只做可插拔实现复现；先测质量-速度曲线，再谈方法 |
| 通用视频评测 | VBench、T2V-CompBench | 提供可复用质量/一致性基线 | 先跑小子集和环境自检 |

## 4. 分阶段执行计划

### Gate 0：证据卫生（现在）

完成条件：

- 固定模型目录、commit、环境 lock、checkpoint 文件清单与 hash；
- 所有实验统一记录 `requested_sigma_points` 与 `actual_nfe`；
- 区分 `base`、官方 Turbo LoRA、runtime-converted LoRA、a8/a128 merged workaround；
- 对历史 JSON 做 schema audit，不能把缺失 status 当失败，不能把 null 人评当 pass；
- 验证 150 个视频文件仍存在且 ffprobe 可读。

### Gate 1：官方 H3-Base 最小复现

在空闲 4×H100 或 8×H100 上启动**未合并 Turbo**的官方 base checkpoint，逐项运行官方固定 T2VA、FL2VA、Ref2VA 768p request。

每项保存：请求 JSON、服务启动命令、代码 commit、环境 freeze、模型文件清单、seed、实际 NFE、耗时、峰值显存、stdout/stderr、输出 MP4、ffprobe、与官方参考视频的帧率/时长/音频规格比较。

成功标准：3/3 completed，MP4 可解码，24 FPS，音频轨存在且接近 32 kHz stereo；不要求逐像素一致，因为分布式 kernel、框架版本和随机实现可能不同。若官方提供 hash 或明确 deterministic 合同，再追加严格比较。

### Gate 2：复现一个公开 evaluator

优先级：

1. Sci-VBench evaluator（已有公开评测代码搜索结果）；
2. VBench 的小规模标准维度；
3. TAVGBench / T2AV-Compass 中可公开下载且无需闭源 judge 的音视频同步指标；
4. reference identity benchmark 的公开 prompt 与身份指标。

先用作者提供示例做 evaluator self-test，再测 H3；否则 evaluator 错误会被误当模型缺陷。

### Gate 3：补齐已有结果的关键对照

- 循环：`base/no-pin`、`base/pin`、`LoRA/no-pin`、`LoRA/pin` 四格；至少 6 内容 × 3 seeds。
- Turbo：base 50-NFE teacher、base 8-NFE、官方 Turbo 8-NFE；至少 16 prompts × 3 seeds。a128 workaround 单独列为工程变体，不混入官方结果。
- Reference：1/2/4/9 张参考图的身份保持、prompt 遵循、运动质量与延迟/显存。
- Audio：事件类、环境类、对白类分层，报告音频存在性、响度/静音率、语义和时序同步；不以“有音轨”等同“音频可用”。

### Gate 4：研究方向决策（不早于前三个 Gate）

只有当一个失败模式满足以下条件才进入方法实验：

- 在至少 2 个公开评测或自建但冻结的测试集上复现；
- 不是部署、LoRA 命中、NFE 口径、judge 或文件损坏造成；
- 有强基线可比较；
- 一周内能实现最小方法并在小规模样本上做否证。

候选方向目前仅作为待验证问题，不是 paper claim：

- H3 full-attention 的 training-free sparse attention / cache 在音视频联合 token 上是否保持同步；
- 多参考数量增加时，身份保持与 prompt 遵循是否发生可测的竞争；
- few-step Turbo 是否先损害 audio-video synchronization，而非单模态画质；
- first/last-frame pin 的闭环收益能否在不牺牲运动量的条件下实现。

## 5. 新增的一手线索：H3 音频与发布完整性 failure modes

背景调研代理报告了一组来自官方 GitHub issue tracker 的具体、可测试问题。关键入口包括 [speaker voice-identity bleed #17](https://github.com/MiniMax-AI/MiniMax-H3/issues/17)、[weight-layout discrepancy #48](https://github.com/MiniMax-AI/MiniMax-H3/issues/48)、[audio onset artifact #50](https://github.com/MiniMax-AI/MiniMax-H3/issues/50)、[audio onset artifact #51](https://github.com/MiniMax-AI/MiniMax-H3/issues/51)、[dialogue sung #68](https://github.com/MiniMax-AI/MiniMax-H3/issues/68)、[V2V timing/voice drift #69](https://github.com/MiniMax-AI/MiniMax-H3/issues/69)、[high-motion failure #74](https://github.com/MiniMax-AI/MiniMax-H3/issues/74) 和 [non-CFG checkpoint request #75](https://github.com/MiniMax-AI/MiniMax-H3/issues/75)。

本会话尝试用 GitHub API 二次核验时命中匿名 API rate limit，web search 也耗尽余额，因此这些线索在进入实验前仍需逐页保存 issue 正文和评论快照。当前不得把 issue 报告当成已证明的普遍模型缺陷。

按“可客观测量、无需训练、一个月可闭环”排序：

1. **多说话人 voice binding**：同/异性组合、Ref2VA voice references、FL2VA composed frame 和多 seeds；用 diarization + speaker embedding 与唇动/说话人归因检查。先做 measurement，训练自由 mitigation 也必须在冻结 benchmark 上验证。
2. **音视频加速的 audio cost**：已有加速报告几乎只用视频 SSIM；比较 Base 50-NFE、Base 8-NFE、Turbo 8-NFE，在 latency/SSIM 外增加 AV sync、speaker identity、静音率和 onset artifact。H100 上不计划复现 SM100-only 的 SubBlock kernel。
3. **开头 boundary ghost**：先用 AudioVAE round-trip 将 VAE 边界问题与 denoiser 问题拆开，再测试 padding/overlap/crossfade；成本低，但论文贡献上限可能只适合 short/workshop。
4. **Context-IR 贡献消融**：raw prompt、本地 Base + hosted Context-IR prompt、官方端到端输出三臂比较；必须记录 API 时间、token 成本和服务版本漂移。
5. **发布完整性审计**：比较不同权重布局/裁剪路径的数值和输出等价性；适合 reproducibility/resource track，创新性较弱。

这些仍是 reproduction candidates，不是已经选择的论文 idea。当前 Sci-VBench 的 SCC 复现和官方三案例优先级不变；voice binding 因一手 issue 报告明确，列为并行的小规模 smoke-test 候选。

## 6. 新增加速证据：基线必须重设

背景代理从 `diffusers` 主线源码和社区 H3 采样文档核验到：H3 transformer 已继承 `CacheMixin`，因此可直接做 cache instrumentation；同时，公开 H3 测量显示二阶 `res_multistep` sampler 本身可能比 cache 有更好的质量—调用次数折中。社区报告的代表数值包括：Cache-DiT 约 1.99×/LPIPS 0.233、velocity cache 约 3.20×/LPIPS 0.426，而 `res_multistep` 21 仅 20 次 DiT 调用、denoise 约 2.46×、PSNR 26.3 dB。来源入口：[Diffusers H3 transformer](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_minimax_h3.py) 和 [H3 sampling notes](https://github.com/HM-RunningHub/ComfyUI_RH_MinMaxH3/blob/main/docs/sampling.md)。

这些结果尚未在我们的 H100/SGLang 路径上复现，因此只用于修正实验设计：

- 加速 baseline 不能只用 Euler-50；至少加入 AdaLN precompute + `res_multistep`-21。
- 同时报告 denoise-loop 与 end-to-end latency，防止忽略 text encoder、权重加载与 VAE decode 的 Amdahl 上限。
- cache 与 higher-order sampler 必须分别比较，不能默认叠加有益。
- H3 是视频、文本、音频 packed full self-attention；全局 drift signal 可能被数量巨大的 video tokens 主导。任何 cache 研究必须从第一天报告 audio/AV 指标，并比较 global 与 per-modality gating。
- SubBlock 当前公开 kernel 的硬件约束偏向 SM100；我们的 H100 主实验不把它列为可直接复现 baseline。

因此，**modality-aware cache gating** 升为潜在方法方向第一名，但仍须先复现公开 cache/sampler 数值，并证明 global cache 的音频损害稳定存在，才能通过 Gate 4。若不存在该损害，立即停止这个 idea。

## 7. 一个月时间盒与停止规则

| 周 | 输出 | Go / No-Go |
|---|---|---|
| 第 1 周 | Gate 0–2；官方 3 case；公开 evaluator self-test | base 或 evaluator 不通则只做复现/系统论文，不启动训练 |
| 第 2 周 | 两条基线矩阵与可靠统计；人工盲评协议 | 没有稳定失败模式则扩大调研或转为 benchmark/resource 贡献 |
| 第 3 周 | 仅对通过 Gate 4 的一个方法做实现、消融与否证 | 主指标无稳定改善或明显牺牲另一模态则停止该 idea |
| 第 4 周 | 补样本、统计、失败分析、论文与开源包 | 不追逐第二个 idea；保证 artifact 可复现 |

计算预算先小后大：smoke test（1 prompt × 1 seed）→ pilot（8–16 prompts × 2 seeds）→ confirmatory（冻结配置后 ≥3 seeds）。任何训练先用小数据/短步验证 loss、加载和输出变化，禁止直接长训。

## 8. 投稿策略

背景 venue 扫描给出了初步 deadline-to-candidate mapping，但由于搜索配额中途耗尽，**所有日期和规则在投稿决策前仍须逐个打开官方 CFP 复核**，当前不能视为最终日历：

| 暂定日期 | Venue | 候选工作 | 当前判断 |
|---|---|---|---|
| Oct 9 abstract / Oct 16 paper | IEEE FG 2027 short | speaker-binding benchmark + training-free mitigation | 新方向中最匹配，4+1 页适合 measurement paper；需立即核验官方 scope/匿名/截止时区 |
| Oct 12 | WVAQ @ WACV 2027 | accelerated AV generation 的 audio-quality measurement | 与 image/video/audio quality 主题高度匹配；方法部分若来不及则只投严谨 measurement |
| Oct 12 | ARR Oct cycle | Context-IR dependence ablation | 语言介导条件重写角度可行，但是否达到 NLP 贡献标准需谨慎 |
| Nov 11 | ICASSP 2027 satellite workshops | onset boundary ghost 或 modality-aware cache | workshop 主题列表尚需官方确认 |
| rolling | ReScience C / TMLR | release-integrity / replication audit | 适合复现和 negative result，但时间可能超过一个月 |

已排除或降级：NeurIPS 2026 workshops 已过期；MLRC 2026 需要已有 TMLR accepted paper，不适用。加速方法在当前窗口没有明确的 B-tier 主会落点，可关注尚未正式公布的 ICME 2027，但不能凭往年节奏假定 deadline。

内部已有 EmoH3 工作可能匹配 Oct 6 的 AINIME @ WACV，但它与本目标下的全新 H3 研究是否共享实验和作者资源，需要单独协调；本计划不能擅自把既有工作当成本项目产出。

## 9. Round 2 更新：第一条论文指标复现链路

已完成：

- 官方 T2VA/FL2VA/Ref2VA 请求已归档并记录 SHA-256；
- 官方权重快照已生成 611 文件、498.57 GB manifest；
- 历史 anime 产物 150/150 存在且 ffprobe 可读，但人工四轴仍全部未评；
- Sci-VBench evaluator 固定到 commit `e8cb3988631d0c22d31c02fc5b1cc3693bfb7031`，源码编译、import、CLI 和 150 条 rubric schema validation 均通过；
- 从作者公开逐视频结果重新聚合 MiniMax-H3 testmini：PG=2.5178、SCC=1.6311、SC=2.6156、judge LPF=3.6311、VT=0.73460；
- 通过 gpu_launcher 在空闲 034 节点启动无 LoRA 的 FL2VA 与 Ref2VA official Base 服务；两个真实 `sglang serve` 已完成 warmup，`/health` 均为 OK，不应重复提交。
- 已完成官方 T2VA、FL2VA、Ref2VA 三个 768p request：3/3 `completed`，全部为 1344×768、24 FPS、AAC 32 kHz stereo；每项保存完整轮询历史、MP4、ffprobe 和 service log。
- 已冻结 Sci-VBench 16 条分层 pilot（每个 domain 四个 prompt-length quantiles）及 raw-prompt、双 seed 协议；尚未运行 pilot judge，不把公开 SCC 结果直接当本地复现。

详细证据和 hash 见 `audits/round2_evidence.md`。

## 10. 下一轮最高优先级

1. 收集 034 两个 Base 服务的终态；ready 后运行三条官方请求，若失败则保存具体 traceback 并停止对应进程组。
2. 给官方请求增加可靠轮询、请求/响应/日志/ffprobe 落盘，不修改原始归档脚本。
3. 下载 Sci-VBench MiniMax-H3 的 released testmini 视频或一个可控子集，按 dataset card 的代码独立复核 VT/聚合。
4. 从 150 条 Sci-VBench 中先冻结 8–16 条 pilot，覆盖四 domain、不同 prompt 长度和因果阶段数；在 H3-Base 上原样生成。
5. 确认作者 judge checkpoint 的可获得性和 GPU 成本；若不可承受，先只复现 released-score aggregate 与 VT，不偷换 judge。
6. 为历史 anime 结果抽取盲评子集，但在 Sci-VBench 基线闭环前不训练新方法。
7. Round 5–31 已通过 GPU launcher 在远程 `/ssd2/jinzili` 完成 16 prompts × 2 seeds 的 raw-prompt pilot（32/32）；全部 ffprobe 通过，统一为 H.264 1344×768 24 FPS + AAC 32 kHz stereo。详细结果见 `audits/round31_pilot_generation.md` 和 `data/scivbench/pilot_full/audit.json`。这仍不是 Sci-VBench judge/VT 分数复现，不启动训练。
8. Round 32 evaluator feasibility audit：judge CLI self-test 通过，但需要外部 vLLM/Gemini multimodal endpoint；本地 VT runner 可导入，但七类 checkpoint/source asset 均缺失。见 `audits/evaluator_feasibility.md`。Evaluator Gate 尚未通过，不把 released aggregate 当作本地 pilot 分数。
9. Round 33：32 条 pilot 的执行元数据已汇总到 `data/scivbench/pilot_full/analysis.json`：四 domain 各 8 条、seed 各 16 条，服务端 inference time 75.05–75.55 s（median 75.42 s）。
10. Round 34：已在 `/ssd2` 配置隔离 VT 环境，八类 checkpoint/source asset 全部 check OK；单视频 `subject_consistency` smoke 成功，得分 0.908659、0 errors。见 `audits/round34_vt_setup_smoke.md`。这只通过 evaluator smoke Gate，尚不是六指标 VT。
11. Round 35：32 条 pilot 完成六指标 VT，32/32、0 errors、351.1 s；fresh H3-Base pilot aggregate 为 VT=0.765678（SC=0.922164、BC=0.950839、MS=0.991714、DD=0.593750、AQ=0.477314、IQ=0.658284）。见 `audits/round35_vt_full.md` 和 `data/scivbench/vt_full/`。不将其与作者 150-case released aggregate 直接等同。
12. Round 36：未发现 Sci-VBench judge endpoint；H3 服务端口不是 judge，现有 Qwen2-VL-7B 是历史 emotion judge checkpoint，不能冒充论文所需 Qwen3.5 judge。见 `audits/round36_judge_feasibility.md`。Judge Gate 暂不通过。
13. Round 37：完成 fresh VT 的 domain/seed 分层及 bootstrap CI，写入 `data/scivbench/vt_stratified_analysis.json`。Healthcare VT=0.8254、Humanities=0.7323；seed 1 VT=0.7754、seed 0=0.7560。该分析仅用于描述性 strata 检查，不做 causal claim；034 H3 服务仍 healthy，但 launcher 容器端口当前不可达，暂不盲目 stop/启动。
14. Round 40：下载并 join released `human_scores.jsonl` 与既有 `vt_scores.jsonl`，2,392/2,392 完整；pooled expert-LPF Spearman(VT)=-0.0423，VT 的人类有效性需要谨慎审计。另完成 32 条 fresh pilot 的基本音频波形审计。见 `audits/round40_human_vt_audio.md`。
15. Round 41：完成 released MiniMax-H3 150-video 六指标 VT reproduction，150/150、0 errors、3317.6 s；与 released rows 的 continuous metric median |delta| 均 <2.5e-5，VT 6.15e-6，DD exact match=149/150=99.33%，E2 reproduction Gate 通过。见 `audits/round41_e2_reproduction.md`。
16. Round 42/43：E2 通过后已在健康 034 FL2VA 启动 E3：冻结 16 prompts 新增 seeds 2–5（64 条），PID 813604；已完成 64/64、0 errors，并完成新视频 VT 评测。见 `audits/round42_e3_seed_launch.md`、`audits/round45_e3_seed_power.md`。仍不训练。
17. Round 45：96 条（16 prompts × 6 seeds）seed-power 结果：DD pairwise flip=73/240=30.42%，10/16 prompts any-flip=62.5%，VT within-prompt SD=0.051963，显著超过相邻 released-model gap 0.0049；E3 Go 条件满足。k=1..6 的 prompt-mean VT SE 为 0.02106→0.01680，未完全消除单 seed 不确定性。该结论是 metric sensitivity/power study，不是 DD 错误证明；已更正早期文档中误写的 146/480 分母。
18. Round 46：盘点远程资产确认 Wan2.1-Fun 1.3B checkpoint，并定位 VideoX-Fun T2V 入口；HunyuanImage-3 是图像模型，不能作为 VT 外部复现。见 `audits/round46_cross_model_choice.md`。
19. Round 47：Wan smoke 尚未启动；探测发现 VT venv 的 system-site-packages 与补装 transformers 造成 diffusers/huggingface_hub 兼容错误。已停止继续污染该环境，下一步建立完全独立 Wan venv。见 `audits/round47_wan_smoke_blocked.md`。
20. Round 48：已建立独立 Wan venv `/ssd2/jinzili/envs/wan_fun_clean`（Python 3.9、CUDA torch 2.8.0），并复制最小 VideoX-Fun 源码；当前阻塞在宽泛 model export 导致的非 Wan 可选模块 import 与 `cv2` libGL 依赖，尚未加载 checkpoint/生成。见 `audits/round48_wan_clean_env.md`。
21. Round 49：完成 Wan transformer checkpoint load smoke，1,564,428,608 参数成功加载；headless OpenCV 与 Wan-only imports 通过。见 `audits/round49_wan_checkpoint_smoke.md`。尚未生成视频。
22. Round 50：Wan2.1-Fun 跨模型 smoke 通过：17 frames/4 steps/seed 0 生成 H.264 832×480@16FPS 视频，同一 VT evaluator 六指标全部成功（VT=0.552052，1/1，0 errors）。这是 pipeline 连通性 smoke，步数/帧数缩减，分数不可与 H3 5s 结果比较。见 `audits/round50_wan_smoke.md`。下一步按官方默认帧数对齐后再扩大。
23. Round 51：Wan 官方协议（81 frames@16FPS=5.0625s、50 steps、TeaCache 官方推荐、seed 0）生成成功，同一 evaluator 六指标评分 VT=0.855886（单视频 sanity，不与 H3 均值直接比较）。跨模型 seed-power replication 的前置 Gate 全部通过。见 `audits/round51_wan_full_protocol.md`。
24. Round 52–53：Wan 16×6 批量 replication。v1 作废（prompt 精确替换未命中，96 条实为同一 prompt×6 seeds 重复，MD5 证实仅 6 个唯一视频；产物已隔离并标记 INVALID）；v2 修复（regex 锚定替换 + 断言）后 96/96 成功、96 个唯一 MD5、0 失败。
25. Round 54：E4 有效结果（见 `audits/round54_e4_wan_replication.md`）：seed 敏感性跨模型复现——within-prompt(seed) SD H3=0.0486 vs Wan=0.0471（均约为 released 相邻模型 gap 0.0049 的 10 倍）；Wan 的 prompt/seed 方差 ICC=0.505（各半），H3 ICC=0.670；DD flip 率 H3 30.42% vs Wan 15.83%，方向一致、幅度模型依赖。论文核心主张（VT 继承 DD 阈值化的 seed 敏感行为）获得跨模型证据。


14. Round 38：尝试利用空闲 GPU 7 运行已有 Qwen2-VL-7B 作为**独立辅助 judge**，进程约 4 分钟无输出且进入 D 状态，已停止；无分数写入。不能替代 Sci-VBench judge。见 `audits/round37_aux_judge_smoke.md`。
