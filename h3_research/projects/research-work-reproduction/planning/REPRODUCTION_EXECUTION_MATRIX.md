# 视频生成复现执行矩阵

**更新：** 2026-09-23  
**范围：** 通用/可控视频生成优先；不把数字人或私有数据探索训练作为主线。

这不是“只验六篇论文”的清单，而是当前已完成资源审计的首批九条路径。
后续论文只有在先补齐同样的资源与协议审计后才会进入队列。每一行的论文数值
都只是待验证主张；只有 C4（完整公开数据、训练、生成、评测和聚合链路）才能称为
“论文主张已复现”。C2 是作者已发布媒体上的评测回放，C3 是本地用匹配公开
checkpoint 生成后按固定协议得到的指标，二者都不是训练复现。

| 优先级 | 工作 / 公开任务 | 已闭环的最高证据 | 现在要验证的论文主张或对象 | 下一步 | 训练复现准入 / 停止条件 |
|---|---|---|---|---|---|
| P0 | **MM-Diffusion**，Landscape，DDPM-1000 + DDIM-25 | C3：两次干净重复及一次受干扰终态 checkpoint 评测；每次 6,144 媒体均通过审计 | CVPR'23 `Ours*`：FVD 117.20、KVD 5.78、FAD 10.72 | 保留 64×64 original run 1 `249.0004/15.4862/10.7107`、run 2 `231.8951/14.3733/10.9198`、run 3 `222.6008/13.1663/10.8026`；run 3 有 GPU-6 后发干扰，不能与前两次合并统计；停止再跑 | 三次都不是论文六次聚合或稳定方差，也不构成论文矛盾。ZIP 原始媒体可审查，但论文所称 928 个 YouTube 上游 URL/许可与重建 manifest 未发布，阻塞 C4 数据来源闭环。默认 DPM-Solver 有公开代码缺陷，不能冒充论文行。 |
| P1 | **T2V-Turbo**，VideoCrafter2，VBench | 独立固定种子 checkpoint 指标：4,720 视频、16 维 VBench | NeurIPS'24 四步 VC2 的 81.01 Total 及 16 个分项 | 固化当前结果、种子缺失与逐维偏差；只补公开主源/训练物料审计 | 停止训练：论文 WebVid-10M 样本/分片清单与媒体不可得，InternVideo2 Stage-2 1B f4 reward 权重访问受限。不能以新拼 WebVid 或不同 reward 替代。 |
| P1 | **VADER**，VideoCrafter2 LoRA，奖励与 VBench | 360 对固定公共提示 checkpoint 效应审计 | 论文 Table 1 的奖励增益与效率，而非仅已发布 demo LoRA | 将已发布 LoRA 与 Table 1 权重身份持续分离；只补 Table 1 的 base、seed、样本数、聚合和 checkpoint 选择证据 | 停止论文训练：这些必要字段或对应 checkpoint 身份未公开。450-step 示例只能是独立流程兼容性复现，不是 Table 1 训练复现。 |
| P0（运行中） | **MiniMax-H3**，官方 768p T2VA/FL2VA/Ref2VA 与 Sci-VBench testmini | 8×H100 SGLang Base 已完成三条官方 frozen case；作者 150 条 testmini 视频的 six-VT evaluator 回放已闭环 | 在作者发布的 150 条 testmini prompt 与逐条 H3 settings（T2VA、768/16:9、15s、24fps、50 steps、seed 20260803）下，本地 released checkpoint 的 six-VT / VT 是否接近发布值 `0.73460` | **服务已恢复且既有 runner 已续跑至 `133/150`、0 error。** 恢复服务报告 MiniMax-H3、8 GPUs、MiniMaxH3Pipeline、无 loaded adapter；第 129 条已 completed，下一条正在 queued。只等待 150/150 后审计、评分 | 无公开训练数据与训练程序，不能构成训练复现。PG/SCC/SC/LPF 的 Qwen3.5 judge 不闭环，不能以替代 VLM 填补。不得使用四卡 OOM 脚本、部分样本或不同并行方案替代本轮预注册。 |
| P2 | **VABench**，I2AV 评测 | C1：代码、媒体和部分 evaluator 权重审计 | H3 或公开基线的正式 VABench 分数 | 仅寻求作者提供图像—标注配对 manifest 和缺失 DNSMOS 实现 | 停止正式计分：配对键与官方实现缺失时，不能猜测文件顺序或以替代 DNSMOS 冒充原评测。 |
| P3 | **MotionCtrl**，VideoCrafter1/LVDM 控制 | 单个官方 camera checkpoint 部署 smoke | SIGGRAPH'24 CamMC/ObjMC/FID/FVD/CLIPSIM 行 | 如有价值，先冻结一个新的公开 camera-control 协议，且与论文表严格区分 | 停止论文训练/指标复现：407/283 case manifest、1,000 reference 视频、评测代码及论文匹配训练程序均未发布。 |
| P3 | **VideoDPO**，VideoCrafter 系偏好训练 | 官方 20,000 MP4 / 10,000 pair 媒体完整性审计 | headline OmniScore reweighted process | 仅寻找每对 OmniScore 权重和论文匹配配置说明 | 停止：权重缺失且配置中 `dupbeta: 0` 禁用 headline 方法；不训练无权重 vanilla DPO 并把它称为论文复现。 |
| P4 | **VideoCrafter2** 基座 | checkpoint 可作为共享基线 | 不是优先的预训练复现对象 | 仅为已准入下游方法保留版本、生成配置和共同 benchmark | 预训练媒体与精确管线不完整；不启动基座预训练。 |
| P1 | **JavisDiT++ / JavisDiT-v1.0**，JavisBench-mini | C2 接口证据：官方代码、released v1.0 模型/依赖、mini CSV 与 FVD/FAD cache 已 hash；4 条官方 CSV prompt 的 MP4/WAV 映射、媒体审计和官方 evaluator 输入链路已闭环 | released v1.0 在官方 240p×4s mini 协议下的 checkpoint 指标；完整 JavisBench 论文表只作报告值，不直接对照 mini | 若进入此路径，须另行预注册 1,000 条生成，保留 1,000 MP4/WAV、全量审计后再运行固定 evaluator；已有 4 条仅是接口 smoke，不可扩写为 mini 结果 | 不训练。Stage 2 TAVGBench YouTube 原媒体不可再发布，JavisBench 真值也只有 cache；可做 checkpoint C3，但不能形成 C4 原媒体/训练闭环。 |
| P4 | **AnimateDiff** / 其他新候选 | 尚未进入 claim 级审计 | 未定 | 先登记论文、代码、权重、原始训练数据、正式评测和 ≤8×H100 成本；当前本地已知材料不足以让 AnimateDiff 入队 | 任一项缺失时仅保留为文献候选，不抢占复现队列。 |

## 近期串行安排

1. MM-Diffusion run 1/6、run 2/6 与预注册 run 3/6 均已完成严格终态门槛：每次审计
   2,048 原始 MP4、2,048 SR MP4 与 2,048 WAV，并分别保留两份独立 evaluator 日志。
   run 1/2 是干净重复；run 3 虽然媒体与评测完整，但启动后 GPU 6 曾有约 37.7 GiB
   外部计算占用，因此是受干扰观察。
2. 64×64 original 的 run 1/2 值为 `249.0004/15.4862/10.7107` 与
   `231.8951/14.3733/10.9198`（FVD/KVD/FAD×10k），其描述性均值为
   `240.4477/14.9298/10.8152`、范围为 `17.1053/1.1130/0.2091`。run 3 的
   `222.6008/13.1663/10.8026` 不得并入该统计。这些不是对论文六次聚合的估计，不能
   据此判定论文对错或启动训练。
3. H3 的 034 8×H100 SGLang Base 服务已完成三条官方 frozen case；testmini 生成在
   `128/150`、`0 error` 后因服务中断阻塞。第 129 条在 denoise 约 84% 后留为 queued，
   30032 不监听。**当前优先级是取得无争用的相同 8 卡配置后恢复原服务与既有 runner，
   再闭合全量媒体审计与 six-VT。**不得以四卡 OOM 路径、不同分片、并发 benchmark、训练、
   调参或部分样本替代；结果仍须与作者发布 `VT=0.73460` 和作者媒体 evaluator 回放
   `VT=0.735706` 分开报告。

### H3 testmini 等待—验证 gate（当前 goal 的阻塞依赖）

当前 H3 目标的下一步被明确绑定到已启动的生成任务，而不是“GPU 空闲”或其他候选实验：

```text
恢复相同 8 卡 H3 Base 服务并使既有 runner 达到 150 条终态
  → 150/150 MP4 的 hash、ffprobe、request/status 与 settings provenance 审计
  → 固定 Sci-VBench six-VT evaluator 的全量评分
  → 与作者发布 VT=0.73460、作者媒体 evaluator 回放 VT=0.735706 分开比较
  → 更新公开权重验证结论
```

- 生成任务：最近确认 runner PID `2824942`（旧 PID `3124320` 已退出）；协议、数据 hash 和停止条件见
  `reproductions/minimax-h3-official-768p/SCIVBENCH_TESTMINI_VT_PREREGISTRATION_20260922.json`。
- **阻断期间**：只允许轮询资源归属、服务健康和已生成媒体完整性；不抢占外部作业，
  不启动第二条 H3 benchmark、不修改 generation settings、不训练、不以部分样本计算或报告 VT aggregate。
- **生成完成后的执行顺序不可跳过**：先全量审计，审计通过才可运行 six-VT；仅 six-VT
  可以进入本地 testmini checkpoint 对照。PG/SCC/SC/LPF 的论文 judge 未闭环，继续不报告。
4. MM-Diffusion 之后是否投入训练，取决于重复 checkpoint 指标、数据身份和正式协议
   是否闭环，不取决于 GPU 是否空闲。

### MM-Diffusion 三次终态后的固定判读

- 只有 closeout 明确写入 `audited_and_both_independent_evaluations_terminal`，且三类
  媒体均为 2,048、`ffprobe_failure_lines=0` 时，某次运行才可进入 C3 台账；任何不完整
  媒体或解码失败均保留失败证据且不计分。
- 以 `original` 64×64 的 FVD/KVD/FAD 为论文 Landscape visual row 的唯一直接比较；
  `sr_mp4` 仍是独立诊断，不能成为第二条论文表格证据。
- run 1/2 只报告逐次值、均值、范围和与论文六次聚合的差；受 GPU-6 后发干扰的 run 3
  单列展示，**不**纳入统计。三者均不据此宣布论文成立、被推翻或估计可靠方差，也不触发训练。
- run 3 的终态媒体/日志 hash 与配置已冻结。除非有新的、独立预注册理由，停止新增
  相同 checkpoint 抽样；任何协议修改只能进入另标的 compatibility lane，不能混入本行。

## 每个新候选的入队表单

进入本矩阵前，必须记录：主论文与精确表格行、官方代码 revision、权重哈希、原始
媒体/标注/切分和许可证、生成 seed/提示/采样器、evaluator 代码和权重、训练 GPU/时长。
缺一项时可做资源审计或 checkpoint smoke，但不能排入“训练复现”。所有缓存必须能通过
manifest 回到原视频、音频、图像与标注。
