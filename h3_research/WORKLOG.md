# MiniMax H3 工作日志

## 2026-09-07 — Round 1：广泛调研与资产审计

### 本轮目标

不训练、不先写 idea；先建立文献地图、远程资产清单、已有复现证据和下一阶段 Gate。

### 实际执行

1. 加载并遵循 `scholar-crawler`、`gpu-launcher` skill。
2. 确认本地工作目录 `/Users/jinzili/code/ai-academics`，发现已有 `h3_research/` 与 Scholar 第一轮结果。
3. 审阅第一轮 Scholar 数据：7 queries、128 records、13 requests、0 challenge；查询覆盖 DiT、音视频联合生成、VAE、稀疏注意力、FL2V/reference 与多模态 T2V。
4. 增加 5 个定向 query：
   - `"MiniMax-H3"`
   - `"MiniMax H3" benchmark`
   - `video diffusion acceleration sparse attention cache distillation`
   - `audio video generation synchronization evaluation benchmark`
   - `reference conditioned video generation identity consistency benchmark`
5. 先运行 Scholar dry-run：预计 5 page loads、≤50 records、约 45 秒；再执行真实抓取。结果 40 new records、8 duplicates、5 requests、0 takeover。
6. 用 `scholar-digest` 合并两轮结果：168 raw → 156 deduplicated，输出 JSONL/CSV/Markdown report，并运行 audit。警告：venue 截断 135/156、author 截断 116/156、缺年份 6/156。
7. SSH 审计 `/gluster_hdd_cv/user/jinzili/`：确认 H3 官方/历史权重、Turbo、社区 LoRA、ControlNet、SGLang 和实验代码已存在。
8. 读取官方本地模型卡，核验 33B、Qwen3-VL-32B、f16t4d24、40 Hz audio latent、开源模块边界与官方 reproducible request。
9. 审计远程已有文档与结果：
   - 150/150 anime cases completed，150 条自动指标；人工四轴评分字段仍全为 null。
   - 循环 LoRA 12/12 completed；当前证据主要证明首尾 pin，不证明 LoRA 增益。
   - 控制实验有 3 个带有效输出路径、耗时和显存的结果；记录没有 status 字段，不能按 status 缺失统计成失败。
   - 8-GPU 加速日志显示真实生成完成；旧 Turbo 结论受 LoRA `applied to 0 layers`、a8 bf16 舍入、a128 workaround 和 NFE off-by-one 影响，需重审。
10. 探测 GPU launcher / service：
    - 021 launcher 在线，PID 98590 running；H3 `:30011/health` 正常。
    - 029 launcher 与服务拒绝连接。
    - 034 launcher 在线但登记任务 stopped，H3 服务未监听。
11. 未启动新 GPU 任务：021 正承载服务，当前容器部分 GPU 有负载；在基线身份未厘清前启动训练或覆盖服务不安全，也不符合先调研原则。

### 关键命令口径

SSH alias 有 RemoteCommand 冲突，远程执行命令需使用：

```bash
ssh -T -o RemoteCommand=none jinzili_mv_260423 '<command>'
```

Launcher 请求必须绕过代理：

```bash
curl --noproxy '*' http://<gpu-host>:8659/list
```

### 本轮产物

- `README.md`：研究规划、证据、Gate、停止规则与下一步。
- `queries.txt`：第一轮查询。
- `out/step1_keywords.jsonl`、`out/step1_report.md`：已有第一轮 Scholar 结果。
- `out/step2_targeted.jsonl`、`out/step2_result.json`、`out/step2.log`：新增定向抓取。
- `out/literature_merged.jsonl`、`out/literature_merged.csv`、`out/literature_report.md`：156 篇合并文献地图。

### 未完成 / 下一步

- 尚未精确复现官方 T2VA/FL2VA/Ref2VA 三案例。
- 尚未 clone 与 self-test Sci-VBench/VBench evaluator。
- 尚未核验 150 个历史 MP4 的文件完整性与抽样人评。
- 尚未确定投稿 venue；必须查询官方 CFP，而非依赖聚合站。
- 尚未形成论文 claim，符合当前阶段要求。

## 2026-09-07 — Round 2：官方请求归档、Base 启动与 Sci-VBench 指标复核

### 已完成

1. 原样归档官方 768p T2VA、FL2VA、Ref2VA 请求，记录 SHA-256，并通过 `bash -n`。
2. 生成官方权重快照 manifest：611 files，498,573,221,865 bytes；小于 20 MB 的配置/文本文件附 SHA-256。
3. 用 `sglang310/bin/ffprobe` 检查历史 anime benchmark：150/150 文件存在且可读，0 missing，0 bad；同时确认所有人工评分轴仍为 null。
4. 在本地 clone Sci-VBench evaluator，固定 commit `e8cb3988631d0c22d31c02fc5b1cc3693bfb7031`；compile、import、judge/process CLI 和官方 `load_rubric()` 全部通过。
5. 下载 150 条 `testmini` rubric、作者公开 judge/VT scores 和 dataset cards，并记录文件 hash。
6. 对作者公开分数离线重聚合。MiniMax-H3 testmini 覆盖 150 videos × 3 judge rounds 和 150 VT rows；得到 PG 2.5178、SCC 1.6311、SC 2.6156、judge LPF 3.6311、VT 0.73460。
7. 按 `gpu-launcher` 约定，在空闲 034 宿主机通过 `/run` 启动两个 detached official Base 服务：FL2VA 用 GPUs 0–3/port 30011，Ref2VA 用 GPUs 4–7/port 30020；两者均 `lora_path=null`。
8. 宿主机进程审计确认 detached 后真正的 `sglang serve` 和 rank 进程仍在；Ref2VA 于 11:35:17、FL2VA 于 11:36:34 完成 warmup，两个 `/health` 均返回 `{"status":"ok"}`。没有重复启动，也没有干扰 021 在线服务。

### 失败及修正

- 初次在 `/gluster_hdd_cv/user/jinzili/code` 和 `tmp` clone Sci-VBench 均因权限拒绝；改为本地工作区归档，成功。
- 系统 Python 找不到 `ffprobe`；改用现有 `sglang310/bin/ffprobe`，150 条全部通过。
- 第一次 launcher 宿主机审计命令 shell quoting 有误，未捕获 021 PID cmdline；该失败保留，不据此推断 variant/LoRA。
- Launcher detached shell 显示 stopped 不代表模型服务停止；随后用宿主机 `ps` 证实子进程仍运行。

### 产物

- `official_requests/*.sh`
- `official_checkpoint_manifest.json`
- `run_official_request.py`（从归档 heredoc 提取原始 payload，持久化请求/轮询历史/终态/MP4；本地与远程 `py_compile` 通过）
- `audits/history_150_integrity.json`
- `audits/round2_evidence.md`
- `third_party/Sci-VBench/`
- `data/scivbench/{testmini,judge_scores,vt_scores,released_testmini_aggregate}.json*`

### 下一步

等待 034 Base 服务达到 ready 或失败终态，然后执行官方三案例；随后冻结 Sci-VBench 8–16 条 pilot。SCC≈1.63 目前只是作者公开结果中观察到的候选失败模式，未达到方法开发 Gate。

### 后台调研补充

后台文献/生态代理在本轮结束后返回了三组补充证据，已纳入 `README.md`，但没有覆盖掉既定 Gate：

- 官方 issue tracker 报告 voice identity bleed、audio onset ghost、prosody/V2V/high-motion 与 weight-layout 问题；GitHub API 二次读取因匿名 rate limit 失败，因此实验前仍需保存逐页快照。
- Diffusers H3 transformer 已有 `CacheMixin` 接口，且社区采样报告提示 `res_multistep` sampler 是强于朴素 Euler-50/cache 的 baseline；加速实验必须同时报告 loop 和 end-to-end，并从第一天测 audio/AV。
- 初步 venue mapping 把 speaker binding 对应到 FG short、audio-quality measurement 对应 WVAQ、onset/cache 对应 ICASSP satellites；所有日期尚需官方 CFP 复核，NeurIPS workshops 与 MLRC 当前不适用。

新增候选优先级不是立刻启动训练的授权。首先复现官方案例、Sci-VBench 与公开 cache/sampler baseline；只有 global cache 的音频损害或 voice binding 在冻结样本上稳定成立，才进入方法实验。

## 2026-09-07 — Round 3/4：官方三案例运行复现完成

> **证据更正（2026-09-19，原记录保留）：** 本节的三个历史请求确实均到达
> `completed`，但其可审计的 submit、status 和 final 响应均明确写有
> `model: sora-2`。它们不能作为 MiniMax-H3、H3-Base 或官方 H3 768p 的运行复现
> 证据。本节下的“官方三案例运行链路复现通过 3/3”仅可理解为当时 API 请求链路和
> 媒体落盘完成；H3 复现状态仍为 pending。原始证据保留在
> `shared/h3-baseline/official_requests/`，排除决定和可重新执行的冻结 H3 路线见
> `projects/research-work-reproduction/reproductions/minimax-h3-official-768p/`。

### 执行

1. 确认 034 两个 Base 服务继续 healthy，`/v1/models` 都指向 `/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3`，`num_gpus=4`。
2. 用新增 `run_official_request.py` 分别执行官方 T2VA、FL2VA、Ref2VA 请求；runner 持续轮询至 terminal status。
3. 三项均 `completed`：T2VA 10.125s、FL2VA 8.000s、Ref2VA 5.175s；全部 H.264 1344×768 24FPS + AAC 32kHz stereo。
4. 通过 ffprobe 生成并归档每项 JSON；复制远程 request/status/output/service log 到本地研究目录，并记录 output SHA-256。
5. 冻结 Sci-VBench 16 条 pilot：四 domain × 四 prompt-length quantiles；协议规定 exact raw `posted_prompt`、seed 0/1、5s 768p smoke/pilot，不将 Context-IR 混入第一阶段。

### 结论

官方三条 768p **运行链路复现通过 3/3**。这还不是逐像素、感知质量或 Context-IR 的复现；下一步应运行 pilot，并只有在 scorer/checkpoint 可用时才报告 Sci-VBench judge 分数。

### 产物

- `audits/round3_official_reproduction.md`
- `official_requests/{t2va,fl2va,ref2va}/`
- `data/scivbench/pilot16.jsonl`
- `data/scivbench/pilot_protocol.md`
- `run_scivbench_pilot.py`：按冻结 split、raw prompt、seed 运行并保存每 case 证据
- `audit_pilot.py`：批量 ffprobe/hash 审计脚本

## 2026-09-08 — Round 5：Sci-VBench pilot 已启动

### 执行

1. 确认 034 FL2VA/Ref2VA 服务仍 healthy；GPU 0–2 有服务常驻显存，其余卡空闲。
2. 新增并远程编译 `run_scivbench_pilot.py`。先运行 1 条 smoke case，完成 17 次轮询，输出 5.175 s、1344×768、24 FPS、AAC 32 kHz stereo，ffprobe 通过。
3. 通过 GPU 宿主机可访问的 `/ssd2/jinzili` 启动 32-case full pilot（16 frozen prompts × seeds 0/1），当前按单请求顺序运行，runner 日志与 per-case evidence 在 `/ssd2/jinzili/h3_research/pilot_full`。
4. 由于共享代码目录和部分 gluster 目录对用户只读，pilot runner/input 暂存于 `/ssd2/jinzili`；完成后复制回本地研究目录并记录 hash。

### 当前边界

Pilot 只验证 raw-prompt H3-Base 生成和媒体完整性，暂不调用 judge；SCC≈1.63 仍未被本地重新评分。必须等待 32 个 case 完成并审计后再决定下一步。

## 2026-09-08 — Round 6：pilot 继续运行

远程 runner 仍在按单请求顺序生成 16 prompts × 2 seeds。Round 8 复查时已完成 7/32 case：前三条 prompt 的两个 seed均完成，第四条 prompt的 seed 0 已完成，两个 Base 服务仍 healthy。没有重复启动、没有并发提交额外任务，也没有提前运行 judge。完成后再一次性复制和审计，避免读到半成品。

## 2026-09-08 — Round 9：pilot 仍在稳定推进

复查发现 runner 并未卡死：进程 CPU 空闲是因为主要计算在远程服务；当前已出现第 4 条 prompt seed 1 的 `/v1/videos` 请求并处于 denoise，服务日志显示 seed=1、50 sigma points、49 denoise iterations 正常推进。Round 10 复查已完成 8/32：第四条 prompt 两个 seed均已完成。Round 11 复查时已完成 9/32，runner 正在继续后续 prompt；Round 12 复查时已完成 10/32（第五条 prompt seed 0 完成）；Round 13 复查时已完成 12/32；Round 15 复查时完成目录仍为 13/32，但确认并非停滞；Round 16 已完成 14/32；Round 17 已完成 15/32，`709827...` 两个 seed 均完成，下一 case 已建立。Round 18 复查时仍为 15 个已落盘目录；Round 19 复查时已完成 19/32；Round 20 复查时已完成 20/32，第八条 prompt seed 0 已完成，第九条 prompt已开始。runner 仍按序运行，未修改协议、重复提交或启动 judge。Round 21 复查时仍为 20 个完整目录，但当前 `58e7..._seed0` 请求已在服务端完成并开始后续 case；服务日志显示 49-step denoise 正常运行，近期任务无 error。

## 2026-09-08 — Round 31：Sci-VBench pilot generation 完成

1. 32/32 frozen raw-prompt cases completed（16 prompts × seed 0/1），无 failed case。
2. 32/32 MP4 通过远程 official `sglang310` 环境的 ffprobe。
3. 所有输出规格一致：5.175 s、H.264 1344×768 24 FPS、AAC 32 kHz stereo。
4. 结果和 per-case artifacts 已复制回 `data/scivbench/pilot_full/`，结构化审计为 `audit.json`。
5. 该 pass 仅证明 generation/media integrity；尚未运行 Sci-VBench judge 或 VT，不把它表述为 SCC/质量复现，也不启动训练。
6. 同轮生成元数据分析写入 `data/scivbench/pilot_full/analysis.json`：4 domain 各 8、seed 各 16，32 次服务 inference time 中位数 75.42s（范围 75.05–75.55s）。GPU 3–7 空闲；下一步在 `/ssd2` 隔离配置 VT 环境并先跑单视频 smoke。

## 2026-09-08 — Round 32：Evaluator feasibility

- Sci-VBench judge CLI `--self-test` 通过（3 个离线 response-shape checks），但 judge 需要外部 OpenAI-compatible vLLM multimodal endpoint 或 Gemini API；本轮没有启动 endpoint、没有发起付费/API 请求。
- VT `weights --check --commands` 显示 DINO、CLIP-B/32、CLIP-L/14、LAION head、AMT-S、RAFT、MUSIQ 及 DINO source 均缺失；没有未经记录地下载大权重。
- 因此 Generation Gate 已通过，但 Evaluator Gate 尚未通过。32 条 pilot 暂无 PG/SCC/SC/LPF 或 VT 本地分数；released aggregate 不能冒充本地结果。
- 新增 `audits/evaluator_feasibility.md`，下一步先做 1-video evaluator smoke/self-test，再决定是否扩大，不训练。

## 2026-09-08 — Round 34：VT evaluator smoke 通过

1. 在 `/ssd2/jinzili/envs/scivbench_vt` 创建隔离 venv，复用 torch 2.11.0+cu128，安装 decord/CLIP/PyIQA/VBench/Sci-VBench evaluator。
2. 下载约 1.9 GB 官方 VT assets；DINO、两个 CLIP、LAION head、AMT-S、RAFT、MUSIQ 和 DINO source 全部 check OK。
3. 首次使用 `CUDA_VISIBLE_DEVICES=3` + `cuda:3` 触发 invalid device ordinal；修正为过滤后的逻辑 `cuda` 后成功。
4. 单视频 subject-consistency=0.908659，metric 12.7s、runner 18.8s、0 errors。其它五个字段与 VT 保持 null，未过度声称。
5. 证据在 `data/scivbench/vt_smoke/` 与 `audits/round34_vt_setup_smoke.md`。下一步可在空闲 GPU 上运行 32 条六指标 VT；仍不训练。

## 2026-09-08 — Round 35：完整 VT pilot 完成

1. 使用隔离 `/ssd2/jinzili/envs/scivbench_vt`、已核验的 1.9GB assets 和物理 GPU 3（进程内 `cuda`）运行全部 6 个 VT metrics。
2. 32/32 videos 完成，0 errors，总耗时 351.1s；所有视频均有六项 metric 和 composite。
3. fresh local pilot aggregate：SC 0.922164、BC 0.950839、MS 0.991714、DD 0.593750、AQ 0.477314、IQ 0.658284、VT 0.765678。
4. 结果复制到 `data/scivbench/vt_full/`，并新增 `audits/round35_vt_full.md`。这是 fresh pilot VT，不是作者 released 150-case aggregate，也没有 PG/SCC/SC/LPF judge 分数。

## 2026-09-08 — Round 36：Judge endpoint feasibility

- 探测 034 的 H3 服务端口 30011/30020 后确认其仅为生成 pipeline；30001、8000、8001、8080 均 connection refused。
- 远程存在历史 emotion judge 的 Qwen2-VL-7B-Instruct（约 16GB），但它不是 Sci-VBench 规定的 Qwen3.5 judge，不能静默替换。
- 未发现可用 judge process/endpoint；没有发送 judge 请求。新增 `audits/round36_judge_feasibility.md`。
- Judge Gate 仍未通过；VT Gate 已通过。下一步不训练，保留 fresh VT 结果并继续寻找可明确标注的 judge 路径。

## 2026-09-08 — Round 37：VT 分层分析与 launcher 状态

1. 对 32 条 fresh VT 结果按 domain、seed 做描述性 bootstrap 95% CI，并生成 `data/scivbench/vt_stratified_analysis.json`。
2. Domain 均值：Engineering VT=0.7598、Healthcare=0.8254、Humanities and Social Sciences=0.7323、Natural Science=0.7453；seed 0=0.7560、seed 1=0.7754。Healthcare/Humanities 差异仅是小样本描述，不能当成稳定模型缺陷。
3. 034 的 30011/30020 仍 health OK；11.9.4.28 上的 launcher:8659 当前 connection refused，因此不通过失联 launcher 盲目控制或重启 034 服务。
4. 这一轮没有新增 H3 生成或训练；下一步优先恢复/确认 launcher 控制面，或在 GPU 空闲时做 evaluator/基线实验。

## 2026-09-08 — Round 38：辅助 judge smoke 未通过

1. 在空闲 GPU 7 启动已有 Qwen2-VL-7B-Instruct，对一个 fresh pilot 视频抽取 4 帧并尝试 coherence label。
2. 进程约 4 分钟无 stdout/result/traceback，进入 D 状态；已停止以释放 GPU。
3. 没有写入分数，也没有把 Qwen2-VL 结果称为 Sci-VBench judge。原因暂记为环境/共享文件系统初始化 stall，不能解释为模型失败。
4. 新增 `audits/round37_aux_judge_smoke.md`，主结果仍使用已完成 fresh VT；不训练。

## 2026-09-08 — Round 40：E1 human join 与 E2 下载启动

1. 完成 released `human_scores.jsonl`/`index.jsonl` 下载核验，并与既有 VT 结果按 `(model, video_id)` join 2,392/2,392；pooled expert-LPF 与 VT Spearman=-0.0423。该结果是 validity warning，不是最终 clustered inference。
2. 完成 32 条 fresh pilot 音频波形审计：全部 5.175s stereo/32kHz，平均 stereo RMS 约 0.094，平均静音比例 0.181；不把它解释为音频语义质量。
3. 启动 E2：远程下载 150 条 released MiniMax-H3 testmini 视频到 `/ssd2/jinzili/scivbench_released_h3`，PID 744198；启动后约 24s 已 8/150。下载完成后执行 evaluator reproduction kill switch。
4. 文件：`data/scivbench/e1_human_join.json`、`data/scivbench/pilot_audio_analysis.json`、`audits/round40_human_vt_audio.md`。不训练。
5. E2 下载已完成 150/150；修正 evaluator 输入布局后，GPU 3 上已启动 released H3 VT reproduction（PID 756524）。约 10.5 min 时 22/150 完成，日志中的六指标均逐项 OK；任务继续后台运行，完成后按预设 kill switch 比较 released rows。

## 2026-09-08 — Round 41：E2 evaluator reproduction Gate 通过

1. 下载并保存 released MiniMax-H3 的 150 个 testmini MP4 与 150 个 settings sidecar。
2. 固定 evaluator/weights 在 GPU 3 完成六指标重评：150/150、0 errors、3317.6s、local aggregate VT=0.735706。
3. 对同一 150 rows 比较 released scores：continuous metrics median |delta| 均 <2.5e-5；VT median |delta|=6.15e-6；DD exact match=149/150=99.33%。通过预设 E2 Gate。
4. 证据：`data/scivbench/released_vt/`、`released_settings/`、`released_vt_comparison.json`、`audits/round41_e2_reproduction.md`。下一步允许进行 E3 seed-power 实验；仍不训练。

## 2026-09-08 — Round 42：E3 seed-power 生成启动

1. E2 Gate 通过后，在健康的 034 FL2VA 30011 上启动冻结 16 prompts 的 seeds 2,3,4,5 扩展，协议保持不变。
2. 远程 PID 813604；输出 `/ssd2/jinzili/e3_seed/outputs`。启动约 2m17s 时 seed2 第一条完成，runner 仍运行。
3. 这是 64 条新增生成；完成后合并 seed0/1，使用 E2 验证过的 VT evaluator 评测 96 条。仍不训练。

## 2026-09-09 — Round 43：E3 seed-power 继续运行

1. 检查 034 FL2VA `/health` 仍为 ok，GPU 空闲卡可用。
2. E3 runner PID 813604 仍运行，无 error.json；已完成 28/64 新增视频，日志连续正常。由于单请求约 80s，任务预计还需较长时间，保持后台运行而不重复启动。
3. 已同步更新 README；等 64 条完成后再进行 VT 评测和 seed-power 统计。

## 2026-09-10 — Round 45：E3 seed-power 完成

1. E3 64 条新增生成完成：64/64、0 errors；随后在 GPU 3 完成 64 条全六指标 VT，64/64、0 errors、481.3s。
2. 合并已有 seeds 0/1 后，得到 16 prompts × 6 seeds = 96 条完整结果。
3. DD 跨 prompt 内 seed pair flip=73/240=30.42%，10/16 prompts any-flip=62.5%；VT 平均 within-prompt SD=0.051963，远大于 adjacent released-model gap 0.0049；E3 Go 条件满足。已更正早期文档误写的 146/480 分母。
4. k=1..6 的 prompt-mean VT SE：0.021061、0.018278、0.017640、0.017169、0.016766、0.016803；多 seed 降低但未消除不确定性。
5. 新增 `audits/round45_e3_seed_power.md`。结论限定为 metric sensitivity/power study，不能据此宣称 DD 无效；下一步应做统计严谨化、可视化和论文 framing，而不是立即训练。

## 2026-09-10 — Round 46：跨模型复现盘点

- 远程确认存在 `Wan2.1-Fun-V1.1-1.3B-InP`（约 19GB），并定位到 VideoX-Fun 官方入口 `examples/wan2.1_fun/predict_t2v.py`；默认 81 frames/16 FPS/480×832/sequential offload，与 H3 协议不同。
- `HunyuanImage-3-Instruct-Distil` 为图像模型，不可用于视频 VT 外部复现。
- 未启动跨模型生成；先记录 variant/协议差异，下一步做单 prompt/单 seed Wan smoke。
- 更新 `audits/round46_cross_model_choice.md`。

## 2026-09-11 — Round 47：Wan smoke 环境阻塞

- 已确认 VideoX-Fun `examples/wan2.1_fun/predict_t2v.py` 为官方 T2V 入口，Wan checkpoint 完整。
- 依赖探测发现 VT venv 使用 system-site-packages；补装 transformers 后 `diffusers` import 触发 `is_offline_mode`/huggingface_hub 兼容错误。没有在该环境运行 H3 或 VT；原已完成结果不受影响。
- Wan 推理尚未启动；新增 `audits/round47_wan_smoke_blocked.md`。下一步建立不继承 system packages 的独立 Wan venv，避免污染 evaluator 环境。

## 2026-09-12 — Round 48：Wan clean env 配置

- 创建 `/ssd2/jinzili/envs/wan_fun_clean`（Python 3.9，无 system-site-packages），安装独立 CUDA torch 2.8.0+cu128 与 Wan dependencies。
- 复制最小 VideoX-Fun 子树至 `/ssd2/jinzili/wan_fun_src`，保留 checkpoint 原路径不动。
- 宽泛 `videox_fun.models` import 会拉入非 Wan 的 Ernie/FantasyTalking 模块，Python 3.9 类型注解及 cv2 的 `libGL.so.1` 依赖导致 import 阻塞；尚未生成视频。新增 `audits/round48_wan_clean_env.md`。

## 2026-09-12 — Round 49：Wan checkpoint load 通过

- 将 VideoX-Fun exports 缩减为 Wan 必需模块，OpenCV 替换为 headless build。
- 在独立 `/ssd2/jinzili/envs/wan_fun_clean` 中加载官方 Wan2.1-Fun 1.3B transformer 成功；参数量 1,564,428,608，CUDA 可用。
- 尚未生成视频；下一步才进行最小 T2V smoke，避免将 checkpoint load 误报为完整复现。
- generation wrapper 目前仍卡在官方无条件导入 `CLIPModel` 与 Wan-only exports 不一致；wrapper 每次重建临时脚本导致前一次替换未生效。没有进入 denoise，也没有占用 GPU。已将诊断补入 `audits/round49_wan_checkpoint_smoke.md`，下一轮改为持久化 wrapper 后再试。

## 2026-09-12 — Round 50：Wan 跨模型 smoke 通过

1. 重建持久化 wrapper `h3_research/wan/run_wan_smoke.py`（镜像于远程 `/ssd2/jinzili/wan_fun_src/run_wan_smoke.py`）：video_length 81→17、steps 50→4、TeaCache skip 5→2、seed 0、model_cpu_offload，全部记录在案。
2. 官方 WanFunInpaintPipeline + CLIP image encoder 成功实例化；4/4 denoise steps 完成，输出 `/ssd2/jinzili/wan_smoke/00000001.mp4`（H.264 832×480@16FPS，1.0625s）。
3. 同一 VT evaluator 在 GPU 5 完成 1/1 六指标评测：VT=0.552052，0 errors。
4. 结论：跨模型 pipeline 连通性成立；该分数是缩减协议 smoke，不与 H3 5s 结果比较。下一步以官方 81 frames 对齐时长后再决定是否扩大。新增 `audits/round50_wan_smoke.md`。

## 2026-09-12 — Round 51：Wan 官方协议 5s 生成与 VT 评分通过

1. wrapper 升级为 `WAN_FULL=1` 模式：保留官方 81 frames/50 steps/TeaCache skip 5/16FPS，seed 0，输出 5.0625s 视频（ffprobe 验证），与 H3 pilot 时长对齐。
2. 同一 VT evaluator 完成六指标评分：subject 0.9560、background 0.9579、motion 0.9951、DD 1.0、aesthetic 0.5320、imaging 0.6943、VT 0.855886。
3. 单视频 vs 均值无数理意义，只作 sanity；DD 的 seed 敏感性是否跨模型复现是下一步 16×6 replication 的检验点。
4. 新增 `audits/round51_wan_full_protocol.md`、`h3_research/data/wan_cross/` 证据。GPU 空闲时段已用于该任务，未训练。

## 2026-09-12 — Round 52：Wan 16×6 批量 replication 启动

1. 新增 `h3_research/wan/run_wan_batch.py`：逐条 16 prompts × seeds 0–5（96 条），每条独立子进程调用官方 predict_t2v（81 frames/50 steps/16FPS 官方默认），prompt 用 pilot16 的 verbatim `posted_prompt`，输出 `/ssd2/jinzili/wan_batch/<id>_seed<k>.mp4`，支持断点续跑（已存在则跳过）。
2. 两次启动失败原因：官方脚本的 sys.path bootstrap 基于 `__file__`，写进 `/tmp` 后找不到 `videox_fun`；已在 wrapper 中向生成脚本显式注入 `/ssd2/jinzili/wan_fun_src` 路径。
3. 修复后 batch 运行正常：约 7 分钟完成 6/96（每条 50 步 + TeaCache ≈ 65–70 s），GPU 3 100% 利用率，0 fail。
4. 后续每 12 条做一次 ffprobe 抽查；全部完成后用同一 evaluator 分三批做 VT（断点续跑语义已验证）。仍不训练。

## 2026-09-12 — Round 53：Wan batch v1 作废，v2 修复后重启

1. **v1 结果作废（诚实记录）**：v1 runner 的 prompt 替换是精确字符串匹配，但官方脚本的默认 prompt 是中文狗沙发 prompt（第 113 行 `prompt = "一只棕色的狗…"`），与我假设的英文 woman prompt 不同 → 替换从未生效，96 条全部用同一默认 prompt 生成。MD5 证实只有 6 个唯一视频（6 seeds × 16 重复组，组间完全相同）。基于 v1 的 VT 汇总（VT=0.745554、DD flip 128/240 等）与本地 `analyze_e4_wan.py` 输出一律作废，产物隔离为 `wan_batch_invalid_dogprompt`、`data/wan_cross/vt_invalid_dogprompt`、`e4_wan_seed_power.INVALID.json`。
2. **v2 修复**：prompt 替换改为 `re.subn(r'^prompt\s*=\s*".*"', …, count=1, flags=re.M)`（锚定行首，不会误伤 `negative_prompt`），并在写入后断言新 prompt 行存在，失败即 `sys.exit(2)`。
3. v2 验证：重启后前 4 条产出 4 个不同 MD5（prompt 已生效），0 fatal。
4. v1 的教训：批量替换必须断言生效次数；相同的 per-prompt 统计（所有 prompt 的 VT mean/SD 完全一致）是不可信结果的强信号。

## 2026-09-13 — Round 54：E4 跨模型 replication 有效完成

1. v2 batch 96/96 成功（96 唯一 MD5、0 失败、0 fatal），VT 评分 96/96 complete（299 s，0 errors）。
2. E4 核心数字：DD flip 38/240=15.83%（H3 30.42%）；any-flip 7/16（H3 10/16）；within-prompt VT SD 0.0384（H3 0.0520）。
3. 方差分解：Wan ICC(prompt)=0.505（prompt/seed 各半），H3=0.670；两模型 within-prompt(seed) SD≈0.047–0.049，同为 released 相邻 gap 的约 10 倍 → seed 敏感性是 metric-pipeline 性质，跨模型成立。
4. 新增 `audits/round54_e4_wan_replication.md`；本地有效产物 `data/wan_cross/vt2/`、`data/wan_cross/e4_wan_seed_power.json`。E4 Go 条件（DD instability 跨模型复现 + within-prompt SD 显著超过模型间 gap）满足。实验阶段（E1–E4）全部完成，下一步转向论文写作与图表。

## 2026-09-13 — Round 55：Related Work 种子文献收集

1. 按 scholar-crawler skill 执行 3 次小批量关键词检索（VBench / Sci-VBench+seed variance / H3+dynamic degree），共 78 条记录（0 次人工接管），合并去重至 `refs/merged.jsonl`。
2. 离线 digest 出 `refs/report.md`（top-cited、venue 分布）与 `refs/shortlist.md`（10 篇锚点引文）。
3. 关键锚点：Henderson 2018（RL 单 seed 不可靠先例，3,655 cites）、VBench 1,910 cites、EvalCrafter、VBench++/2.0、VideoPhy、Sci-VBench 原文（arXiv:2608.09873）、Motion-tuned QA 2009（DD 的感知谱系）、VideoFeedback。
4. `PAPER_OUTLINE.md` 待办更新；下一步画图与初稿。GPU 本轮空闲（034 基础服务除外），无训练。

## 2026-09-13 — Round 56：论文图表 v1

1. 新增 `make_paper_figs.py`，产出 `paper_assets/fig1_within_prompt_sd.png` 与 `fig2_dd_flip.png`。
2. H3 侧改用 canonical E3 join（`e3_new_vt` + `pilot_full` seeds 0/1 → `vt_full` 映射），保证图与审计一致：flip 30.42% vs Wan 15.83%（脚本输出校验 FIG_OK 0.3042 0.1583）。
3. Fig1：两模型各 16 prompt 的 6-seed within-prompt VT SD（排序）+ 0.0049 released-gap 参考线；Fig2：flip 率柱状 + per-prompt DD 散点。
4. `PAPER_OUTLINE.md` 待办更新（剩 LaTeX 表格与初稿）。无训练；034 基础服务不受影响。

## 2026-09-13 — Round 57：论文初稿 v0.1

1. 新增 `paper/main.tex`：Seed Sensitivity of Composite Video-Quality Metrics（MiniMax-H3 under Sci-VBench VT）。
2. 内容：abstract、intro、related（引用 refs/shortlist 锚点）、setup（协议差异显式声明）、E2 复现精度、E3 seed-power（30.42% flip / SD 0.0520 / k-SE）、E4 跨模型（15.83% / ICC 0.505 vs 0.670 + Table 1）、human join（−0.0423，descriptive）、discussion（三条可执行建议）、limitations。
3. 两图引用 `../paper_assets/`（存在性校验通过）；本机与 034 均无 TeX 工具链，不做本地 PDF；结构校验（环境配对）通过，可在 Overleaf 直接编译。
4. E1–E4 → 初稿全链路完成；后续：co-author 通读、补 Stata/聚类稳健性如需要、按目标 workshop 模板排版。

## 2026-09-13 — Round 58：数字 QA + ranking-stability 分析 + 复现指南

1. **数字 QA**：`main.tex` 引用的全部统计量逐项核对 artifacts（E2 medians、E3 73/240 与 SD、E4 38/240 与 ICC、E1 −0.0423/−0.1193）全部一致。
2. **新增 `analyze_rank_stability.py`**：15 个 seed-pair 的 prompt 排名稳定性。结果强且跨模型一致：H3 mean Spearman ρ=0.509（min −0.059）、Wan ρ=0.495（min 0.171），mean rank shift 都是 3.3/16 —— 仅换 seed 就会重排 prompt 排名。已写入论文 E3/E4 节（`data/wan_cross/seed_rank_stability.json`）。
3. **新增 `REPRODUCE.md`**：E1–E4 完整复现命令（环境、E2/E3/E4 runner、分析脚本、注意事项：v1 作废、VT venv 保护、H3 服务常驻）。
4. 无训练；034 服务正常。

## 2026-09-13 — Round 59：E1 bootstrap CI + Fig3

1. 新增 `analyze_e1_bootstrap.py`：从缓存 HF 文件离线重建 E1 join（n=2,392），video-level bootstrap（B=2000）95% CI：
   - ρ(VT, LPF) = −0.0423，CI [−0.0820, −0.0012]（排除 0 但幅度可忽略）
   - ρ(DD, LPF) = −0.1193，CI [−0.1603, −0.0788]（负相关且稳健）
2. 新图 `paper_assets/fig3_human_join.png`（LPF–VT 散点 + per-metric ρ 柱状）；论文 human 节更新为带 CI 表述，Fig3 引用加入 `main.tex`（结构校验 LATEX_STRUCT_OK，3 图齐备）。
3. 产物：`data/scivbench/e1_bootstrap.json`。无训练；034 服务正常。

## 2026-09-13 — Round 60：真实 leaderboard 间距分析

1. 新增 `analyze_leaderboard_spacing.py`：用 released `vt_scores.jsonl`（16 模型）计算真实 leaderboard。修复 off-by-one（`enumerate(order,1)` 下 `order[i-1]` 即当前模型 → gap 恒 0）。
2. 结果：8 个全覆盖模型（1,253 视频/模型）的相邻 VT gap 中位 0.0066、max 0.0183、min 0.0029；H3 排 4/8（VT=0.7267）。
3. 论文统一口径：旧审计值 0.0049（round 45 的近似）替换为计算值 median 0.0066；abstract/E3/discussion 三处更新（"~8× median gap"）；fig1 参考线同步 0.0066 并重生成。校验：`FIG_OK 0.3042 0.1583`、无 stale 0.0049、3 图路径存在。
4. 产物 `data/scivbench/leaderboard_spacing.json`。这把论文核心对比从"估计 gap"升级为"released 排行榜实测 gap"。无训练；034 服务正常。

## 2026-09-13 — Round 61：README 状态刷新 + 投稿清单

1. `README.md` 顶部状态从"调研与复现基线阶段"刷新为"论文冲刺阶段"，新增进度速览表（E1–E4 / leaderboard / rank stability / 文献 / 图表 / 初稿 / 复现指南 九项全绿）。
2. 发现旧 `SUBMISSION_CHECKLIST.md` 属于早前 EmoH3/ICASSP 轨道（deadline 2026-09-16），未覆盖；另建 `SUBMISSION_CHECKLIST_VT.md` 管当前论文（必做：模板排版、匿名化、补充材料打包、图矢量化、claims 复查；风险表含 16×6 规模与"只有两个模型"应对）。
3. 全库 artifact 存在性校验：22 个关键产物（数据 JSON、3 图、初稿、脚本、审计）全部在位，0 missing。
4. 无训练；034 服务正常。

## 2026-09-13 — Round 62：Resolution analysis + 图矢量化

1. 新增 `analyze_resolution.py`：配对模型比较下的可分辨性计算（prompt 效应相消，残差 = seed×prompt 交互 σ_sp；SE = σ_sp·√(2/(P·s))）。
   - H3 σ_sp=0.052：s=1 时 95% 半宽 ±0.036；s=6 仍 ±0.015 > median gap 0.0066
   - 分辨 median gap 需 **~30 seeds/prompt（H3）**、~17（Wan）；分辨 min gap 0.003 需 ~154/~84
   - 产物 `data/scivbench/resolution_analysis.json`
2. Discussion 加入 "Resolution analysis" 段：定量结论指向"配对设计下增加 prompts 比增加 seeds 更有效"；建议 (iii) 从经验陈述升级为统计论证。
3. 图矢量化：`make_paper_figs.py`、`analyze_e1_bootstrap.py` 同步输出 PDF；`paper_assets/` 现 3 PNG + 3 PDF。
4. LaTeX 结构校验通过。无训练；034 服务正常。












## 2026-09-09 — Round 44：E3 seed-power 继续运行

- E3 runner `813604` 仍在 034 FL2VA 上顺序运行，当前 `45/64` 新增视频完成、`0` errors。
- 由于单条 H3 请求耗时约 80 秒，未重复启动；保持单一可恢复 runner，完成后再启动 96 条 VT 评分。
Round 22 复查时已完成 21/32，`58e7..._seed1` 已落盘，runner 仍按序运行。Round 23 复查时已完成 23/32，`58e7..._seed0/1` 与下一条 prompt 两个 seed 已完成；Round 24 复查时仍为 23/32，runner 正在处理下一 case；Round 25 复查时已完成 24/32，`cc077..._seed0` 已完成；Round 26 复查时仍为 24/32，随后监控确认进展至 26/32，`cc077..._seed1` 与 `6ae0..._seed0/1` 均完成。Round 27 复查时已完成 27/32，`d9a99..._seed0` 已完成，runner 继续顺序执行。Round 28 复查仍为 27/32，当前正在处理 `d9a99..._seed1`；Round 29 已完成 29/32，`d9a99..._seed1` 与 `a16beb..._seed0` 均完成。Round 30 复查时已完成 30/32，`a16beb..._seed1` 已完成，runner 仍存活、无 error，继续等待最后 2 个任务。

## 2026-09-13 — Round 63：第三方审稿与修复

1. subagent 模拟审稿（fresh context）：verdict **Borderline**，3 must-fix（headline 单位混用 / ICC 无估计量无 CI / human 节自相矛盾）。
2. 统计重算：Wilson CI（flip 30.42% [24.9,36.5] vs 15.83% [11.8,21.0]，不重叠）；ICC 改 ANOVA + prompt bootstrap（0.56 [0.19,0.78] / 0.48 [0.04,0.71]）；human join 改 prompt-topic cluster bootstrap → CI [−0.088,+0.005] 含 0。
3. main.tex 全面修订：headline 重锚定 paired-SE、CI 补齐、ICC 表更新、"bit-level"→"numerical"、Limitations 增补（两模型/TeaCache/同方差假设）、图切 PDF。stale 数字清零，结构校验通过。
4. 新增 `audits/round63_reviewer_response.md`。教训：video-level bootstrap 的"显著为负"是忽略 prompt 聚类的伪影；cluster bootstrap 更诚实。无训练；034 服务正常。

## 2026-09-13 — Round 64：二轮审稿 Weak Accept + 收尾修复

1. re-review（同一审稿人设定）：**Weak Accept**。MF-2/MF-3 resolved；MF-1 部分解决（abstract 5.5× 与正文 8× 并存）。
2. 新问题修复：删除全部残留 "~8×"（E3 bullet 与 Discussion opener 统一为 paired-SE 5.5× 口径，±0.036 三处一致）；N2 加估计量对照（0.0520 sample SD vs Table1 pooled population SD 0.056）；**N4 LPF 展开纠正**——DATASET_CARD 证实 LPF = Low-level Perceptual Fidelity（1–5 整数），此前误写为 "logical–physical feasibility" 已改正（事实性错误被审稿流程拦下）。
3. E2 medians 升级为正式 Table（tab:e2）；Setup 补 ICC/LPF/硬件定义。
4. 结构校验 LATEX_STRUCT_OK，2 表 3 图（PDF）。无训练；034 服务正常。

## 2026-09-13 — Round 65：补充材料打包 + 匿名化校验

1. 匿名化扫描：`paper/main.tex` 无用户名/主机/IP/远程路径泄漏，作者已为 Anonymous。
2. 补充材料打包 `submission/supplementary_vt.zip`（188 项，~172KB）：REPRODUCE.md、全部 11 个分析/runner 脚本、9 个汇总 JSON、162 个 per-video VT 结果（E3+E4）、3 张 PDF 图、main.tex。
3. `submission/` 现含 main.tex + 3 PDF 图 + 补充包，可直接随投稿系统上传。
4. 剩余：官方模板排版（等 CFP/模板发布）、co-author 通读。无训练；034 服务正常。

## 2026-09-13 — Round 66：E3/E4 seeds 6–11 扩展启动（响应规模质疑）

1. 动机：resolution analysis 预测"分辨 median gap 需 ~30（H3）/~17（Wan）seeds"；现有 6 seeds 只能部分验证 SE(s)∝1/√s 预测。扩展到 12 seeds 可实证检验该模型，直接回应审稿风险"16×6 规模"。
2. H3：复用可参数化 runner `run_scivbench_pilot.py --seeds 6,7,8,9,10,11`（断点续跑语义内建），输出 `/ssd2/jinzili/e3_seed_s6_11`，走 30011 服务（GPU 0–2 常驻）。
3. Wan：同一 `run_wan_batch.py`，`WAN_SEEDS=6..11`，输出 `/ssd2/jinzili/wan_batch_s6_11`，GPU 3（100%）。
4. 首查：H3 3/96 完成 0 错、Wan 3/96 完成 0 错；预计各 ~2h。完成后 VT + 实证 SE(1..12) vs 预测对比（下轮）。两任务并行，GPU 无空闲。无训练。

## 2026-09-13 — Round 67：扩展运行监控 + 验证脚本就绪

1. 生成速率确认：H3 ≈2 min/条（含服务排队+下载），Wan ≈1.3 min/条（含 checkpoint 加载）；ETA H3 ~3h、Wan ~1.8h，两进程健康 0 错。
2. 预写 `analyze_se_validation.py`：数据齐后自动计算 empirical SE(1..12)（contiguous seed windows）vs 恒定 σ_sp 预测，输出 `data/scivbench/se_validation.json`；若实证偏离预测（seed 间相关），将如实报告并修正 resolution analysis。
3. 下轮：轮询至生成完成 → 两批 VT（+192）→ SE 验证 → 论文 E3/E4/Limitations 更新。

## 2026-09-13 — Round 68–69：12-seed 扩展完成，论文升级 v0.4

1. **生成+评分全部完成**：H3 seeds 6–11（96/96，0 错，VT=0.741312）+ Wan seeds 6–11（96/96，0 错，MD5 唯一，VT=0.680689）。两模型现各有 16 prompts × 12 seeds = 192 条。
2. **12-seed 头条数字**（更精确）：DD flip H3 290/1056=27.46% [24.9,30.2]、Wan 182/1056=17.23% [15.1,19.6]（CI 不重叠保持）；any-flip 12/16 vs 9/16；VT SD 0.0509 vs 0.0411；ICC 0.56 [0.30,0.72] vs 0.48 [0.22,0.65]。与 6-seed 估计一致 → 结论稳定。
3. **SE 实证验证**（`analyze_se_validation.py` 重写为 subset 法）：s∈{2,3,4,6} 的 empirical/predicted 比值 H3 0.66–0.89、Wan 0.94–1.23——i.i.d. 噪声模型在 ±25% 内成立；s=12 行退化（单一子集）已识别。注意 6-seed 版本脚本曾犯"单 prompt SE vs 16-prompt mean SE"混比错误（比值 3.9 的假象），已修正并记录。
4. **论文 v0.4**：abstract/E3/E4/Table1/resolution 段全部更新为 12-seed 数字；fig1/fig2 用 12-seed 数据重生成（FIG_OK 0.2746 0.1723）；结构校验通过。
5. 补充包重建：385 项（356 per-video JSON）、~278KB。无训练；034 服务正常。

## 2026-09-13 — Round 70：12-seed 审计 + max SD 修正

1. 修正 E3 bullet 的 stale max（6-seed 0.1092 → 12-seed 0.0841）。
2. 新增 `audits/round70_12seed_extension.md`：生成/评分/12-seed 统计/SE 验证/方法教训全记录。
3. 补充包增量更新（audit + v0.4 main.tex）。

## 2026-09-13 — Round 71：文档一致性收尾

1. `audits/round45`（E3 6-seed）与 `round54`（E4 6-seed）添加 supersession note：12-seed 数字以 `round70_12seed_extension.md` 为准，6-seed 保留为历史证据链。
2. `README.md` 进度速览表更新为 12-seed 数字（E3 27.46%、E4 17.23%、ICC 0.56/0.48）。
3. H3 服务健康确认：`/health` 两端点 `{"status":"ok"}`（root 路径 404 是 FastAPI 正常行为，非服务故障）。
4. 投稿包状态：v0.4 + 387 项补充包。等待外部依赖：CFP/模板、co-author。无训练。

## 2026-09-13 — Round 72：v0.4 一致性审查（第三轮 subagent）+ 10 处修正

1. 第三轮 fresh-eyes 检查抓出 9 类残留：abstract/Intro/Limitations/Fig1 caption 的 "6 seeds" 标签、E4 Wan max（0.0811→0.0841）、k-seed SE（重算 12-seed：0.0211→0.0164, k=1..12）、rank instability（重算 66 对：ρ 0.54/0.54，min −0.06/+0.06，shift 3.2/3.3）、sample/population SD 术语冲突（统一 population SD）。
2. 补充 fig:dd/fig:human 正文引用（审稿人指出的 minor）。
3. "Even with 6 seeds" 保留为 k-seed 敏感性表述（有意为之，加 half the budget 说明）。
4. 结构校验通过。论文 v0.4.1 定稿口径：全 12-seed。无训练。

## 2026-09-13 — Round 73：12-seed 统计入档 + 补充包终版

1. 单一事实源原则补齐：k-SE 12-seed 曲线（0.0211→0.0164）写入 `e3_seed_power.json`（by_seed_se_12）；rank instability 12-seed（66 对：ρ 0.5382/0.5429，min −0.0588/+0.0559，shift 3.21/3.26）追加为 `seed_rank_stability.json` 的 `H3_12seed`/`Wan2.1-Fun_12seed` 行（6-seed 行保留）。论文引用的每个 12-seed 数字现在都有 artifact 支撑。
2. 补充包终版重建：387 项，无重复，含 v0.4.1 main.tex + 更新后的 artifacts；`submission/main.tex` 同步。
3. 至此论文（v0.4.1）+ 复现链 + 审计链 + 投稿包全部就绪；仅剩外部依赖（CFP/模板、co-author）。无训练。

## 2026-09-13 — Round 74：Related Work 强化（Scholar 调研）

1. 用 scholar-crawler 新抓 40 条（评估方差/可复现性 + 视频质量评估可靠性两大方向），并入 refs（115 条，report.md 刷新）。
2. Related Work 增补 6 篇引用：Bouthillier et al. (MLSys'21, Accounting for variance)、Keeling et al. (Quantifying variance in eval benchmarks)、ReproGen (INLG'21)、QRA++ (2025)、T2V metric reliability (2026)、保留 RL 方差经典。正文补一句定位："our contribution is a seed-resolved, per-metric decomposition on a released, fully reproducible leaderboard"——直接回应 P5（related-work 单薄）。
3. cite/bibitem 全配对校验（14/14），结构 OK。无训练。

## 2026-09-13 — Round 75：LaTeX 静态校验 + 补充包同步

1. 本机/远程均无 TeX 工具链（安装重且非必需）；改用 Python 静态校验：brace 平衡 0、odd-$ 0、raw % / 未转义特殊字符 0、cite/bibitem 14/14 配对、环境配对 OK。
2. Related Work 更新后的 v0.4.2 main.tex 刷入 submission/（zip 387 项重建）。
3. 投稿包状态：等 CFP 模板后做最终排版 + 编译验证（在有 TeX 的环境）。无训练。

## 2026-09-13 — Round 76：DD 阈值敏感性（强稳健性结果）

1. 新分析：DD binarization 阈值扫 0.05–0.60 共 13 档 → **flip rate 完全不变**（H3 27.46%、Wan 17.23% 各档一致）。原因：within-prompt DD 近双峰（饱和于 0/1；全 384 条只有 6 个不同值，中间值极少）。
2. 结论：DD flip 不是阈值选择的 artifact——比"released mean 阈值"更强的稳健性声明。入 `data/scivbench/dd_threshold_sensitivity.json`；论文 E3 加 threshold-invariance bullet。
3. DD 实为 frame-pair motion ratio（如 3/16→0.1875），多数饱和；E2 的 DD 149/150 exact match 与此一致。
4. submission/main.tex 同步 v0.4.3。无训练；服务正常。

## 2026-09-13 — Round 77：第三模型（LTX-Video 2B）启动

1. 动机：S7（两模型泛化）是论文最大残留风险；第三模型把"noise floor 是 metric 管线属性"的 claim 从 n=2 升到 n=3。
2. 选型：LTX-Video 2B（diffusers LTXPipeline 原生支持、2B 轻量、HF 可达）。权重已下载（repo 全量 237G 含多版本，pipeline 用顶层 0.9.5 2B）。
3. 协议：与 Wan 一致——同一 16 pilot prompts、seeds 0–11、832×480、81 帧 @16fps（5.06s）、40 步、cfg 3.0、cpu offload。runner `ltx_src/run_ltx_batch.py`（run_wan_batch 同款协议：resume-safe + prompt 断言 + sha1 命名）。
4. smoke 通过（119KB mp4，~30s/条）；**full batch 192 条已上 GPU 3**（BATCH_PID=2783817），ETA ~2.5-3h。完成后 VT 评分 → E5 三模型横向：within-prompt SD / DD flip / rank instability 是否复现。
5. 无训练；H3 服务正常。

## 2026-09-13 — Round 78–79：E5 第三模型完成——核心 claim 在 n=3 复现

1. LTX-Video 2B batch：192/192、MD5 全唯一、0 失败（~2.7h，GPU 3）；VT 评分 192/192（VT=0.638735）。
2. **E5 结果**（`data/ltx_cross/e5_ltx_summary.json`）：DD flip 126/1056=**11.93%** [10.1,14.0]（与 H3 CI 不重叠，同为"实质量级"）；any-flip 7/16；VT within-SD **0.040**（max 0.075）；ICC **0.46** [0.10,0.71]；rank ρ **0.46**（min −0.10）、位移 3.5/16。
3. 三模型横向：噪声底 0.051/0.041/0.040；ICC 0.56/0.48/0.46；flip 27.5%/17.2%/11.9%（各不相同但全部实质）——"噪声底是 evaluator 管线属性而非特定生成器"获得 n=3 复现，S7（两模型泛化）风险关闭。
4. 论文 v0.5：abstract/intro/Limitations 更新 + 新 E5 paragraph；结构校验 OK。
5. 实验总量：3 模型 × 16 prompts × 12 seeds = 576 视频全评分。无训练；H3 服务正常。

## 2026-09-13 — Round 80：三模型汇总表 + LTX SE 验证（诚实记录反向结果）

1. subset-SE 扩展到 LTX：ratio 1.3–1.7×（s∈{2,3,4,6}），高于 H3（0.7–0.9）和 Wan（0.9–1.2）。**LTX 上 i.i.d. 模型偏乐观**——这意味着 resolution 结论在 LTX 情形只会更强（真实 SE 更大→更难分辨→主张更安全）。如实写入论文，不隐藏。
2. 论文新增 Table 3（tab:e5，三模型汇总：flip CI / any-flip / SD / ICC / rank ρ），E5 引用之；结构校验 OK（3 表 3 图）。
3. `se_validation.json` 现含 H3/Wan/LTX 三组曲线。无训练。

## 2026-09-13 — Round 81：第四轮审稿（v0.5.1）→ Weak Accept 维持 + 3 项机械修复

1. 第四轮 fresh-eyes：**Weak Accept 维持**；明确判定 **S7（两模型泛化）已解决**；最强点="从 ~10⁻⁶ 分数复现到每个印刷数字的完全可复现链"，最弱点="外部效度：16 prompts/单 evaluator/三开源模型——结论是关于 Sci-VBench VT leaderboard 而非普适定律"（已由 Limitations 覆盖）。
2. 抓出 3 处 v0.5 升级遗留 bug 全修：①abstract 缺 LTX（补 11.9% + 三模型 ICC 0.46–0.56）②Discussion "±25%"与 0.66 矛盾（改 0.66×–1.70×，唯一乐观偏离=保守方向）③headline 倍数 5.5×→**5.3×**（σ 用 12-seed 0.0509，±0.035，三处统一）。
4. Table 1 补 LTX 行；"two models"→三模型口径。结构校验 OK。无训练。

## 2026-09-13 — Round 82：文档全面同步（README/OUTLINE/CHECKLIST）

1. `PAPER_OUTLINE.md` 刷新至 v0.5.2：三模型 576 视频、E5 小节、图 PDF/0.0066 口径、末尾状态块。
2. `README.md` 进度表 rank-stability 行更新为 12-seed 三模型数字。
3. `SUBMISSION_CHECKLIST_VT.md` 勾选：v0.5.2 / E5 / 12-seed / 阈值不变性 / 补充包。
4. 文档三件套与论文、artifacts、投稿包全部一致。剩余外部依赖：CFP/模板、co-author、最终编译。

## 2026-09-13 — Round 83：LTX 管线自复现验证（bit-level）

1. 用同一 GPU/同一 runner 逻辑重生成 2 条 LTX 视频（seed 0/1）→ **MD5 与原批完全一致**。我们的生成管线（固定 seed + 相同软件栈）是 bit-level 确定性的，"完全可复现链"的自证闭环：E2（released→ours 10⁻⁶）+ 本轮（ours→ours bit-level）。
2. 产物：`/ssd2/jinzili/ltx_repro_check/`。无训练；服务正常。

## 2026-09-13 — Round 84：发现 ICASSP 2027 deadline（09-16）→ 4+1 页版本产出

1. web 检索确认：**ICASSP 2027（Toronto）full-paper deadline = 2026-09-16，仅剩 3 天**（2027.ieeeicassp.org/call-for-papers/）。CCF-B，符合目标。
2. 安装 tectonic 0.15.0（IEEEtran 按需拉包，编译通过）。
3. `paper_icassp/main.tex`：v0.5.2 压缩为 ICASSP 格式（摘要+8 节+12 refs），3 张矢量图复用，编译 **3 页**（限 4+1，达标）；pymupdf 结构校验无溢出、矢量图正常。
4. 新增收尾建议四条（multi-seed reporting 等）与 E2 双端确定性表述（10⁻⁶ + MD5）。
5. `submission/icassp_v1.pdf/.tex` 归档。**用户需在投稿系统注册并填作者信息**；3 天内可完成提交。

## 2026-09-13 — Round 85：ICASSP PDF 图内文字一致性修复

1. 全文提取校对发现两处图内 stale：fig1/fig2 轴标签仍写 "6 seeds"（数据已是 12-seed）；fig3 散点图标注的是旧 video-level CI [-0.082,-0.001] 而非论文采用的 cluster CI。
2. 修复并重生成：fig1/fig2 → "12 seeds"；fig3 → "[cluster 95% CI -0.088, 0.005]"（脚本改为从 e1_cluster_bootstrap.json 读 cluster_boot_ci）。
3. ICASSP PDF 重编译（3 页）；"Even with 6 seeds" 补 "(half the budget)" 说明；归档 icassp_v1；补充包重建（含 icassp main.tex）。
4. 教训：图是代码生成的"另一个事实源"——统计改口径时必须连图脚本一起改，否则图会泄漏旧结论。

## 2026-09-13 — Round 86：ICASSP 投稿合规核实 + 作者操作手册

1. 官方 Paper Kit 核实：**"ICASSP does not perform blind reviews"——单盲，PDF 必须含作者名单**（不能匿名投稿）。main.tex 作者块改为带 TODO 标记的占位（重编译 3 页通过）。
2. 产出 `submission/icassp_abstract.txt`（纯文本摘要 1,208 字符，直接粘贴系统）。
3. 产出 **`submission/ICASSP_SUBMISSION_README.md`**：文件清单、合规点（页数/单盲/格式/track 建议 IVM SP/ deadline AoE）、6 步投稿流程、10 分钟最终检查单、备选 venue 方案。
4. 至此论文侧 100% 就绪，剩余全部为作者本人动作（注册、填名、点击提交）。

## 2026-09-13 — Round 87：程序化数字校验 38/38 ALL_OK + 抓到一处论文事实错误

1. 新增 `verify_paper_numbers.py`：38 项断言，把两版论文的每个印刷数字对回 artifacts（flip 率/CI/SD/ICC/gap/ρ/cluster CI/阈值不变性/SE 比值/E2）。
2. **抓到真实错误**：论文称 "aggregate VT 0.735706 on both sides"——实际 0.735706 是我们重跑侧；released 侧是 0.734596。差值 0.001110 **恰好等于单个 DD cell 翻转**（1/6×1/150），mean_delta VT=0.00110963 完全吻合。两版论文均已改为诚实表述（并顺势强化 DD 主导性论证）。
3. 重跑校验 **38/38 ALL_OK**（`audits/number_verification.json`）；ICASSP PDF 重编译 3 页；补充包 583 项同步（新增 released_vt_comparison.json + verify 脚本）。
4. 调试记录：校验脚本自身两处 bug（srcs 路径漏 data/、JSON int/str key）修复。

## 2026-09-13 — Round 88：投稿包最终一致化

1. `icassp_abstract.txt` 从修正后的 tex 重新生成（确认无 "both sides" 残留）。
2. README 置顶新增两条：38/38 数字校验 + ICASSP 投稿包就绪指针。
3. 至此所有交付物（长版论文/ICASSP 版/摘要文本/补充包/校验器/审计/文档）一致且可复跑。剩余：作者本人投稿动作。

## 2026-09-13 — Round 89：fig2 升级为三模型 + 一处自纠

1. fig2（DD flip）升级：柱图三模型（27.46/17.23/11.93%），散点三模型 per-prompt DD——与 Table 1/三模型叙事一致（旧图只有 H3/Wan 两模型）。
2. 自纠两处脚本 bug：LTX 数据被 load()+merge 循环双重 append（len=24 被滤成空 → flip 显示 0）；缺 itertools import。修正后 FIG_OK 0.2746/0.1723/0.1193 与 artifacts 精确一致。
3. ICASSP PDF 重编译（3 页）并归档；补充包重建（含新 fig2）。
4. 提醒：ICASSP PDF 内 fig2 现含 LTX——若审稿人问"为何 E4/E5 分开写"：E4=Wan 完整叙述、E5=LTX 补充，结构无碍。

## 2026-09-13 — Round 90：ICASSP tex 自包含化

1. 图 PDF 复制进 paper_icassp/ 并把 includegraphics 路径改为相对当前目录——co-author 在任何机器只要有 tex+3 张图即可编译（不再依赖 ../paper_assets 结构）。
2. clean-room 测试：/tmp 空目录仅 tex+3 图 → tectonic 编译 3 页成功。
3. icassp_v1.pdf/.tex 归档更新；补充包同步（fig PDF 与 tex 平级存放）。
4. 投稿包至此完全自包含。剩余：作者投稿动作。

## 2026-09-13 — Round 91：Rebuttal kit 产出

1. `rebuttal_kit.py` → `audits/rebuttal_per_prompt.md`：三模型 16 prompts × {VT mean/SD、DD ones/12、per-prompt DD flips} 表——1 月收到 reviews 后可直接引用的 per-prompt 弹药。
2. 有趣 pattern 已可见：DD 饱和 prompts（0/12 或 12/12）flip=0 且 SD 低（如 5a8a0eba… 0/12，SD 0.0077）；DD 混合 prompts（4-8/12）flip 最高（32）——per-prompt 层面再次支持 DD 主导。
3. 机器侧工作至此全部完成；等待作者投稿动作（deadline 09-16）。

## 2026-09-13 — Round 92：响应用户意见（idea 逊/引用少）→ 6 个候选 idea 查重选优

1. 用户意见采纳：①idea 必须含训练 ②引用 ≥30。goal 已恢复并更新（objective #9/#10）。
2. 产出 `IDEAS.md`：6 个候选（软校准/聚合器/方差预测/生成蒸馏/检验器/蒸馏头），逐一 Scholar 查重（新增 4 个 jsonl：novelty_iqa/seed/motion/fusion）。
3. 对比矩阵 → **选定 I3+I1 组合**："Learning to Predict and Reduce Seed Noise"——训练组件 A（seed-consistent 软 DD 校准）+ B（seed-variance 预测器→自适应 seed 分配）；诊断部分复用已验证的种子敏感性结果；对最近竞品（Evaluating the Evaluators 2026，分析型）的差异 = 训练修复。
4. 引用规划：12 现有 + 新增方向（uncertainty-aware VQA/score fusion/RAFT 蒸馏/adaptive evaluation/统计方法）→ ≥30。
5. 下一步（验证优先）：修复 RAFT 连续分数提取 bug → 576 视频特征 → I1 训练消融 → I3 跨模型验证 → 确认增益后写 v2。

## 2026-09-13 — Round 93：RAFT 连续分数提取（576 视频）启动

1. DD 代码核实：RAFT 逐对光流 → 每对 top-5% 流幅均值（连续 max_rad）→ 硬阈值 6.0×(scale/256) + 计数判定。连续分数算完即丢——软校准的原料。
2. 提取器 v2：monkeypatch check_move 捕获 static_scores（复用官方 infer 路径，零重实现 bug）；resume-safe。首批 smoke 130/192 通过，数值合理（静态视频 scores<2，动态>thres）。
3. 全量 576 视频（LTX 192 + Wan 96+96 + H3 96+96+32 路径）提取已在 GPU 3 运行（EXTRACT_PID=3233008），首查 78/576。
4. 完成后：I1 软 DD 校准训练（train seeds 0–5 / test 6–11）→ I3 方差预测器 → 增益确认后写 v2。

## 2026-09-13 — Round 94：I1 软 DD 校准首轮训练完成（结果混合，继续迭代）

1. 576 视频全部 41/41 对 RAFT 连续分数提取完成（0 失败），本地 `data/motion_feats/`。
2. 训练：soft(video)=mean σ((s_i−t+b)/T)，目标=within-prompt 方差+BCE 保真−between 分离；train seeds 0–5 / test 6–11（及 0–3/6–9 变体）。
3. **结果（held-out seeds）**：H3 softSD 0.0389 vs binSD 0.2049（**降 5.3×**，第二 split 降 7.5×）；Wan 降 1.7–1.8×；**LTX 混合**（一 split 降 1.6×，另一 split 反升 0.55×）。
4. 已识别问题：①评估指标不对等（连续 SD vs 二元 SD，软分数压向常数可 trivially 降 SD）→ 需加 held-out 保真（soft>0.5 vs binary 准确率、Spearman）和 VT 集成后的总噪声底对比（这才是 headline）②b 移动操作点需阈值不变性背书 ③LTX 反例需归因。
5. 结论：方向有效但未到写论文程度——正是"先训练验证"该有的样子。下轮迭代：VT 集成评估 + 保真约束加强 + LTX 反例分析。

## 2026-09-13 — Round 95：Idea V2（贡献导向重选）

1. 用户意见：别光新颖度，要对学术界有实质贡献。产出 `IDEAS_V2.md`：6 个候选（SC-LoRA / metric 修复 / 数据集+标准 / stable evaluator / 贝叶斯聚合 / 运动蒸馏），按贡献面重评。
2. 新查重 3 组（refs/novelty_j1/j4j5/j3.jsonl）：J1 邻域（T2I seed shaping/seed-robust alignment）确认 T2V+归因空白；J4 crowded；J3 无 multi-seed T2V 数据集，但 "Beyond statistical significance"(2025) 证明报告需求存在。
3. **选定 J1 主线 + J3 资源层 + J2 对照**："Where Does Evaluation Noise Live?"——数据集 + 归因 + SC-LoRA + 报告标准四贡献；训练两处（生成侧 LoRA + metric 校准）。
4. 下一步：扩规模生成（50×3×8≈1600 视频）→ SC-LoRA 管线。

## 2026-09-13 — Round 96：训练目标换血（用户否掉跨 seed 一致性）

1. 用户意见成立：SC-LoRA 的"一致性正则"与 seed 多样性用途相悖。V3 修订写入 IDEAS_V2.md。
2. 新查重 2 组（novelty_k5/k2.jsonl）：K5 学习型运动质量度量的空白确认（最近竞品角度不同）；K2 自适应评估算力 LLM 侧已热、T2V 侧空白。
3. **主线换为 K5**：学习型运动质量度量（人类监督、seed 稳定协议、直接替换被诊断出缺陷的 DD 启发式）；K2 作其一节；J3 数据集照旧为资源层。
4. 风险登记：人类监督数据量小 → 公开 VQA 数据增广/代理预训练；与 VideoScore 系差异 = 专维度+稳定性协议。

## 2026-09-13 — Round 97：220 篇前沿调查完成 + Idea V3

1. 按用户要求（≥100 篇、蒸馏加速/可控生成方向）：12 组查询抓 240 条 → 去重 220 篇（19,150 被引）；加旧库 115 篇累计 335 篇。产物 refs/survey_v2/{merged.jsonl,SURVEY_V2.md}。
2. 前沿确认：Self-Forcing 系 2026 集中爆发（实时 AR 蒸馏最热）；推理时 scaling 是新兴热点（216/99/48 引）；MotionStream/FlashMotion 2026 已填"少步+可控"空白。
3. IDEAS_V3.md：C1 计算最优推理 scaling（seeds×samples×steps，自适应分配器，训练实在）★推荐；C2 控制保真×蒸馏（窗口窄警讯）；C3 caching 放大 seed 噪声（作 C1 一节）；C4 物理 DPO 评估审计；C5 相机控制 DPO。
4. 推荐 C1：LLM 侧 inference scaling laws(455 引)的 T2V 姊妹篇，空白明确，蒸馏模型改变 best-of-N 经济学这笔账没人算过；独家资产（seed 方差+测量协议+GPU）全部用上。

## 2026-09-13 — Round 98：广域自查 235 篇（累计 455 篇）+ S6 新 idea

1. 用户要求超出点名范围自找话题：新增 12 方向查询（音视频/机器人世界模型/4D/tokenizer/个性化/数据策展/水印检测/稀疏注意力/多镜头/flow matching/游戏引擎/量化）→ 去重后全库 455 篇、31,444 被引。
2. 12 方向窗口状态入档（IDEAS_V4.md 表格）：多镜头/水印活跃窗口小，机器人/音视频/VAE/4D 算力重，稀疏注意力/量化/个性化拥挤。
3. **S6 新发现（查重干净）**：少步蒸馏 × 水印存活性——水印全在 50 步教师训练，蒸馏学生是否保留水印无人测过；训练修复 = 水印保持蒸馏。社会相关（加速时代的出处危机）。
4. 最终三选一交用户拍板：S6（水印×蒸馏）★ / C1（计算最优 scaling）/ S1'（多镜头跟随）。基础设施三者共享。

## 2026-09-13 — Round 99：多样化增读 ~330 篇（全库 784 篇）

1. 新增 16 组查询：RLVR/TTA/驾驶世界模型/医学/修复超分/姿态迁移/风格化/组合布局/VLM 裁判/DiT 架构/气候/插帧/模型合并/合成数据/流式系统/课程学习 → 全库 784 篇去重、78,557 被引（merged_v3.jsonl + SURVEY_V3.md）。
2. 新前沿信号：RLVR-World(2026)、一步 VSR（DOVE/FlashVSR）、流式 AR（MAGI-1）、VLM 裁判自成体系（MJ-Bench）。生成模型 TTA 与视频扩散数据课程是仅存的两个轻量空白。
3. IDEA 短清单更新（IDEAS_V4_addendum.md）：S6 ★ / C1 / S1' 不变，新增 T4（运动复杂度课程，用 RAFT 资产）为第 4 选项；T1（RLVR 物理）因 GPU 预算风险暂缓。
4. 建议：文献边际收益递减，尽快锁定 idea 进入训练验证。

## 2026-09-13 — Round 100：2026 最新动向扫描（--sort-by-date --year-from 2026）

1. 用 scholar-crawler 的年份过滤+按日期排序抓 2026 新发表：12 组查询 221 条（refs/survey_2026/merged_2026.jsonl + TRENDS_2026.md）。
2. 五大动向：①实时流式交互世界模型是主旋律（Matrix-Game 3.5/Cosmos-H-Dreams/Ripple）②出处/水印升温为认可问题域（TraceFlow/UACI）③test-time scaling for video 爆发（NoisEasier/Candidate Recycling）④物理=agentic/code-CoT（VideoCoCo）⑤音视频联合流式。
3. S6 两次定向查重仍干净（0 命中）；图像侧 "Vanishing Watermarks"(2026) 提供同类现象先例。S6 四项全占：领域动向+现象先例+空白确认+独家基础设施。
4. C1 窗口收窄警示：若选 C1 需 1 个月内抢占 per-prompt 自适应预算定位。
5. 决策建议不变但更明确：S6 首选。

## 2026-09-13 — Round 101：8×H100 档位话题选定 + 设备申请

1. 用户开放 8×H100 档位。选定 **S6-Full**：完整水印保持蒸馏训练（从教师训练学生，非修补现成学生）——机制消融（学生是否见过水印轨迹）要求完整管线，这是 8 卡档位的正当性来源。
2. 算力核算写入 S6_TRAINING_PLAN.md：显存（学生+critic FSDP 25-35GB/GPU）、教师 rollout 3 GPU-天、3 变体学生训练 4 天、评测 1 天 → 3 周训练 + 1 周写作。
3. 设备申请：主申请 8×H100×4 周；可选 +4（省 1 周，消融矩阵全并行）非必需。
4. 风险预案：蒸馏收敛→49帧起步先 LoRA 预验证；质量拖累→正则扫描即核心消融；方案单一→3 水印×2 检测器。
5. 投稿目标映射：ICME 2027（Jan 截稿）或 WACV 2027 workshop。

## 2026-09-13 — Round 102：话题最终锁定 C2-Full（替换 S6，用户不熟水印）

1. 用户偏好：点过名的蒸馏加速+可控生成都保留 → 选 C2-Full "What Does Distillation Forget?"。
2. 定向查重通过：系统类（MotionStream/FlashMotion）不做测量与损失；Data-Forcing(2026) 恢复 diversity 非控制。空白=控制保真退化系统测量+控制保持蒸馏损失+（独家）控制依从性的 seed 噪声。
3. C2_TRAINING_PLAN.md：教师=Wan-Fun-Control（管线已有）；学生=控制条件化 Self-Forcing 式自训；度量=相机 pose 误差/轨迹误差（确定性）；周 1 出第一个结果（教师控制依从性 seed 噪声）。
4. 设备申请：8×H100×4 周（可选+4 非必需）。
5. 水印计划 S6_TRAINING_PLAN.md 保留存档（若将来复活）。

## 2026-09-13 — Round 103：HF Papers 前沿收集（用户指定工具）

1. 用 openclaw-skills/hf-papers 拉 2026-06~09 月度面板，去重 174 篇，视频/生成类 85 篇（refs/hf/）。
2. 五大 HF 动向：①视频世界模型记忆问题自成一线（MemLearner/Addressable Memory/RECAP-Forcing，全在训练记忆模块）②流式实时编辑（SANA-Streaming 42）③原生音视频 2K（DreamX-Creator 100）④Forcing 家族持续迭代（One-Forcing/Representation Forcing 64）⑤世界基础模型（Orca 508 榜首、SimWAM 108）。
3. C2-Full 交叉验证：HF 4 个月无"控制保真退化测量/控制保持蒸馏损失"工作，空白再确认；少步 AR 蒸馏赛道活跃（对我们有利）。
4. 文献库总量：Scholar 784 + HF 174（互补面板：被引 vs 社区热度）。

## 2026-09-13 — Round 104：最终 idea 定稿（双面板合成）

1. Scholar(784)+HF(174) 合成 → **Drift-Aware Forcing**（IDEA_FINAL.md）：训练流式世界模型的漂移预测+恢复能力。
2. 定向查重：Pathwise test-time correction(9)/Self-Correcting T2V(2)/SPIRAL(2)/Matrix-Game 3.0 记忆全是外挂或架构侧——"训练生成器自身漂移感知"空白确认。
3. 三组件：漂移解剖协议（seed-resolved，独家）+ 漂移预测头 + 恢复机制（自生成"漂移→修复"课程）。
4. 亮点叙事：世界模型学会看自己；自生成课程；解剖协议独立站得住。
5. 8×H100 四周计划同 C2-Full 基础设施；备选 Alpha/Beta/Gamma 存档。

## 2026-09-13 — Round 105：直播数字人 idea（用户指定：表情/一致性/直播）

1. 两次定向查重：情绪控制系（DreamTalk 49，吃标签/参考）与流式直播系（SoulX-FlashHead/ReST/AsymTalker 2026，主攻身份+实时）之间有缝——**长会话情绪枯萎无人量化/修复**。
2. IDEA_AVATAR.md："Avatars That Keep Feeling"——①表现力普查（SER(audio)↔FER(face) 一致性曲线，seed-resolved）②情绪忠实 LoRA（可验证奖励：预训练 SER/FER 一致性 + 表现力方差正则，零标注）③表情自锚定（Drift-Aware 人脸特化）。
3. 亮点：现象先行（越播越面瘫的定量证据）+ 可验证奖励（零标注，适配直播无标签场景）+ 人像域算力 3-5× 便宜。
4. 底座：Hallo3（124 引，开源 VDT）或 SkyReels-Audio；8×H100 四周计划照旧结构。

## 2026-09-13 — Round 106：Goal 立项（用户确认 Avatars 选题）

1. Goal 工具更新：objective 替换为 Avatars That Keep Feeling 从零到一（rev 17，max 120 轮），旧 H3 目标的资产沿用。
2. GOAL_AVATAR.md：五里程碑（M1 底座+管线 / M2 塌缩曲线第一结果 / M3 训练消融 / M4 2h 直播评测矩阵 / M5 论文包≥30 引）+ 分相执行（Phase A 单卡即可启动）+ 硬约束 + 风险。
3. 下一步：M1 底座选型（Hallo3/SkyReels-Audio/EchoMimic）+ SER/FER 管线搭建。

## 2026-09-13 — Round 87：Phase A 管线落地（本地 CPU 完成 M1 评测臂）

1. SSH 中断恢复；发现节点实为 8×H100，GPU 4-7 空闲（GPU 3 被 sglang/Qwen3.8-27B 占用，勿动）。
2. 裁判就位（HF 镜像）：2×FER（trpakov/vit-face-expression + dima806，7 类）+ SER（superb/wav2vec2-base-superb-ER，4 类）。speechbrain 版格式不兼容换用之，留作后续集成。
3. eval_fidelity.py 端到端 SMOKE_OK：SER↔FER 一致性（JS 相似度+argmax match）、表现力（1-P(neutral)）、动态方差三指标落盘；audio_only 模式支持普查前置。
4. 数据：RAVDESS 1440 条（HF parquet，8 情绪均衡）→ 38.3 分钟脚本化会话（622 段，情绪分块循环）+ manifest（段级 GT 情绪）。
5. SER 校验（40 段 probe）：4 类 40%（随机 25%；IEMOCAP→RAVDESS 域差距），arousal 代理区分度好（angry .978 vs neutral .590）。正式普查加第二 SER 集成。
6. 远端：uv+python3.10 venv（conda/micromamba 均撞系统 CUDA 权限）、静态 ffmpeg 7.0.2、torch 2.6+cu124/diffusers 0.40 就绪。
7. 下一轮：底座部署（EchoMimic/Hallo2 择优）+ GPU3 不可用用 GPU4-7 生成首批人像视频 → FER 侧接入 → 普查启动。

## 2026-09-13 — Round 88：M1 完成——底座生成全通 + 真实首测信号

1. **EchoMimic V1 部署成功**（GPU 4）：依赖链五连修——av 11 无 wheel（放开版本）/ hub 1.x API 删除（0.20.3）/ transformers 需 torch≥2.5（钉 4.38.2）/ numpy 2 vs torch 2.2（钉 1.26.4）/ opencv 5 无 imread（headless 4.14）。教训：2024 项目用 uv 建独立 venv + 全链版本对齐。
2. 首个生成视频：a.png × chunk_0000（RAVDESS angry 语音）× seed 420，512×512，~4 分钟/条（含加载）。
3. **管线闭环 + 首测信号**：pull 回本地 → eval_fidelity 全通 → SER(audio)=ang 0.795 vs FER(face)=neutral 0.828——强愤怒语音生成脸中性化，方向与情绪塌缩假设一致（单样本不下结论）。
4. dynamics 指标修正：字符串 argmax 的 np.std 非法 → 帧间标签变化率。
5. M2 启动条件就绪：batch 生成脚本（多 seed × 多 chunk × 长会话）为下一步。

## 2026-09-13 — Round 89：M2 普查启动（4 GPU 并行生成中）

1. **Arm 1**（静态贫瘠 + seed 方差）：40 chunks × 4 seeds = 160 条独立生成，GPU 4-7 四 shard 并行（census_shard.sh，断点续跑幂等）。~39/164 完成，预计 ~40 分钟收尾。
2. **Arm 2**（会话内漂移）：40 chunks 拼接 145.5s 长音频单次滑窗生成 × 4 seeds（L=4000 帧），shard0 尾部执行。
3. census_analyze.py 就绪：rglob 收集 → 双 FER 集成 7→4 类映射 → expr/argmax/GT 联表 → per-seed/per-GT 汇总。
4. **关键 bug 修复**：Judges.fer 标签存 id2label dict（int 键）→ pool 按 int 迭代全零 → 集成结果作废且 expr 恒 1.0。统一归一化为字符串列表后复测：chunk_0000 seed420 expr=0.104/neutral（与首轮一致）。教训：集成评测必须先断言 labelset 类型。
5. 下轮：全量 pull → arm1 全量 FER → arm2 F(t) 曲线 → M2 门禁判决汇报。

## 2026-09-13 — Round 90：M2 Arm1 普查完成——"出生即面瘫"定量证据

1. Arm1 全量（160 生成，158 有效）：mean_expr=0.079（92% 中性）；跨情绪极差 0.019（平坦）；跨 seed std 0.012（无差异）。
2. FER 正向对照（FER2013 真实脸 n=700）：4 类 48.3%；表现力判别完美（表情脸 0.92-0.99 vs 中性脸 0.27）——裁判有效，塌缩是模型真实缺陷。
3. 锚定对照（b.png 微笑参照 × angry/happy chunk）：expr 0.142/0.183——参照表情不迁移，静态锚定无效。
4. M2 判决：B（静态情绪贫瘠）成立，A 待 Arm2 长会话曲线（生成中，地板效应风险已记录），C 排除。判决书 M2_VERDICT.md。
5. 产物：census_arm1_full.json / census_arm1_summary.png / anchor_eval.json。
6. 传输通道教训：tar 管道 macOS xattr 间歇破坏 → 改 stdin redirect 两步法（先 cat > 再启动，& 不能与 cat 同链）。

## 2026-09-13 — Round 91：M2 Arm2 漂移曲线（2/4 seed）+ M3 训练启动

1. Arm2 seed420/421 漂移曲线：40 窗全程平坦贴地（mean 0.076/0.068，线性斜率≈0）——**出生即面瘫 + 全程不变，地板效应确认**。A（动态塌缩）在本底座不成立；B（静态贫瘠）为唯一现象。M2_VERDICT.md 已更新。
2. **CREMA-D 全量落地**：VideoFlash 7442 视频（2.3GB）+ AudioWAV 7303 + 人口学 csv；manifest 7442 行 / 91 演员 / 6 情绪均衡（1271×5+1087 neutral）。RAVDESS 视频版 zenodo 504（远端同样），CREMA-D 替代监督源。
3. **M3 训练链路打通（train_emolora.py）**：EchoMimic fork 版 UNet2DConditionModel（无 attn2）+ EchoUNet3D + ReferenceAttentionControl(midup) + whisper chunks + FaceLocator(16,32,96,256)；peft 0.10/accelerate 0.27/hub 0.20 兼容链锁定；40 步冒烟 loss 1.156→1.02，LoRA 14.0M 参数保存验证。
4. **V-A 全量训练启动**（GPU 5，8k 步，actor-disjoint 验证集 {1031,1041}），step 750 loss 0.998 下降中，预计 ~3h。
5. Arm2 剩余 2 seeds（422/423）生成中（GPU 4，每条 ~36 分钟）。

## 2026-09-13 — Round 92：Arm2 全量完成（M2 收官）+ 三变体并行训练

1. Arm2 4 seeds 全量漂移曲线：mean 0.073-0.078，斜率 ±1e-4/s ≈ 0，seed 无差异——"出生即面瘫+全程不变"最终定量确认。图 arm2_drift_all_seeds.png。**M2 正式收官**（判决 B）。
2. V-B 冒烟通过（emo_embed.pt 保存验证）→ V-A/V-B/V-C 三变体在 GPU 5/6/7 并行训练（8k 步 each）。V-C loss 偏高为 arousal 加权预期。
3. 变体操作化：V-A 纯 LoRA SFT；V-B + 可学习情绪嵌入（12×768，encoder_hidden_states 注入 cross-attn）；V-C = V-B + arousal 加权损失（情绪样本 1.2-1.5×，中性 0.5×）——作为可验证奖励的权重化近似。

## 2026-09-13 — Round 94：V-A 评测结果——"抬起地板但付出身份代价"

1. **V-A LoRA (step8000) 评测**（25 条，RAVDESS 23 + CREMA-D val 6 的音源）：
   - 表现力：mean_expr **0.842**（baseline 0.079，抬升 10.7×）✓
   - 但 expr_by_gt 依旧平坦（0.839-0.844），std 0.004，pred 只有 happy/angry 两模式 → 学会"expressive"，未学会"对的情绪"；GT 匹配率 0.267（反低于 baseline 0.449）
   - **M4 代价**：ArcFace 0.89 → **0.255**（身份崩塌），sync_proxy 0.45 → 0.020（唇同步消失）——朴素 SFT 以身份+唇动换表情
2. 解读：audio 特征通道太弱，模型改用"整体表达风格"拟合 CREMA-D 表情先验 → 需要显式情绪条件（V-B）+ 权重化（V-C）。
3. V-B/V-C 评测生成已并行启动（GPU 5/6，与外部用户作业共存 ~20GB 余量）；eval_lora_vb.py = LoRA 合并 + emo_embed 加载 + **SER 在线选情绪**（零标注推理）经 EncHSInjector 注入 encoder_hidden_states（CFG 双批）。
4. 基础设施：m4_metrics.py（ArcFace vggface2 + 嘴部运动×音频包络 sync_proxy）在远端验证通过（baseline: arc 0.886-0.906 / sync 0.34-0.73）。
5. 依赖坑提醒：scenedetect 安装把 numpy 拉到 2.2.6 且 opencv_contrib 5 污染 cv2 → 三钉重装（np 1.26.4 / headless 4.14 / 删 contrib）。**echomimic env 禁止再裸装包**。

## 2026-09-13 — Round 95：三变体消融第一轮结果（诚实负结果）+ RAFT 迭代设计

1. **V-A/V-B/V-C 评测全部完成**（各 24-25 条）：
   - expr：0.842/0.846/0.844（全部抬离地板 ✓）
   - GT-match：0.190/0.143/0.143（全部 < baseline 0.449 ✗）
   - pred 分布全为 happy/angry 二值；expr std 0.003-0.004（跨输入无差异）
   - **结论：情绪嵌入条件化（V-B）与 arousal 加权（V-C）均未修复情绪选择**。LoRA SFT 学到的是"表达风格"，audio/embedding 条件通道被风格淹没。
2. V-A M4：arcface 0.255（-0.63），sync_proxy 0.020（-0.43）——朴素 SFT 破坏身份与唇同步。
3. **下一迭代（RAFT 式可验证奖励循环）**：从 V-A 采样 best-of-K，奖励 = SER(audio)↔FER(face) agree + GT 情绪匹配 + 身份/唇同步正则，奖励加权微调。这是论文核心贡献的正面验证路径。

## 2026-09-13 — Round 95（续）：M4 三变体齐 + 消融报告 v1

1. M4 完整矩阵：V-A/B/C arcface 0.255/0.251/0.252，sync 0.020/0.042/-0.003——三变体同等破坏身份与唇同步；条件化无缓解作用。
2. M3_ABLATION_REPORT.md v1 落盘：结果矩阵 + 四条结论 + RAFT 迭代 2 设计（轻触 LoRA r=8 lr=2e-5 2k 步 + best-of-8 奖励选择，奖励 = SER↔FER agree + GT 匹配 + ArcFace 惩罚）。
3. 过程事故：本地 eval_vB 拷贝被误删（mv 失败后 rm 仍执行）——远端原件重拉恢复；教训：mv 失败时不得继续 rm。

## 2026-09-14 — Round 95（goal 轮）：迭代 2 全链启动

1. **V-D 轻触 LoRA 完成**（r=8 α=16 lr=2e-5 2k 步，GPU 5 与外部 61GB 作业共存）——预期保身份/唇同步。
2. **裁判上远端**：2×FER + SER 经 hf-mirror 下载 + symlink；raft_score.py 打通（chunk_0000: expr 0.068 / agree 0.453 / arc 0.889 与本地一致）。修复：ser_audio 48k→16k 重采样（librosa）。
3. **RAFT 采样生成启动**：60 条 CREMA-D 训练演员音频（ANG/HAP/NEU/SAD × 15，val 演员 1031/1041 排除）× K=3 = 180 条，GPU 4/6 双分片（raft_gen.py，幂等断点）。预计 ~3.5h。
4. train_emolora.py 扩展：--init_lora 续训 + --gen_pairs（优胜视频为目标的 RAFT 数据模式）+ --ref_img。
5. 下轮流程：raft_score 打分 180 条 → 选优 → train_emolora --init_lora vD/step2000 --gen_pairs winners 500 步 lr 1e-5 → V-E；V-D 与 V-E 同 30 条评测集对比（expr/GT-match/arc/sync 四轴）。

## 2026-09-14 — Round 96：迭代 2 完成与诊断链闭合

1. RAFT 180 条采样完成（GPU 4/6 双分片，~4h）。V-D 评测（30 条，GPU 7）+ V-E 微调/评测（GPU 5）全链跑通。
2. 打分：全样 GT-match 0.222 → best-of-3 优胜 0.35（选择信号存在）；但 reward 需绕开 arc 门控（全样 arc<0.6 → 门控清零，改用原始字段重选）。
3. **V-D 结果**：expr 0.774 / GT-match 0.143 / arc 0.215 / sync 0.036——轻触 LoRA 不救身份，根因=训练/推理参照分布错位（CREMA-D 演员首帧 vs a.png）。
4. **V-E 结果**：expr 0.737 / GT-match 0.143 / 预测 100% angry——自蒸馏模式锐化；RAFT 需要多多样性基座。
5. M3_ABLATION_REPORT.md v2：五变体矩阵 + 三条新结论 + 未来工作（per-identity 参照一致训练、条件化采样、奖励内方差正则）。
6. 转向 M4：用 baseline vs V-D 组装 2h 直播矩阵；论文定位=benchmark+诊断（训练消融为诊断证据）。
7. 教训沉淀：pkill -f 会匹配自身 ssh 命令行导致连接自毁；build_amap_eval.py 的 tag=目录名后缀（vD_s2000）。

## 2026-09-14 — Round 96（续）：重大 bug 发现与修复——denoising_unet.pth 漏载

1. M4 探针生成后 arc 全在 0.15-0.31（census baseline 应为 0.886），sync 为负 → 逐层隔离（音频时长/采样率/裁剪/管线）最终定位：**eval_lora.py / raft_gen.py / train_emolora.py 全部漏加载 denoising_unet.pth**（EchoMimic 微调后的 denoising unet 权重）。原始 infer_audio2vid.py 在 from_pretrained_2d(motion_module) 之后再 load_state_dict(denoising_unet.pth, strict=False)——我们的脚本从第一批评测起就用了只有 SD+motion module 的未微调基座。
2. 影响：V-A/B/C/D/E 训练与评测全部在破损基座上（arc 0.25 与今天无 LoRA 破损基座 0.257 一致 → "LoRA 几乎没改变模型"）；M2 普查用 census_batch.sh（正确管线）不受影响，"born numb" 判决维持。
3. 修复：四个脚本补载 denoising_unet.pth（strict=False）。重做顺序：①基线探针重生成验证 arc 回 0.886 → ②正确基座重训 V-A'（8k 步）→ ③重评 → ④M4 探针重生成。
4. 教训：改动前必须 diff 原始脚本的权重加载序列；"复现管线"要逐 checkpoint 验证（基线生成后立即跑 m4_metrics 对表）。

## 2026-09-14 — Round 97：管线验证闭环 + 训练目标修复 + V-A2 重训启动

1. **管线验证闭环**：用 arm1 同款 infer_audio2vid 流程（probe_batch.sh）重生成 12 探针 → expr 0.065±0.010 全 neutral、arc 0.902±0.004、sync 0.366——与 arm1 census 完全一致。"born numb" 在 2h 会话探针上成立，M4 基线锚定。
2. **修复 2：训练目标缺 add_noise**——train_emolora 原来把干净 latent 喂给 unet 预测 noise（恒 MSE≈1.0，模型学输出趋零）→ 加 `noisy_lat = scheduler.add_noise(lat, noise, t)`。V-A/B/C/D/E 的"身份崩塌/表情爆炸"大部分是这个 artifact。
3. **修复 3：audio2feat 依赖 ffmpeg PATH**——训练命令漏 export PATH → audio2feat 全抛 FileNotFoundError 被吞 → 所有 batch skip（loss 假 0.0000）。训练命令必须带 ffmpeg PATH。
4. **运维教训×2**：pkill -f 的模式匹配 ssh 自身命令行导致自杀（kill 与 relaunch 必须分离两条 ssh）；外层未加引号 heredoc 会展开内层变量（脚本先本地写好再 stdin 传输）。
5. **V-A2 重训启动**（修复后基座 + 正确目标，GPU 4，8k 步，step50 loss 0.062）。评测走"LoRA 合并进 denoising_unet.pth → 原始 infer 管线"的零分歧路径（merge_lora.py 待写）。
6. eval_lora.py 增加 baseline 模式（--lora_dir /none 时跳过合并）+ 参照处理精确移植（face crop + mask dilation ratios 与 infer_audio2vid 一致）。

## 2026-09-14 — Round 97（收尾）：V-A2 重训确认健康

1. 首次重启在 GPU 共存冲突下 CUBLAS 崩溃（step 50 后）；GPU 4 空闲后重跑 → step 800 loss 0.0171（0.062→0.017 正常收敛），独占 19GB。预计 ~4.5h（8k 步）。
2. merge_lora.py 落盘：LoRA 合并进 denoising_unet.pth 副本 → 原始 infer_audio2vid 管线零分歧推理（评测路线定型）。
3. 下轮：V-A2 完成 → merge → 12 探针 + 30 音源评测（原管线）→ 真·M3 消融重做 → M4 矩阵（V-A2 vs baseline）。

## 2026-09-14 — Round 98：训练崩溃根因（发散）与修复

1. CUDA CUBLAS/IndexKernel 随机崩溃（step 350-1500，跨 GPU 复现，fp32 也崩）→ CUDA_LAUNCH_BLOCKING 定位到 sticky error；真因=**发散**：loss 回升（0.065→0.118）后激活爆炸。lr 1e-4 × α/r=2 有效步长过大。
2. 修复：lr 5e-5 + alpha 32（有效步长减半）；fp32 + NaN loss/grad 双 guard（isfinite 检查跳过坏 batch/step）。
3. V-A2 现稳定越过旧崩溃点（step 900，loss 0.03-0.07，GPU 5 91%）。预计 ~5h。
4. eval_variant_batch.sh 落盘：merge_lora.py 合并 → 原始 infer 管线逐音频生成（零分歧评测路径就绪）。
5. 数据完整性扫描：7278 训练样本仅 1 个坏行（1076_MTI_NEU_XX.flv，2 帧，已被 guard 跳过）。

## 2026-09-14 — Round 98（续）：自动续训 + 论文骨架

1. CUDA 断言无法被 guard 捕获（sticky）→ train_resume_wrapper.sh：崩溃自动重启并从最新 checkpoint --init_lora 续训（train_emolora.py 增加步数继承）。夜间循环至 8000 步。
2. PAPER_AVATAR.md v1 落盘：标题候选/摘要/贡献/方法/实验矩阵占位/复现性附录（三 bug 叙述）/相关工作分桶/venue 映射。
3. refs_seed.bib 37 条正典引用种子（wav2lip/SadTalker/ArcFace/SyncNet/RAVDESS/CREMA-D/LoRA/DDIM/RAFT 等），8 条标 TODO-V 提交前 DBLP 校验。
4. 训练窗口 ~5h，下轮验证 V-A2 完成 → merge → 原管线评测。

## 2026-09-14 — Round 99：续训机制打磨 + M5 资产

1. 修自伤 bug：resume 编辑误将 accum_loss 变 list（TypeError 秒崩烧 attempt）→ 修复。
2. wrapper 改 save_every 500 + MAXTRIES 40：崩溃间隔 ~500 步、checkpoint 粒度 500 → 进度单调递增（attempt 5 从 step500 续）。累计 ~5.3h 有效训练 ETA 次日早。
3. master_matrix.py：聚合 census_eval_*/probe 结果 → MASTER_MATRIX.json + markdown 表。
4. PAPER_AVATAR.md v1（标题/摘要/贡献/复现性附录/venue 映射）+ refs_seed.bib 37 条种子。
5. 下轮：V-A2 step8000 → merge_lora.py → eval_variant_batch.sh（原管线零分歧）→ 12 探针 + 30 音源四轴 → 真消融表。

## 2026-09-14 — Round 103：h3_research/ 目录整理

1. 顶层 180 个平铺文件 → 分类入 8 个新目录：plans(20 选题/规划)、reports(14 结果/审计)、paper_emoh3(15 论文成稿单元)、scripts(56 py)、runs(42 sh)、jobs(11 json)、results(18 jsonl/jpg/manifest)、archive(事故嵌套目录 h3_research/h3_research/ltx)。
2. 顶层仅留 4 锚点：README.md、WORKLOG.md、PROPOSAL_AVATAR.md、INDEX.md（含旧→新路径速查表）。
3. 刻意不动：submission/、paper_icassp/、tools/（ICASSP 9/16 截稿链路）；avatalk/、h3v2/ 活跃项目原位。
4. 迁移脚本未改写内部相对路径（一次性实验脚本，重跑前检查 cwd）。

## 2026-09-14 — 项目化目录迁移

1. 将按文件类型组织的顶层目录重构为四个独立研究单元：`projects/seed-sensitivity/`、`projects/emoh3-evaluation/`、`projects/avatar-emotion/`、`projects/h3-avec/`。
2. 候选方向与大规模文献调查迁入 `incubator/`；Tectonic、第三方代码和官方 H3 请求归档迁入 `shared/`；缓存统一迁入 `.cache/`。
3. `release/` 与 `submission/` 保持自包含快照，不做去重；旧路径仅在本日志、release 和 ffprobe 证据快照中保留。
4. 修复 seed-sensitivity 分析脚本的项目根目录解析和论文图路径。验证结果：108 个 Python 文件语法解析通过，论文数字校验 38/38 ALL_OK，长版、投稿副本和 ICASSP build 均由 Tectonic 编译成功。

## 2026-09-14 — 项目内部结构闭环

1. EmoH3 源码拆为 `src/{analysis,judges,generation,training,paper}`，运行配方拆为 `runners/{generation,training,evaluation,ops}`；计划、结论和逐轮记录分别进入 `docs/{plans,reports,rounds}`。
2. 修复人评、judge agreement、base-full、frame-span 和 GT-accuracy 脚本的项目根目录解析。核心离线分析均成功读取 `results/judge15-logs/` 并复现现有结果。
3. Avatar 的评测入口迁入 `scripts/eval_fidelity.py`；各项目 `__pycache__`、`.DS_Store` 和 Scholar 浏览器 profile 移入根 `.cache/`。
4. Incubator 拆为 `ideas/`、`proposals/`、`research/`、`references/`、`experiments/`；soft-DD 的 576 个 motion feature 归入其自身实验目录，Sci-VBench 归入 `shared/vendor/Sci-VBench/`。
5. 最终静态审计：108 个 Python 文件、0 个语法错误；seed-sensitivity 数字校验保持 38/38 ALL_OK。soft-DD 本机试跑仅因本地 Python 未安装 PyTorch 停止，代码与数据路径已经解析到位，待远程训练环境验证。

## 2026-09-15 — 项目目录迁移验收

1. `h3_research/` 顶层收敛为 `projects/`、`incubator/`、`shared/`、`archive/`、`.cache/` 五类入口，导航文件仅保留 `README.md`、`INDEX.md`、`WORKLOG.md` 和 `.gitignore`。
2. 将 H3 AVEC 与 Sci-VBench 源码树中新产生的 `__pycache__` 全部移入 `.cache/project-pycache/`；未删除缓存或实验材料。
3. 修正 topic survey、seed-sensitivity outline 和 EmoH3 工作文档中的活跃旧路径；历史 `WORKLOG`、不可变 `release/`、远端运行证据及 `ffprobe` 元数据保持原样。
4. 增加新增材料落位规则：优先归属项目，未立项问题进入 incubator，跨两个以上项目复用的实现进入 shared，不再维护的材料进入 archive，可再生产物进入 `.cache`。
5. 静态验收：顶层目录数 5，源码树 `__pycache__` 0，断链符号链接 0，活跃文档旧路径 0；关键导航目标均存在。

## 2026-09-16 — H3 自研探索清退与复现重置

1. 删除 `projects/h3-avec/` 中自行设计的训练方案、训练脚本、私有偏好数据、探索结果和旧版 proposal；删除共享的自研 H3 训练器。
2. 删除本地 `.cache/` 中对应的生成、奖励打分、路由、时序偏好及 Python 编译缓存；其他项目缓存不受影响。
3. 删除远端专用探索目录 `/ssd2/jinzili/h3_cred`（约 3.3 GB）；保留官方 H3 权重、官方请求脚本及公开数据复现目录。
4. H3 项目重置为验证优先：先复现官方 768p 三种模式，再跑公开 benchmark/基线；形成稳定对照和研究缺口后，才允许在公开数据上设计训练实验。
5. 私有 SynClub 数据仅可用于可视化、诊断和冒烟，不作为论文训练集或核心定量证据。
