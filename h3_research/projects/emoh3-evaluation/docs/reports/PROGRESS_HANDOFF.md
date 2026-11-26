# EmoH3 项目全量进度档案（断点续跑用）

> 目标（round goal，goal-8ab4df5d-e92f-4ca8-86e7-bfc4ec550b87）：化身顶级学府硕士，探索可发会议的 AI 论文话题，须**能用 8×H100 训练**。当前话题 = EmoH3（动漫角色情绪表情可控视频生成）。

## 1. 研究话题与问题陈述

- **话题**：EmoH3 —— 用 MiniMax-H3 Ref2VA（参考驱动视频生成，官方权重 33B transformer + 32B 文本编码器）做**跨情绪表情可控生成**：给一张角色情绪肖像 reference + 一句情绪 prompt，生成该角色做出目标情绪的短视频（含原生音频）。
- **问题（已被实验证实）**：H3 Ref2VA **基座**用情绪 prompt 做跨情绪转换，**8 个 case 失败 7 个**（Qwen2-VL-7B 判定仍是 neutral/原情绪）——角色身份保持得好，但目标情绪表达不出来。这证明「参考生成不能保证表情正确 + 身份一致 + 非编辑区稳定三者兼得」。
- **方法（论文卖点）**：对 Ref2VA transformer 做 **LoRA 情绪条件化微调**（8×H100，caption→video/audio latent 的 rectified-flow 回归），把「情绪表达」作为可训练先验补进模型。
- **评审卖点**：a) 问题有 8×H100 可复现数据；b) 训练链路在 8×H100 实测跑通；c) 基座 vs LoRA 情绪保真对比（核心结果表，正在做）。

## 2. 各轮进展

| round | 内容 | 结果 |
|---|---|---|
| 1 | scholar-crawler 调研 MiniMax H3 可做方向 | `minimax_h3_research_topics.md`（173 条 Scholar 记录），定下 EmoH3 方向 |
| 2 | 8×H100 生成管线试点 + 情绪判官 | Ref2VA serving 8 卡跑通（768²×124f，原生 AAC 音频）；8 case 基座跨情绪 **7/8 失败**；Qwen2-VL-7B 判官脚本就绪 |
| 3 | 8×H100 LoRA 训练链路全通 | 新建 `h3ft` env（torch 2.11 + pinned diffusers）；9 条训练数据 cache（video+真实音频+Qwen3VL text）；**8 卡 LoRA 训练 300 步跑完**（checkpoint-50..300.pt，loss 0.4-1.1 稳定）；修掉 IPv6/TCPStore 坑（FileStore） |
| 4 | LoRA 推理验证搭建 | 写 8×H100 FSDP2 Ref2VA 推理脚本；修 launcher 引号坑、torchrun IPv6 坑（手写 8 进程 spawn）；**FSDP2 管线 + LoRA 首次真正生成出视频张量**（只剩保存函数形状 bug，已修） |
| 5 | **LoRA 生成 + 判官对比（核心结果）** | 8×H100 FSDP2 生成全部 8 个跨情绪 case（768²×124f + 原生音频）；Qwen2-VL 判定：**基座 1/8 → LoRA 3/8（+25pp）**，x01 neutral→joyful 与 x07 sad→cheerful 新成功，x06 部分迁移（neutral→sad）。已写入 proposal §5.2 |
| 6 | **数据策展升级 + LoRA2 重训** | 生成 10 条新情绪保持视频（base 8×H100，覆盖 sad/angry/surprised/anxious/joyful/excited/curious/amused/relaxed/scared）并 mux 真实音频；构建 **18 条纯情绪保持训练集**（8 旧 v2/v3 + 10 新）；prepare_cache 全量重编码（video=448×96，**audio=real**，text=Qwen3VL）；**8×H100 LoRA2 训练 600 步跑完**（loss 0.4-0.9，loss_a~0.15-0.3），checkpoint-600.pt（108 LoRA 张量，54 模块）就绪 |
| 7 | **稳健性研究（推翻 round-5 乐观结论）** | LoRA v2 重测 8 case：单 seed 8 步 3/8，但 **seed-1234 只有 1/8（仅 x03 稳）**；**8 步采样欠解析**——基座 x01 16 步也 Happy:90；**LoRA v2 + 16 步把 neutral→joyful 变成跨 seed 稳健（2/2 vs 基座 1/2）**；负向情绪（sad/worried/relieved/angry）全配置失败 → reference 情绪占主导 = 未解核心问题。论文叙事重构为「评估协议 + 受控 LoRA 增益 + 明确局限」 |
| 8 | **Reference 条件化训练（结构性方法修复）** | patch prepare_cache（ref 图 → video VAE sample(seed42)+float16 → ref_video 行）+ patch train.py（ref 行进 packed sequence，clean 输入，timestep 0.999，loss 只回归 target 行）；构建 **22 条 traindata3**（12 跨情绪对 ref=A/target=B + 10 情绪保持）；1-GPU smoke 通过（seq 601-603）；**8×H100 LoRA v3 训练 900 步完成**；**16 步 × seed42/1234 评估完成：s42=3/8、s1234=4/8，首次 sad 成功（x02 sad:80 @ s1234，base 同配置对照 neutral:100 = 真实效应），x07 cheerful 2/2 seed（v2 是 1/2）；worried/angry 仍全失败** |
| 9 | **768² 分辨率对照实验（证伪假设，负结果）** | 同一份数据在 768² 重训（ref=576 行与 eval 同布局）→ LoRA v4：**s42=0/8、s1234=2/8**，比 v3 更差（x01/x03 退化到 neutral，loss 更低但过拟合）；**分辨率假设被证伪**；x02 sad 在 v3/v4 都稳（唯一跨模型新能力，base 对照 neutral:100）；worried/angry 全模型全配置从未成功 → 指向 base 表达容量边界；**v3（256²）是最佳配置** |
| 10 | **论文写作（主交付）** | 写完整 workshop 论文稿：`paper_emoh3.md`（markdown）+ `paper_emoh3.tex`（LaTeX 可编译，结构校验通过：环境平衡/引用全解析）；编 `DEFINITIVE_RESULTS.md` 全量结果矩阵；标题 *Reference-Conditioned LoRA for Emotion-Controllable Anime Character Video Generation: A Multi-Seed Robustness Study on MiniMax H3*；三贡献：①稳健评估协议（step/seed/frame 会翻结论）②ref 条件化 LoRA 解锁 sad（有对照）③worried/angry 硬边界（768² 消融证伪分辨率假设） |
| 11-14 | 投稿质量打磨（接续轮） | seed999 重测（sad=1/3 seed 诚实修正）、CLIP 身份指标（seed 稳定 0.078）+ Figure 2、全文数字审计（Table1-4 逐项一致）、多角色事实纠错 + **未见角色泛化主张**（a15 不在训练集） |
| 15 | **NeurIPS 2026 模板适配（上一轮）** | 用户选定 NeurIPS 2026 CreativeAI 类 workshop。官方 `neurips_2026.sty` + `paper_emoh3_neurips.tex`（`[dblblindworkshop]` 双盲）+ 修 numeric natbib 引用 + tectonic 编译 8 页 PDF。产物 `release/neurips2026/` |
| 16 | **降档改投 IEEE ICME 2027** | 用户要求降档 → ICME 2027（CCF-B）。`paper_emoh3_icme.tex`（IEEEtran，6 页，0 overfull，引用按 IEEE 顺序）。产物 `release/icme2027/` |
| 17 | **最终改投 ICASSP 2027** | 用户要求「DDL 近 + 接受率高」→ **ICASSP 2027**（截稿 2026-09-16 ≈10 天，接受率 ~50%）。`paper_emoh3_icassp.tex` 4 页精简版。产物 `release/icassp2027/` |
| 19 | **3rd-party 独立 reviewer 审稿 → Weak Reject → 全部 3 项 must-fix 回应（本轮）** | 用 **subagent 当独立 reviewer**（Weak Reject: Soundness 2/Overall 2）抓到：① 成功判据缺陷（joyful/cheerful/worried 不在 12 类 taxonomy，worried 主张空洞；主表/附录矛盾）② 1/8 vs 0/7 统计无效（Fisher p≈1）+ 决定性对照 n=1 ③ 8-vs-16 steps 主张被帧数混淆。**回应**：a) **15 类判官全量重判**（~100 clip-config × 8fr/4fr，judge15.py）——判官只输出 4 词（neutral/happy/sad/surprised），worried/relieved/angry 永不出词→**不可测任务**；b) **判官有效性自检**：21 张已知标签立绘只对 3/21（7/21 宽松）→成功是保守下界；c) steps 主张在固定 8fr 下不成立→撤回；d) 统计诚实：Fisher p=1.00 + CI 重叠入文，x02 重定为"受控演示"；e) seed/帧数(4/88)/taxonomy(12↔15) 三轴敏感性实证。**ICASSP 主投版重写**：4 页 0 overfull，新 Result-1"评估是薄弱环节"，Findings A-D 全改，判官协议/附录/图1 更新；icme 6pp/neurips 9pp/通用 8pp 全部同步（0 overfull）。判官自检代码/日志入 release。**定性图重生成**（15 类判词）。下一步：送 subagent 二轮审稿验证"reviewer 一致可接收" |
| 18 | **审稿驱动的修订（本轮）** | 模拟 reviewer 审稿 → 修：标题软化（"...and the Limits of Emotion Control..."）；`DEFINITIVE_RESULTS` 6/8→**5/8** 一致性；`supplemental/`（judge protocol+12类 taxonomy+verbatim prompt、reproducibility、human spotcheck kit）；release 声明如实化；并跑两个补证据实验：**M1** x02 多 seed 扫描（base 7 seed 全 neutral=0/7，v3 8 seed=**1/8** 仅 s1234 → sad 是罕见事件，诚实降频）+ **M4** reference 互换机制测试（固定 "She looks sad."，同角色 4 情绪立绘：**base 输出跟随 reference 6/8 = reference-emotion dominance 直接证实；v3 抑制 dominance** → 机制从"推断"变"被证明"）。论文全 5 版（icassp/icme/neurips/通用/md）同步更新：Finding A 重写为 1/8 vs 0/7、新增 Finding D+Table 4（tab:mechanism）、摘要/结论/局限（vi）更新。**ICASSP 4 页 0 overfull、全部无泄露、[1]-[11] 干净**。26 条新判官结果入 `release/judge_logs/`（17 文件）。r16 实验日志见 `round16_scratchpad.md` |

## 3. 环境与关键资产（都在 /gluster_hdd_cv/user/jinzili/）

| 资产 | 路径 | 说明 |
|---|---|---|
| 训练 env | `env/h3ft` | Python 3.11 + torch 2.11.0+cu128 + pinned diffusers(commit abc5e9b，in-memory 命名 to_qkv/linear_1/linear_2) + transformers 5.16.1 + peft 0.20.0 + deepspeed 0.19.6 |
| 旧推理 env | `env/minimax_h3_vxf` | torch 2.5.1，旧 diffusers（命名 to_q/to_k/to_v）；用户官方推理脚本基于它，但**与我的 LoRA 命名不兼容** |
| 官方权重 | `weights/MiniMax-H3` | 顶层 transformer_ref / transformer / vae / audio_vae / text_encoder(Qwen3VL-32B) / tokenizer / processor；也有 Ref2VA/、FL2VA/ 子目录 |
| 训练器 | `tmp/h3ft`（IAmIronMan42/MiniMax-H3-FineTuning） | 已打补丁：prepare_cache.py ×2（target_video 键、stdlib wave 读音频）、train.py ×2（--local_rank、FileStore init） |
| 训练数据(旧) | `tmp/pilot/traindata/` | train.jsonl（9 条）+ cache/（9 个 .pt：video latents(448,96) + 真实 audio + Qwen3VL layer-50 text） |
| **训练数据(新)** | `tmp/pilot/traindata2/` | **train.jsonl（18 条纯情绪保持目标）+ cache/（18 个 .pt，audio=real）**——round 6 数据策展升级的核心 |
| **LoRA v1** | `tmp/pilot/runs/lora/checkpoint-300.pt` | round 3 训练，300 步，8 case 判官 **3/8 成功** |
| **LoRA v2** | `tmp/pilot/runs/lora2/checkpoint-600.pt` | **round 6 训练，600 步，108 LoRA 张量/54 模块（新命名）**，基于 18 条数据；**待评测** |
| 情绪保持视频 | `tmp/pilot/outputs_preserve/pres_*_base.mp4`(+wav) | round 6 用 base 模型生成的 10 条情绪保持视频，已 mux 音频轨（train target） |
| 推理脚本 | `tmp/pilot/infer_lora_ref2va.py` | 8×H100 FSDP2 Ref2VA 生成（本地源码为 `../../src/generation/infer_lora_ref2va.py`）；`--lora ""` = base 模式 |
| jobs | `tmp/pilot/jobs_ref2va_emo.json` | 8 个跨情绪 case（情绪肖像 ref + 情绪 prompt） |
| 启动脚本 | `tmp/pilot/run_lora_infer.sh` | 手写 8 进程 spawn（torchrun 被 IPv6 卡死），NCCL_DEBUG=INFO |
| 判官 | `tmp/pilot/judge_emotion.py` | Qwen2-VL-7B（GPU 7），读 `frames/<case>/f*.jpg` 前 4 帧，写 judge_results.jsonl |
| 输出 | `tmp/pilot/outputs_lora/` | LoRA 生成结果（本轮） |

**本地源码**：项目根目录 `../..`；代码在 `src/`，任务配置在 `jobs/`，启动脚本在 `runners/`，计划与逐轮记录在 `docs/`。

## 4. 当前状态（round 17，最终选定 ICASSP 2027）

**✅ 本轮完成（round 17）**：按用户标准（DDL 近 + 接受率高）最终选定 **IEEE ICASSP 2027**，产出 `paper_emoh3_icassp.tex/.pdf`（`release/icassp2027/`）：
- **截稿 2026-09-16**（[call4papers](https://call4papers.org/conf/icassp-2027) 确认；提交前再核官网 CFP）；接受率 ~50%（IEEE 顶会中最高一档）
- 4 页精简版：身份分析 → 补充材料；三大发现（A 解锁 sad / B 正向稳健 / C 硬边界）+ 分辨率消融 + 主表 1-3 + 定性图全保留
- 所有审计数字逐项在位（sad(80)、neutral(100)、2/3、1/3、0.078、unseen-character 等）
- **tectonic 编译 4 页、0 overfull/0 error、无身份泄露、[1]-[11] 干净、2 图嵌入**
- 编译：`TECTONIC_CACHE_DIR=/tmp/tectonic-cache /tmp/tectonic paper_emoh3_icassp.tex`

**⏳ 提交前（ICASSP 2027，9/16 前）**：
1. 官网 CFP 核验模板/匿名政策/4 页上限；非盲提交时填真实作者
2. 人工评估（暂缓）；checkpoint 托管（需 HF token，暂缓）；camera-ready（录用后）

**备选 venue 版本（均保留）**：NeurIPS 2026（`release/neurips2026/`）、ICME 2027（`release/icme2027/`）

**已就绪资产**：全部判官结果（base/v1/v2/v3/v4 × 多 seed/步数/帧数）、`DEFINITIVE_RESULTS.md`（全量矩阵）、`paper_emoh3.md`+`.tex`+`.pdf`（通用版）、三个 venue 版（neurips/icme/icassp）、`release/`、`round7/8/9` 报告、traindata3/4 cache、runs/lora3（最佳）+lora4 ckpt。

## 5. 关键技术坑（务必记住）

1. **034 内核没有 IPv6**：torch≥2.6 的 TCPStore / torchrun rendezvous 都会 `errno 97` 失败。任何分布式都要 **FileStore**（`init_method="file://..."`）+ 手动 spawn 或 deepspeed launcher，**不能用 torchrun**。
2. **launcher 引号坑**：`/run` 的 `cmd` 里带 `&&`/`>`/空格+引号会被 shell 拆错。**一律写成脚本文件**，launcher 只跑 `bash <script>`。curl 必须 `--noproxy "*"`。
3. **SSH percent-expansion**：远程命令别用 `%`（printf 的 %s 会炸）；避免嵌套单引号（用 base64 传文件）。
4. **命名分裂**：盘上权重是旧命名（to_q/to_k/to_v/ff.net.0.proj），pinned diffusers 加载时映射成新命名（to_qkv/linear_1/linear_2）。**LoRA 只能用 h3ft/pinned diffusers 推理**，minimax_h3_vxf 不能用。
5. **FSDP2 分片很慢**（33B+32B 约 5-8 分钟），不是死锁——等它，别急着杀。
6. **prepare_cache 不打包 reference 图片**（官方训练器已知 TODO）——训练目标是 caption→target-latent 映射；真实论文要补 ref 编码进 conditioning。
7. **prepare_cache 只从视频音轨抽音频**（`target_audio` 字段不用）。视频必须内嵌音频轨，否则 cache audio=zero。生成脚本存的是 video-only mp4 + 独立 wav → 需 **ffmpeg mux**：`ffmpeg -i v.mp4 -i v.wav -map 0:v -map 1:a -c:v copy -c:a aac -shortest v.muxed.mp4`（**输出文件名要有 .mp4 扩展名**，.tmp 会 "Unable to choose an output format"）。
8. **prepare_cache 依赖 ffmpeg**：h3ft env 的 PATH 里没有 → 脚本里 `export PATH=/gluster_hdd_cv/user/jinzili/env/sglang310/bin:/usr/bin:$PATH`。
9. **infer 脚本保存名**：base 模式（`--lora ""`）存 `<case>_base.mp4/wav`；LoRA 模式存 `<case>.mp4/wav`（round5）或 `<case>_<lora名>.mp4`。manifest 的 target_video 路径要按实际文件名写。
10. **训练数据质量（round 6 教训）**：round 3 的 9 条里 5 条跨情绪 target 是 neutral 输出（脏数据）。round 6 改为**全部用情绪保持视频**（base 模型能稳定表达 ref 情绪，目标干净）→ 期待负向情绪也学会表达。

## 6. 后续方向（论文推进路线）

1. ✅ 问题证明（基座 7/8 失败，8 步下）——已完成（round 7 修正：含低步数假象）
2. ✅ 8×H100 训练可行——已完成
3. ✅ LoRA 推理对比 + 稳健性研究（round 5/7）——proposal §5.2/§5.4，`round7_robustness.md`
4. ⏳ **reference 条件化训练**（下一轮核心）：ref 编码进 conditioning + 跨情绪目标对 → 冲击负向情绪
5. ⏭ 更大 benchmark（20 角色 × 8 情绪，多 seed）→ 稳健性矩阵做全
6. ⏭ 消融：LoRA 秩/α、音频是否进 loss、步数/CFG、情绪类别扩展
7. ⏭ 身份保持指标（与 ref 相似度，facenet）+ 非编辑区稳定性
8. ⏭ 写作：按「评估协议 + 受控 LoRA 增益 + 明确局限」叙事（proposal 已重构）

## 7. 回滚/复用命令

- 重测（LoRA v2，8 case）：`curl --noproxy "*" -X POST http://osa00-h100-034.osa00.bdjp.io:8659/run -d "cmd=bash /gluster_hdd_cv/user/jinzili/tmp/pilot/run_lora2_eval.sh"`
- 任意小规模生成：改 `jobs_*.json` + 对应 launcher（参考 `run_steps16.sh`/`run_base16.sh` 改 seed/steps/outdir/lora）
- 抽帧：`python extract8_frames.py <src> <dst> [cases...]`（h3ft python + sglang310 ffmpeg）
- 判官：`H3_JUDGE_FRAMES=... H3_JUDGE_OUT=... python judge8.py`（容器 GPU7，sglang310）
- 训练（数据/cache 就绪后重训）：`curl --noproxy "*" -X POST http://osa00-h100-034.osa00.bdjp.io:8659/run -d "cmd=bash /gluster_hdd_cv/user/jinzili/tmp/pilot/run_lora2_train.sh"`

## Round 19c (second judge + honest downgrade)
- **第二判官 Qwen2.5-VL-7B**（`judge25.py` + `judge25_results.jsonl`）：决定性 v3 x02 s1234 = **neutral:85**（Qwen2-VL 判 sad:80），base = neutral:95；35 张自检立绘 **0/35 严格正确**（比 Qwen2-VL 3/21 更盲）。→ **sad:80 是判官特定判词，不被佐证** → 论文诚实降级为 "controlled, non-significant, Qwen2-VL-specific demonstration; not judge-invariant"。
- **两判官分歧成为论文最强证据**：Result 1 新增 judge-choice 不稳定性轴；摘要/结论以此为头号发现（评估不稳定、非判官不变）。
- **协议修复全部落地**：judge_sanity15.jsonl 已填充 21 行；"9 of 15"→"11 of 15"；v1 列 8 帧重判（x07 1/1→0/1）；base x02 脚注核实全 16-step；附录列出全部计数配置。
- **产物**：ICASSP 4pp/0 overfull（已 stage release/icassp2027/）；generic 8pp、icme 6pp、neurips 9pp 全 0 overfull（已 stage）；md、DEFINITIVE §N、review2_audit §第三轮 已更新。
- **round-3 已发给两位 reviewer（11e9e8de、d485f11d）**，等最终 verdict。

## Round 20 (final) — 两位 reviewer 一致 Weak Accept
- round-4 阻塞项修复全部落地（拼图完整披露、base 逐协议数值、限制(i) 两判官、appendix 9→11、sad/surprised 诚实化、两判官不可比注记）。
- **最终 verdict：11e9e8de = Weak Accept（核实数字与文本一致）；d485f11d = Weak Accept clean，无阻塞项**。目标达成。
- 最终产物：ICASSP 4pp/0 overfull/0 泄漏（release/icassp2027/）；generic 8pp、icme 6+1pp、neurips 9pp 全 0 overfull；全部判官日志 + 脚本在 release/judge_logs/（含 judge25_results.jsonl 第二判官、judge_robust.jsonl 协议稳健性）；DEFINITIVE §N/§O、review2_audit 最终结论、checklist §0c 已记录。

## Round 21 (strong-accept 轮) — Qwen3.8-27B 最强判官佐证决定性 x02
- 用户指令："用 3.8"（最新 Qwen 代判官）。SSH relay 曾断（仅交互式认证）→ 用户恢复后继续。
- 部署 **Qwen3.8-27B**（ModelScope，52G bf16，1×H100，Qwen3_5ForConditionalGeneration 原生 VLM；`enable_thinking=False` + 强 system 提示 + 900 token 才拿到可解析 `WORD:SCORE` 输出）。
- 四判官严格同口径（补跑 Qwen2-VL 全 35 张 + Qwen3-VL-32B + v2raw）：
  - **自检（同 17 张词表内立绘）**：Qwen2-VL 1/17、Qwen2.5-VL 0/17、Qwen3-VL-32B 1/17、Qwen3.8-27B 0/17 → 任何开放判官过不了基本动漫情感自检。
  - **决定性 x02**：v3 = sad:80（Qwen2-VL）/ neutral:85（Qwen2.5-VL）/ worried:85（Qwen3-VL-32B）/**sad:82**（Qwen3.8-27B，两次判读 sad:85/82）；base = neutral 四判官全一致（50/95/95/78）。
  - **x04 (joyful→worried) 解锁**：新判官 worried:95/80（7B 判官从不说 worried）→ v3 x04 可测且成功（带 worried-bias 警示）。
- **叙事升级**：x02 从 "judge-specific artifact" → "bounded, judge-dependent but independently corroborated by the strongest judge"；分歧由异议判官的负情绪识别缺口解释（两异议判官都认不出哭泣立绘）。
- 论文四版本 + md 全部更新（四判官表 tab:judges + 佐证叙事）；ICASSP 内容 4 页 + 参考文献第 5 页，0 overfull/0 错误。
- 数据：`judge15_logs/` 新增 judge38_results.jsonl（53 行）、judge3vl_results.jsonl、judge_extra.jsonl、judge15_35.jsonl、MASTER_4JUDGE.md。
- 待：round-5 发给两位 reviewer（携 Qwen3.8 佐证证据），寻求 Weak→Strong Accept。

## Round 22–23：全 88 配置多判官一致性 + 人类标注工具就绪
- **全 88 clip-config × 多判官**（`judge_all88.py`，纯判官推理，3 卡并行）：Qwen2.5-VL ✓88、Qwen3-VL-32B ✓88、Qwen3.8-27B 进行中。
- **一致性统计**（`analyze_agreement88.py`，可复现）：同代 7B κ=0.49/71.6%，**跨代 κ=0.24/42.0%**，三判官 Fleiss κ=0.318（同义词折叠 0.396），完全一致仅 29/88。判官用词量：Qwen2-VL 4/15（72% 判 neutral）、Qwen2.5-VL 6/15、Qwen3-VL-32B 9/15。
- **★ 决定性负结果：干预效果符号随判官翻转**。同一矩阵同一判据：Qwen2-VL base 19.0% → v3 26.1%（+7.0pp）；Qwen2.5-VL 33.3% → 30.4%（−2.9pp）；Qwen3-VL-32B 42.9% → 34.8%（−8.1pp）。严格匹配 4 任务：原判官 base 1/4→v3 4/4，两个新判官都是 3/4→3/4。→ 报告的增益主要是原判官中性偏置的产物；论文已改为"不声称判官无关的提升"。
- **论文（ICASSP）已按此重写**：摘要 (i) 含 κ 与符号翻转；Result 1 含全 88 一致性；**Finding B 改为"增益不经受判官更换"**；Result 2/Finding C/D、限制（新增"无人类真值"）、结论全部同步。0 error/0 overfull，正文 4 页 + refs 第 5 页。
- **两位 reviewer 的 3 项 must-fix 已修**：Qwen3.8 的 fearful:85 对称披露（删除单边"只有异议判官盲"的论证）；3/21↔1/17 对账；`verify_judge_counts.py` 路径 bug 修复（自动探测日志目录，已实测从 release 目录可运行）。
- **人类标注工具就绪（等用户标注）**：`human_label/index.html`（自包含，无需服务器/联网；94 题 = 55 个盲化视频片段 + 39 张已知立绘；随机顺序、文件名 i001…i094 不含条件信息；15 类强制选择 + 把握度；快捷键；localStorage 断点续标；导出 `labels_<代号>.json`）。生成器 `make_label_tool.py`；打分器 `score_human_labels.py`（已用合成标签跑通全流程：标注者间一致性、立绘可靠性、**每个判官 vs 人类共识的一致率/κ**、决定性 x02 人类判词、**用人类标签重算 base vs v3 效应方向**）。说明见 `human_label/README.md`。
- **下一步**：用户标注（建议 2–3 人各一份）→ 跑 `score_human_labels.py` → 按结果定稿（哪个判官可信 / 效应方向是否有真值支持）→ 传播到 8pp/icme/neurips/appendix/md → reviewer round-6。

## Round 24（round-7 审稿修正）：撤回"符号翻转"，改用选项序不稳定性做核心证据
- **两位 reviewer 独立指出同一缺陷（他们是对的，已修）**：`+7.0pp / −8.1pp` 的"符号翻转"建立在**任务不匹配的分母**上。88 配置池异质（16 步迁移任务 49、refswap 探针 12、x02 扫描 11、非 16 步/4 帧 5），且 base 只在 4/8 个任务上有 16 步片段。已在四个 tex 版本 + md + appendix 全部撤回并加**明确的 retraction note**。
- **脚本 bug 修复（M2）**：`frames_refswap_lora_*` 原被误分到 v1，实为 v3；`analyze_agreement88.py` 已修，并改为：pairwise κ 用每对自己的交集（数据最大化）、Fleiss 用四判官全交集、匹配比较用每个判官自己的覆盖范围。新增 `pool_kind()` 披露池组成、`fisher_exact_2x2()` 纯 Python 精确检验。
- **任务匹配后的正确结论**：Qwen2-VL base 1/4 → v3 4/4(s1234) / 2/4(s42)（Fisher p=0.143 / 1.00）；Qwen3.8-27B base 2/3 → 3/3；Qwen2.5-VL 与 Qwen3-VL-32B 均 3/4 → 3/4（打平）。**四个判官、两种选项序下都不存在 base > v3**，但 2/4 判官看不到增益且无一显著 → 仍然"无判官无关的改进"。
- **★ 新的核心证据：判官对选项顺序都不稳定**。同判官、同帧、greedy，仅反转 15 个候选词的顺序：Qwen2-VL 改判 17/88（自一致 80.7%，κ=0.651），Qwen3-VL-32B 改判 10/88（88.6%，κ=0.857）；而重复运行同一配置 88/88 完全一致（仅 1 例置信分不同）→ 不稳定性来自 prompt，不是采样噪声。**变更落在结论依赖的格子上**：`base16 x01` neutral→happy（基线"失败"变成功）、`lora3s x02`（决定性片段）在 Qwen3-VL-32B 下 worried→**sad**（异议本身是选项序产物）。
- **四判官全 88 一致性**（Qwen3.8-27B 已跑完 88 行，77 行可解析）：层内高、层间低 —— 两个 7B κ=0.493/71.6%，两个最新 κ=0.633/68.8%，跨层 41.6–47.7%；Fleiss κ=0.376（同义词折叠 0.479），完全一致 23/77。**分歧集中在"有没有情绪"**：剔除双方都判 neutral 后仅 36.7–50.0%，但双方都认定有情绪时选词一致率 87.9–100%。
- **Finding D 是唯一跨判官复现的结论**：参考情绪跟随率 base 5/8、4/8、4/8 → v3 2/8、1/8、2/8（三判官一致方向）；判官共识标签下 3 个不一致配对全部支持抑制（精确 McNemar p=0.25，n=8 功效不足）。
- **陈旧正向表述已全部清除（M3）**：`modest but real robustness gains`、`cheerful 2/3 vs 1/2`、`positive-target seed robustness` 在 3 个 sibling tex + 3 个 release 副本 + md 中全部替换。
- **M4 已修**：88 配置的一致性统计原写"four judges"，现按实际覆盖写明（四判官交集 77 行）。
- **编译状态**：`paper_emoh3.tex` 9 页、`paper_emoh3_icme.tex` 7 页、`paper_emoh3_neurips.tex` 10 页（正文止于 p9，References 在 p10）、ICASSP 5 页，全部 0 error / 0 overfull；appendix 片段用 wrapper 验证通过。
- **正在跑（决定性实验，两位 reviewer 共同指定）**：`run_basefull.sh`（pid 170189）—— base 模型在**全部 8 个任务 × 2 seed × 16 步**、与 `run_lora3_eval.sh` 完全相同的推理设置下生成，使 base 与 v3 任务完全匹配。完成后需抽帧 → 四判官重判 → 用完整匹配矩阵重算效应。
- **下一步**：等 base 全矩阵生成完 → 抽帧 + 四判官重判 → 更新 Finding B 为 8 任务匹配结论；用户标注 94 题（`human_label/`）→ `score_human_labels.py`。

## Round 25（round-7 决定性发现）：判帧只覆盖片段前 19%，改成全片段后结论翻转
- **★ 发现协议缺陷**：`extract8_frames.py` 用 `-ss t`、`t = i/7` **秒**取 8 帧，即全部落在 **0–1.0 s**；而片段是 **5.17 s / 124 帧 / 24 fps**。所有判官只看到片段开头 **19%**，而开头恰是参考立绘表情最可能残留、扩散尚未展开处。此前 7 轮、两位 reviewer 均未发现。
- **控制实验**（`extract8_full.py` 全时长均匀取 8 帧，其余完全不变）：同判官在同一批 81 个片段上 **Qwen2-VL 改判 27/81（自一致 66.7%，κ=0.459）**、**Qwen3-VL-32B 改判 30/81（63.0%，κ=0.549）**。**这是本项目最大的单一测量方差来源**，大于判官身份、大于选项序。内部校验：全片段集 f01 与原集 f05 字节相同（t≈0.72 s），证明两套帧来自同一视频。
- **是系统性偏差而非噪声**：非 neutral 判决 Qwen2-VL **23/81 → 45/81**、Qwen3-VL-32B **60/81 → 70/81**；命中目标率（LoRA 臂）**13/63 → 23/63（精确 McNemar p=0.002）**、**20/63 → 33/63（p=0.001）**。**这是全项目唯一显著的效应，而它是关于评测协议本身的。**
- **★ 直接推翻本文残余的正面主张**：`base x02 判为 sad` 由 **0/6 → 2/6**（Qwen2-VL，seed 777/780）、**1/6 → 2/6**（Qwen3-VL-32B，seed 777/778）——"base 从不表达 sad"这一对照前提失效；`base16 x01`、`base16b x01/x07` 由 neutral 翻成 happy，任务匹配下 base 由 1/4 升到 3/3。**任务级结论已全部撤回**：按现有证据 reference-conditioned LoRA 无可测的任务级提升。
- **唯一在四种打分条件（2 判官 × 2 判帧协议）下方向一致的仍是机制项**：参考情绪跟随 base 5/8、4/8（0–1 s）与 3/8、4/8（全片段）对 v3 恒为 2/8；n=8、McNemar p=0.25 → 只作"方向一致"，不作既成效应。
- **论文已彻底重定位为测量学论文**，四个版本全部重写并编译通过（0 error / 0 overfull）：新标题 *Emotion-Control Evaluation Is Not Invariant: Judge, Option Order and Judged Frame Span Each Flip 11–58% of Verdicts on MiniMax H3*；ICASSP 版新增 `tab:sensitivity`（三轴敏感性主表，含"重复运行 0/88"噪声底），删除 `fig:method` 与 `tab:main` 以腾出版面；正文 4 页 + References 第 5 页。ICME 7 页、NeurIPS 正文止 p9、8pp 版 9 页。
- **新脚本（已随 release 发布）**：`extract8_full.py`（集群）、`judge_frameset.py`（集群，`FRAMESET`/`JUDGE`/`VARIANT` 三参数）、`analyze_framespan.py`、`analyze_basefull.py`。新判官日志：`judge_full_q2.jsonl`、`judge_full_q3vl.jsonl`（各 86 行）。
- **运维经验（重要）**：① gpu-launcher 每次 `/run` 似乎只跟踪一个进程，**新的 `/run` 会连带杀掉上一个进程组** —— 同一时刻只跑一个 launcher 任务。② 8 卡任务在共享节点上被反复抢占/杀掉（无 traceback），改为 **4 卡（CUDA_VISIBLE_DEVICES=4-7）** 后稳定。③ 单实例并发会让两组 4-rank 进程抢同一个 NCCL FileStore 而死锁，`run_basefull4.sh` 已加 mkdir 锁。④ 通过 `/tmp/jc.sh` 传命令时**不要在外层双引号里再用单引号**，会被 RemoteCommand 的单引号包裹截断。
- **下一步**：base 全 8 任务 × 2 seed 生成（s42 已 8/8，s1234 进行中）→ `run_judge_bf.sh`（两种判帧协议 × 四判官）→ `analyze_basefull.py` 出完全任务匹配 × 两判帧协议的最终矩阵 → 用户标注 94 题。

### Round 25 追加（round-8 审稿修正 + 效度论证）
- **第二位 reviewer 的 round-8 复核抓出 7 处可核对的内部矛盾，全部已修**：① 摘要写 v3 参考跟随"每种条件都是 2/8"，实际 Qwen2.5-VL 是 1/8 → 改为 1/8–2/8；② "判帧是最大的轴"与自家表格（跨代判官 51/88 = 58%）矛盾 → 全部改为"**最大的 within-judge 轴**"并注明跨代换判官更大；③ "任务匹配 base 由 1/4 升到 3/3"换了分母（3/3 来自 62 片段中间集）→ 重算完全配对矩阵，**base 1/4 → 3/4、v3 2/4 → 3/4（打平）**，Qwen3-VL-32B 两协议均 3/4 对 3/4，x02 上两臂命中率相同（base 2/6 对 v3 4/12）；④ 判官一致率混用 77/88 两个集合 → 逐个注明 n（q2–q25 在 88 集是 71.6%、在 77 集是 75.3%）；⑤ "共识标签 3 个不一致配对、p=0.25"其实是 Qwen2-VL 单judge 的数 → 改为 **≥2/3 多数共识 base 4/8 对 v3 2/8、2 个不一致配对、精确 McNemar p=0.50**，并注明单判官 p=0.25；⑥ Finding C 的 worried:95 是旧日志的值 → 改为 85（与发布日志一致）；⑦ 标题里的 "Each Flip 11–58%" 有分配律歧义（没有单个轴同时达到 11% 和 58%）→ 去掉 "Each"。
- **同时补上 reviewer 指出的唯一"无需人工标注也能做"的效度实验，并且成功**：全片段判帧使**两个不同代际判官彼此更一致**（50.6%→56.8%，κ 0.344→0.463，Δκ=+0.119，配对 bootstrap P(Δκ≤0)=0.047）。这把"判帧协议只是改变了标签"升级为"更长的窗口是更可靠的测量"，从而使"应当按整段判帧"的建议有可复现依据。实现在 `analyze_framespan.py` 的 `reliability()`。
- **ICASSP 版结构再调整**：新增 `tab:retract`（一张表说清撤回：两种判帧下的任务匹配矩阵 + x02 命中率），贡献段重写为"三轴不变性缺失 + 释出语料与工具包 + 用自家 LoRA 做的一次结果消解演示"，并新增一段说明撤回逻辑的**非对称性**（显著的 McNemar 效应在 LoRA 臂，摧毁对照的 base 侧变化不显著 → 我们不主张"base 更好"，只主张"对照失效使原结论无支撑"）。`fig:qual` 改为单栏以腾版面；`tab:tasks` 内联为正文。
- **四版最终页数**：ICASSP 正文 4 页 + References 第 5 页；ICME 7 页；NeurIPS 正文止 p9、References p10；8pp 版 9 页。全部 0 error / 0 overfull。
- **集群运维再补一条**：本容器的 `nvidia-smi` **有时看不到其他租户占用**（曾显示 4 MiB 而实际有 44 GiB），所以不能靠查显存挑卡；`run_basefull4.sh` 改为**依次尝试 5 组 4 卡组合（4567/0123/2367/0145/1357），以是否真的产出 mp4 为成功判据**，并保留 mkdir 锁（按 lock 的 mtime 判活，因为 launcher 在另一个 pid namespace 里，查 pid 无意义）。

## Round 26（round-9：完全任务匹配矩阵 —— 本轮唯一真正的新证据）

### 结果（决定性）
把 base 用与 v3 **完全相同的推理设置**在全部 8 任务 × 2 种子上重新生成（16/16 mp4 全部产出），
再用同一套 15 类判帧协议判官，得到每判官 16 个完全配对格子：

| 判官 | base | v3 | Fisher p | 配对 McNemar p |
|---|---|---|---|---|
| Qwen2-VL-7B | 4/16 | 6/16 | 0.704 | 0.688 |
| Qwen2.5-VL-7B | 6/16 | 6/16 | 1.000 | 1.000 |
| Qwen3-VL-32B | **9/16** | 8/16 | 1.000 | 1.000 |

**三判官全零，最强判官给 base 更高。** 撤回从"换协议后增益消失"的推论升级为**配对实验的直接测量**，
正好回答 round-8 reviewer 的第 7 条批评（"唯一显著的效应都在被撤回的那条臂上"）。
两个附带观察在主协议下就摧毁了旧前提：**base x02 s42 = sad（Qwen3-VL-32B）**、
**base x08 s1234 = angry（Qwen2-VL + Qwen3-VL-32B）**。

判帧轴同时扩到 3 判官 × 97 片段：翻转 32/97、32/97、34/97（κ 0.48/0.48/0.57）；
base 臂 8/34→16/34（**p=0.021**，第一次在 base 臂拿到显著）；
LoRA 臂 13/63→23/63（p=0.002）、17/63→18/63（**Qwen2.5-VL 完全没动**）、20/63→33/63（p=0.001）。

### 一个被降级的强主张（必须记住）
round-8 写下的"全片段判帧**更可靠**（Δκ=+0.119, p=0.047）"在补齐三判官后**不成立为强主张**：
三个配对 Δκ = +0.114（p=0.035）、+0.011、+0.001 —— 只有一个显著。
四篇论文 + appendix + DEFINITIVE_RESULTS R5b 已全部改写为
"不更差、可能更好，但不能称系统性更可靠"。`analyze_basefull.py` 里也打印了这段限定。

### 运维教训（本轮浪费了约 2 小时，务必记住）
1. **gpu-launcher 的 `/run` 不会杀掉上一个任务** —— 它把所有 pid 都记进列表。我本地的
   `/tmp/drive4.sh` 每 6 分钟 poke 一次 `/run`，于是**同一个脚本被并发拉起了 9 次**，
   互相抢 GPU 并 OOM，看起来像"外部租户占了 44 GiB"。真正的修复是
   **`POST /stop`（不带 pid）一次性杀光全部 tracked pid**，然后只 `/run` 一次。
   停掉本地 driver + 单实例后，剩下 3 个 case 在 14 分钟内全部跑完。
2. **容器内的 `pgrep` 看不到 launcher 起的进程** —— launcher 跑在宿主机上。
   所以 `S=0 I=0` 不代表没有任务在跑，**用 `/list` 判断，不要用 pgrep**。
   同理，容器内 `nvidia-smi` 看到的"外部占用"很可能就是自己的进程。
3. **Qwen3.8-27B 在 plain transformers 下约每片段数分钟**（先输出一大段思考再给 WORD:SCORE），
   16 个片段跑了 50 分钟才 4 行。判官顺序要把它放最后，或者干脆先跑 q2/q25/q3vl
   （97 配置 × 3 判官不到 1 小时，见 `run_judge_full3.sh`）。

### 论文状态（全部 0 error / 0 overfull）
- `paper_emoh3_icassp.pdf` 5 页，正文止于第 4 页 ✓（主投）
- `paper_emoh3.pdf` 9 页；`paper_emoh3_icme.pdf` **7 页**（删掉冗余的 tab:unstable 后达标）；
  `paper_emoh3_neurips.pdf` 10 页，正文止于第 9 页 ✓
- 删除了两张已被取代且会误导的表：`tab:main`（分母不匹配的旧逐任务表）与
  `tab:unstable`（已被 `tab:sensitivity` 定量取代）。
- 仍然缺的唯一东西：**人工标注**（`human_label/labels_*.json`）。没有它，
  "全片段判帧是否更*正确*"永远只能停在可靠性论证上。

### 本轮收尾（第四个判官 + 外部审稿修正）
- Qwen3.8-27B 补齐配对矩阵：**8/12 vs 8/12（p=1.000）** → **四判官全零**，撤回没有"漏测判官"的缺口。
  它在 plain transformers 下每片段数分钟，因此**只跑 0–1 s 的 16 格**，不跑 97 配置全片段（论文已注明）。
- 外部 reviewer（重跑全部脚本+核对原始 log）给 **Weak Accept (0.60)**，指出 8 处问题，**全部已修**：
  选择性只报 3/6 个显著格子（现已全报 6 格 + Bonferroni 说明 + "平的那格本身就是论点"）、
  "only significant effects anywhere" 与自己的 Δκ p=0.035 冲突（改为 model-side）、
  结论 37%→35%、Limitations 81 clips/2 判官→97/3、
  正文与 appendix 的"排除双中性"区间没写子集（两处都写明并互引）、
  摘要 "v3 1/8--2/8" 不精确（四条件下都是 2/8）、
  x04/x02 置信度跨 log 不一致（注明来源）、"100+ configs"→88、
  "three generations"→"four releases spanning three generations"、
  option-order 的 88/88 test–retest 其实是反序变体的重复（已注明）。
- 论文最终状态：ICASSP 5 页（正文止于 p4）、generic 9 页、ICME 8 页（正文 7 页）、NeurIPS 10 页（正文 9 页），
  全部 0 error / 0 overfull。
- **仍然唯一的 Strong Accept 缺口 = 97 个片段的人工标注**（`build_labelpack.py` 已就绪，等用户打标）。
  没有它只能停在"可靠性不更差"，无法证明全片段判帧更*正确*。

## Round 27（round-10：**准确率**证据落地 —— 第一次能说"更正确"）

reviewer 说唯一能到 Strong Accept 的路是"证明全片段判帧更**正确**而不只是更一致"，
并把它归为"必须人工打标"。**本轮找到了不需要人工打标的外部锚点**：
`traindata3/train.jsonl` 引用的 14 个源片段来自与 `expressions_0825` 已知情绪立绘同一生产批次，
情绪写在文件名里（论文本来就把这批立绘标签当 ground truth 用于 17 立绘效度检查）。

结果：strict 0/14→3/14、0/14→4/14、5/14→9/14；lenient 4/14→8/14、3/14→**9/14 (p=0.031)**、7/14→11/14。
**六个比较全部改善，25 个对错变化全部（25/25）站在全片段一侧，零反例。**
改善集中在 4 个 cross 片段；`pres_*` 里 anxious/curious/relaxed/scared 两协议都失败
（判官根本看不见这些情绪，与 0/5 negative 一致）——两条限制都写进论文了。

四篇论文的摘要 + 判帧段 + Limitations 全部更新；appendix 加了 14 行逐片段表（`\resizebox` 防 overfull）。
ICASSP 为塞进这段压缩了 (b)(c) 两个轴的行文、Finding C/D、resolution ablation 和结论，
并把 figure 缩到 0.93 columnwidth —— **正文仍止于第 4 页**，references 独占第 5 页。

新文件：`analyze_gt_accuracy.py`、`run_gt.sh`、`judge_gt{1s,full}_{q2,q25,q3vl}.jsonl`（各 14 行）。
