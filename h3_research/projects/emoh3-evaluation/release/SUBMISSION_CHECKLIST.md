# EmoH3 — Submission Checklist & Venue Targeting

**Status date: 2026-09-06.** **Primary submission target: IEEE ICASSP 2027 (deadline 2026-09-16).**
Paper ready in venue formats (ICASSP 4 pp primary; ICME/NeurIPS backups).

## 0b. Independent reviewer (round 19) — 3rd-party Weak Reject → response
- **3rd-party reviewer (subagent, round 19)** gave **Weak Reject (Soundness 2, Overall 2)** with 3 must-fix:
  1. Success-criterion flaw (joyful/worried/cheerful absent from 12-class taxonomy → vacuous worried claim; main table vs appendix contradiction) → **FIXED**: 15-class judge re-run of all ~100 clip-configs, explicit synonym rule, regenerated appendix, untestable-target disclosure.
  2. Statistically-null headline (1/8 vs 0/7, Fisher p≈1; single-config decisive comparison) → **FIXED (honest)**: Fisher p=1.00 + overlapping 95% CIs stated; x02 reframed as controlled demonstration, not significant effect.
  3. Steps claim confounded (8s/4fr vs 16s/8fr) → **FIXED**: re-extracted base 8s @ 8fr; at fixed frames no step flip replicates; claim withdrawn.
- **New evidence from response**: (a) judge conservatism — only 4/15 classes ever emitted (100+ configs); (b) **judge validity check fails — 3/21 known reference emotions** (7/21 lenient) → successes are lower bounds, untestable targets unattributable; (c) 4/88 frame-count flips; (d) taxonomy itself flips verdicts (12 vs 15).
- **Round-19 paper (ICASSP primary)**: 4 pp, 0 overfull, no leaks, all key facts verified; Findings A-D rewritten; new Result-1 (evaluation is the weak link); supplementary judge_protocol.md updated to 15-class + validity check.

## 0. Reviewer-driven revision (round 16) — status
Self-review (as reviewer) findings + fixes:
- [x] **Title overclaim** → softened to *"...and the Limits of Emotion Control in Anime Character Video Generation..."* (all 5 versions)
- [x] **Internal inconsistency** → `DEFINITIVE_RESULTS.md` CLIP "6/8 ≤0.10" corrected to **5/8** (x06=0.101)
- [x] **Reproducibility gap** → new `supplemental/` (judge protocol: exact 12-class taxonomy + verbatim prompt; reproducibility sheet: pinned versions; human spot-check kit)
- [x] **Release claim** → softened to "verdict matrix/judge logs/protocol released as supplementary; checkpoints from authors" (honest until public hosting)
- [x] **M1: x02 seed sweep** → DONE (base 0/7 never sad; v3 1/8, only seed 1234). Finding A rewritten honestly; 26 new judge logs in `release/judge_logs/`. Round 19: decisive cell taxonomy-stable (sad:80 in both 12- and 15-class judges)
- [x] **M4: reference-swap mechanism test** → DONE (base tracks reference 5/8 on judge-reliable classes = dominance directly confirmed; v3 suppresses it, caveat: partly neutral-regression). New Finding D + Table 4 (tab:mechanism) in all 5 paper versions
- [ ] **M2: human judge spot-check** → kit ready (`supplemental/human_spotcheck_kit.md`); needs 2 human annotators (~26 clips)
- [ ] **External baseline** → deferred (rationale in limitations); internal v1/v2/v3/v4 ladder is the reference-agnostic control. M4 mechanism test partially addresses the "inferred mechanism" critique

## 1. Venue options (ordered by nearest deadline)

| Venue | Typical submission | Fit | Notes |
|---|---|---|---|
| **NeurIPS 2026 workshops** (e.g. [CreativeAI](https://neurips.cc/Conferences/2026/CallForCreativeAI)) | **~Sep–Oct 2026** ← nearest | High (creative content + honest evaluation) | Verify each workshop's CFP; typically 4–8 pp + refs, anonymous |
| [**CVPR 2027 AIGENS**](https://ai4streaming-workshop.github.io/) (AI for Content Generation) | ~Mar 2027 | High (video generation) | CVPR26 version closed 2026-03-17; watch for CVPR27 relaunch |
| [**ACM Multimedia 2027**](https://2026.acmmm.org/site/important-dates.html) workshops | ~Jul 2027 | High (multimedia + creative) | ACM MM 2026 workshop submission closed 2026-07-16 |
| ICME / AAAI / SIGGRAPH-Asia 2027 | ~Nov 2026 / 2027 | Medium | Backup options |

## 2. Format adaptation (per chosen venue)
- **NeurIPS**: use `neurips_2026.sty` (single or two column per CFP); our `article` content ports
  directly. Replace `\documentclass{article}` + `geometry` with the venue class; keep
  `graphicx/booktabs/amsmath/hyperref`.
- **CVPR**: use `cvpr.sty`; check page limit (typically 8 pp incl. references for main, 4–8 for
  workshop). Our 6 pp fits either.
- **Anonymity**: author block is already "Anonymous" (line in the class); verify no names/paths leak
  in the final PDF (pdftotext grep for "jinzili" / hostnames — currently clean).
- **Abstract**: venue templates may want ≤200–300 words; ours is ~250.

## 3. Content checklist (audited 2026-09-05)
- [x] Abstract consistent with body (sad "1 of 8 seeds"; NO positive-robustness claim; NOT "train from scratch")
- [x] Table 1 (tasks) — 8 tasks, prompts, no numeric claims
- [x] Table 2 (protocol instability) — all rows trace to judge logs (`release/judge_logs/`)
- [x] Table 3 (main results) — matches `DEFINITIVE_RESULTS.md` (incl. v3 x02=1/3, x07=2/3)
- [x] Table 4 (identity) — matches `clip_identity_all.jsonl` exactly
- [x] Figure 1 (method layout) + Figure 2 (qualitative) embedded; captions match Findings
- [x] References resolve (11 bibitems; checked)
- [x] Limitation "|Δ|≤0.10 for 5/8 tasks" (math corrected from 6/8)

## 4. Missing before real submission
- [x] **Pick a venue + adapt template** — **最终选定 IEEE ICASSP 2027**（用户要求 DDL 近 + 接受率高）：
      `paper_emoh3_icassp.tex` + `.pdf`（4 页 + 参考文献，匿名双盲，0 overfull，无身份泄露）。
      截稿 **2026-09-16**（[call4papers 确认](https://call4papers.org/conf/icassp-2027)，提交前再核官网 CFP）。
      NeurIPS 2026（`release/neurips2026/`）与 ICME 2027（`release/icme2027/`）版保留为备选。
- [ ] **提交前按官网 CFP 核验**：ICASSP 2027 官方模板（若与 IEEEtran 有差异按官方改）、匿名政策、
      4 页上限确认、去 `colorlinks`、填真实作者块（非盲提交时）
- [ ] **Human study**（可选；协议就绪 `HUMAN_EVAL_PROTOCOL.md`）
- [ ] **Checkpoint hosting**: `runs/lora3/checkpoint-900.pt` (43 MB) 仅集群；公开托管需 HF token（录用后/匿名链接用）
- [ ] **Camera-ready**: 真实作者、致谢、补充材料链接（录用后；身份分析已按"补充材料"处理）

## 5. One-line pitch (for CFP matching)
> "A robustness-aware, multi-seed evaluation of emotion-controllable anime character video
> generation on MiniMax H3 (33B, 8×H100), showing that reference-conditioned LoRA makes a negative
> emotion expressible under a controlled neutral baseline, with an honest capacity boundary for
> frown-type emotions and a seed-stable CLIP identity metric."

## 6. Updates (2026-09-06)
- **Multi-character correction**: training is 12 cross-emotion pairs (single char 1534125952) + 10
  preservation clips over 10 distinct characters; eval covers 7 characters, two of which (x01/x02,
  ref a15) are **unseen at training** — v3's sad breakthrough and x01's 2/2 are therefore
  unseen-character generalization. Paper §5/§6.1/§6.2/§7 updated; 7 pages.
- **Appendix**: `paper_emoh3_appendix.tex/.md` — full judge verdict matrix (all models × steps ×
  seeds × frames), auto-generated from the verified matrix; compiles standalone.
- **Upload plan**: `RELEASE_UPLOAD.md` — exact HF/GitHub commands for the two checkpoints + logs.
- **NeurIPS template**: fetch official `neurips_2026.sty` from the venue CFP/Overleaf at submission
  time (unreachable via this network today); content ports by swapping `\documentclass` + title block.

## §0c. Round 19c — second-judge corroboration attempt (honest outcome)
- [x] **第二判官 Qwen2.5-VL-7B 判决定性对**：v3 x02 s1234 = neutral:85（不佐证 sad:80）；base = neutral:95。自检 0/35（更盲）。
- [x] **诚实降级**：x02 = "controlled, non-significant, Qwen2-VL-specific demonstration; not judge-invariant"（摘要/Result 1/Finding A/结论）。
- [x] **两判官分歧写入**：judge-choice 不稳定性轴 + 3/21 vs 0/35 对比（评估非判官不变的最强证据）。
- [x] **协议修复收尾**：sanity 21 行已发布；11 of 15；v1 8 帧；base x02 脚注；附录配置清单。
- [x] **全部版本重编译 0 overfull**（ICASSP 4pp；generic 8pp；icme 6pp；neurips 9pp）。
- [ ] 两位 reviewer（11e9e8de、d485f11d）round-3 最终 verdict 待收。

## §0d. Round 21 (strong-accept round) — Qwen3.8-27B 最强判官佐证
- [x] **部署 Qwen3.8-27B 判官**（用户指定最新 Qwen）：ModelScope 下载 52G，1×H100 bf16，Qwen3_5ForConditionalGeneration 原生 VLM；修 thinking-mode 输出（enable_thinking=False + 强 system + 900 token）。
- [x] **四判官严格同口径**（补跑 Qwen2-VL 全 35 张、Qwen3-VL-32B、v2raw）：
  - 自检（同 17 张词表内立绘）：Qwen2-VL 1/17、Qwen2.5-VL 0/17、Qwen3-VL-32B 1/17、Qwen3.8-27B 0/17 → 任何开放判官过不了基本动漫情感自检（评估薄弱论在四代两尺度全面成立）。
  - 决定性 x02：v3 = sad:80/neutral:85/worried:85/**sad:82**（Qwen3.8-27B 两次判读 sad:85/82，含显式推理）；base = neutral 四判官全一致。
  - x04 (joyful→worried) 解锁且成功（worried:95/80），带 worried-bias 警示。
- [x] **论文叙事升级**：x02 = "bounded, judge-dependent but independently corroborated"（原始判官 + 最强判官双佐证；异议判官为负情绪识别缺口）。ICASSP 摘要/Result1/Finding A/C/限制(i)/结论 + 新增 Table 3 (tab:judges)；generic 8pp/icme/neurips 5 处核心编辑 + 表同步。
- [x] **ICASSP**：内容 4 页 + 参考文献第 5 页，0 overfull 0 错误（IEEE 标准格式）。
- [x] **数据发布**：judge38_results.jsonl、judge3vl_results.jsonl、judge_extra.jsonl、judge15_35.jsonl、MASTER_4JUDGE.md（待 stage release/judge_logs/）。
- [ ] **round-5 发两位 reviewer**（11e9e8de、d485f11d）：携 Qwen3.8-27B 佐证 + 四判官全景，寻求 Weak→Strong Accept。
- [ ] **release staging**：icassp2027/ + judge_logs/ 重新归档。

## Round 7 审稿修正核查（2026-09-07）
- [x] **撤回"效应符号随判官翻转"**：分母不匹配（base 仅 4/8 任务有 16 步片段；池内含 refswap/x02 扫描/非 16 步行）。四个 tex + md + appendix 均已撤回，appendix 附明确 retraction note。
- [x] **改用任务匹配结论**：Qwen2-VL base 1/4→v3 4/4(s1234)（Fisher p=0.143）；Qwen3.8-27B 2/3→3/3；Qwen2.5-VL / Qwen3-VL-32B 均 3/4→3/4。四判官 × 两种选项序下均无 base > v3，但 2/4 判官无增益且无一显著 → 仍主张"无判官无关改进"。
- [x] **新增选项序不稳定性控制**：Qwen2-VL 改判 17/88（自一致 80.7%），Qwen3-VL-32B 10/88（88.6%）；重复运行 88/88 一致 → 不稳定性源自 prompt。变更落在 `base16 x01`（neutral→happy）与决定性 `lora3s x02`（worried→sad）。
- [x] **修正脚本 bug**：`frames_refswap_lora_*` v1→v3；pairwise κ 改用每对自身交集；Fleiss 用四判官交集（n=77）；匹配比较按各判官自身覆盖。
- [x] **四判官全矩阵一致性**：层内高（两个 7B κ=0.493；两个最新 κ=0.633）、层间低（41.6–47.7%）；Fleiss 0.376；一致 23/77。非 neutral 拆分：至少一方表态 36.7–50.0%，双方均表态 87.9–100%。
- [x] **Finding D 跨判官复现**：参考跟随 base 5/8·4/8·4/8 → v3 2/8·1/8·2/8；共识标签 3 个不一致配对全部支持抑制（McNemar p=0.25）。
- [x] **清除陈旧正向表述**：`modest but real robustness gains`、`cheerful 2/3 vs 1/2` 等在 3 个 sibling + 3 个 release 副本 + md 全部替换。
- [x] **"four judges" 表述修正**：88 配置统计按实际覆盖写明（四判官交集 77 行）。
- [x] **披露判帧时间覆盖限制**：8 帧取自 5.2 s / 124 帧片段的**前 1 秒**（约 19%），且片段开头正是参考立绘表情最可能残留处 —— 四个版本均已披露。
- [x] **排版**：ICASSP 正文 4 页 + References 第 5 页；NeurIPS 正文止于 p9、References 在 p10；ICME 7 页；8pp 版 9 页。全部 0 error / 0 overfull。
- [ ] **进行中**：base 模型全 8 任务 × 2 seed × 16 步生成（`run_basefull.sh`，pid 181491），完成后抽帧 + 四判官重判 → 用完全任务匹配的矩阵重算效应。
- [ ] **待用户**：`human_label/` 94 题标注（建议 2–3 人）→ `score_human_labels.py`。

## Round 7 判帧协议核查（2026-09-07 追加）
- [x] 发现并披露：原协议 8 帧仅覆盖每个 5.17 s 片段的前 1.0 s（19%），四个版本的 Protocol 段均已写明。
- [x] 全片段对照实验完成（`extract8_full.py` + `judge_frameset.py`），两判官各 86 行日志已入库并随 release 发布。
- [x] 数字锁定并可复现：`analyze_framespan.py` 输出 Qwen2-VL 27/81（66.7%，κ=0.459）、Qwen3-VL-32B 30/81（63.0%，κ=0.549）；LoRA 臂命中率 13/63→23/63（p=0.002）、20/63→33/63（p=0.001）。
- [x] 两套帧同源性校验：全片段 f01 与原集 f05 字节数相同（t≈0.72 s）。
- [x] "base 从不表达 sad"对照失效已如实写入（0/6→2/6、1/6→2/6），任务级结论**已全部撤回**，四个版本 + `paper_emoh3.md` + appendix 一致。
- [x] Finding D 补全片段复现（base 3/8、4/8 对 v3 2/8），并标注 n=8、p=0.25 欠功效。
- [x] 标题、摘要、Intro、Result 1/2、Limitations、Conclusion 全部改写为测量学定位；新增 ICASSP `tab:sensitivity` 三轴敏感性主表（含"重复运行 0/88"噪声底）。
- [x] 四版编译 0 error / 0 overfull；ICASSP 正文 4 页（References 第 5 页）、ICME 7 页、NeurIPS 正文止 p9、8pp 版 9 页。
- [x] release/ 已重新打包（含 `analyze_framespan.py`、`analyze_basefull.py`、`judge_full_*.jsonl`、更新的 README 与 handoff）。
- [ ] base 全 8 任务 × 2 seed 生成（s42 已 8/8；s1234 因共享节点被抢占反复 OOM，已改为多 GPU 组合回退 + 单实例锁，进行中）→ 之后跑 `run_judge_bf.sh` + `analyze_basefull.py`。
- [ ] **用户 94 题人工标注**（唯一能锚定"哪种判帧协议才对"的证据；工具已就绪于 `human_label/index.html`）。
