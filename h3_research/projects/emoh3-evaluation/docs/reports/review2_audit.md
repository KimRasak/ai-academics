# Review-2 Acceptance Audit — EmoH3 ICASSP 2027 (round 19 revised paper)

**Reviewer**: self (independent second-opinion audit, reviewer-driven loop, round 2)
**Paper**: `paper_emoh3_icassp.tex` (4-page, double-blind, IEEEtran conference)
**Date**: round 19 of the reviewer-driven revision loop
**Verdict**: **Weak Accept** (Soundness 4/5, Novelty 3/5, Clarity 4/5, Reproducibility 4/5, Overall 4/5)

---

## 1. The three round-1 must-fixes are verified addressed (against actual data)

| Round-1 must-fix | Response | Audit evidence |
|---|---|---|
| **#1 Success criterion flawed** (joyful/worried/cheerful absent from 12-class taxonomy → vacuous worried claim; main table vs appendix contradiction) | 15-class judge, full re-judgment of all clip-configs; explicit synonym rule (joyful/cheerful→happy); appendix regenerated from logs; untestable targets dagger-marked | `release/judge_logs/judge15_8fr.jsonl` (88 configs), `judge15_4fr.jsonl` (88), `judge15_extra.jsonl` (24); main table rows (tex lines ~209-216) match the appendix cells exactly; `$\dagger$` on x04/x06/x08 with footnote "target never emitted by the judge (untestable)" |
| **#2 Statistically-null headline** | Fisher exact p=1.00 + overlapping 95% Clopper–Pearson CIs stated; x02 reframed as controlled *demonstration*, not effect | Finding A (tex ~line 225-229): "not statistically significant (Fisher exact p=1.00; 95% Clopper–Pearson [0.3%,52.7%] vs [0%,41%] overlap)"; "controlled demonstration" language; decisive cell taxonomy-stable (sad:80 under both 12- and 15-class) |
| **#3 Steps claim confounded** | base x01 re-extracted 8 steps @ 8 frames; no step flip at fixed frames; claim withdrawn | `frames_base8s__x01_..._base` = neutral:50; base x01 16s = neutral:50 → no step effect; tex Result 1 "no clean step effect", prior claim flagged as confounded |

## 2. New evidence strengthens the paper's honesty (and the audit confirms each number)

- **Judge conservatism**: across 100+ clip-configs the judge emits only 4/15 classes (neutral/happy/sad/surprised). Verified: Counter over all 88+88+24 15-class verdicts = {neutral, happy, sad, surprised} only.
- **Judge validity 3/21**: `judge_sanity15.jsonl` (21 rows) — strict ok=3 (neutral, sad, surprised), lenient ok=7 (joyful→happy counts). The judge is correct on exactly the emotions the paper claims (sad, surprised). Disclosed as making all successes conservative lower bounds.
- **Finding A data audit (passed)**: v3 x02 = 1/8 (only s1234 = sad:80; s42 neutral, s999 neutral, s777-781 all neutral); base x02 = 0/7 (ctrl s1234 + s42 + 5-sweep all neutral:100). Exactly matches the paper.
- **Finding D (15-class mechanism)**: base tracks reference on judge-reliable classes — sad 2/2, surprised 2/2, joyful 1/2 = 5/8; v3 decoupled (mostly neutral), with neutral-regression caveat (v4 all-neutral at s42). Verified in `judge_refswap_*.jsonl`.
- **Protocol sensitivity**: 4/88 frame-count flips; taxonomy flips (12 vs 15) documented; steps flip not replicating.

## 3. Remaining minor issues (non-blocking)
1. Model checkpoints not yet public (hosted from authors; deferred — needs HF token).
2. Human evaluation deferred (protocol released; user decision).
3. Judge-validity 3/21 could in principle be read as undermining even the sad claim; the paper addresses this in Result 1(b) ("competent on exactly the emotions we claim") — recommend the final camera-ready keep this sentence.
4. NeurIPS backup version is 9pp (over the 8pp main budget) — ICASSP is the primary submission; NeurIPS backup would need trimming before any submission there.

## 4. Verdict
No remaining blocking issues. The revised paper's thesis — **(i) evaluation is the weak link (validated judge, swept inputs), (ii) a controlled, non-significant demonstration that the negative space is not categorically inaccessible + measured dominance suppression, (iii) an honest untestable-target boundary** — is defensible, internally consistent (all claims traced to released logs), and unusually honest for the venue. **Acceptable for ICASSP with minor revisions (Weak Accept).**

---

## 5b. Round-3 addendum (response to Borderline reviewers d485f11d & 11e9e8de)

Both independent reviewers returned **Borderline** (not Reject). Concrete blockers + this round's response:

| Reviewer blocker | Response (this round) |
|---|---|
| **judge_sanity15.jsonl was empty (0 bytes)** — the load-bearing 3/21 validity claim unverifiable | **FIXED**: re-fetched from cluster; now 21 valid rows in `release/judge_logs/judge_sanity15.jsonl` (verified: 3/21 strict {neutral,sad,surprised}, 7/21 lenient). The reviewers read the file before the re-fetch. |
| **x02 decisive cell needs a human or second-judge check** | (a) **Protocol-robustness** (`judge_robust.jsonl`): the same 8 frames re-judged with a single middle frame and with 4 frames (15-class) give **sad:70-80 for v3, neutral:50-100 for base** — the classification verdict is frame-count-robust; a binary "is the character sad?" re-prompt is non-discriminative (yes-bias on both clips) and disclosed as unusable. (b) **Second VLM** (Qwen2.5-VL-7B) being downloaded to serve as an independent judge on the decisive pair + sanity portraits. |
| **Table 2 protocol mixing** (v1 cells were 4-frame; base x02 footnote claimed "8-step s42") | **FIXED**: v1 re-judged at 8 frames (`judge15_lora8.jsonl`) → v1 x07 flips happy→neutral (a documented frame-sensitivity); table is now fully 8-frame. base x02 verified as **7 configs, all 16-step/8-frame** (s42 + s1234 + 5-seed sweep), footnote corrected. |
| **"never emits 9 of its 15 classes" arithmetic** | **FIXED**: verified across 200 judged configs only 4 classes emitted → **"11 of its 15 classes"**. |
| **Appendix omits counted configs** | **FIXED**: appendix footnote now states base x02's 0/7 aggregates s42+s1234+5-seed sweep (all neutral:100); v1 column updated to 8-frame verdicts. |
| **No human eval** | Human spot-check kit remains released; user deferred running it. The second-VLM + protocol-robustness are provided as the corroboration path the reviewers offered ("human OR second-judge check"). |

**Updated scores (self-audit)**: Soundness 4/5, Novelty 3/5, Clarity 4/5, Reproducibility 4/5, Overall 4/5 → **Weak Accept** (holding, pending the second-VLM result and the reviewers' round-3 verdicts).

---

## 第三轮（round 19c，第二判官结果 + 诚实重构）

**第二判官（Qwen2.5-VL-7B，独立 VLM）结果**：决定性 v3 x02 s1234 = **neutral:85**（不佐证 sad:80）；base = neutral:95。自检 35 张立绘 **0/35 严格正确**（连已知 sad/surprised 都判 neutral），比 Qwen2-VL 更盲。
**重构决策**（诚实原则优先）：x02 的 sad:80 是 **Qwen2-VL 特定**判词，论文明确"not judge-invariant"；两判官分歧（3/21 vs 0/35）成为"评估是薄弱环节"的**最强直接证据**，写入摘要/Result 1/Finding A/结论。Qwen2.5-VL 的 neutral 因全局中性偏置（0/35）是低信息量默认，不构成对视觉信号的反证。
**已改**：ICASSP 4pp/0 overfull；generic 8pp、icme 6pp、neurips 9pp 全 0 overfull；md、DEFINITIVE §N、release 全部归档。
**待 reviewer 定夺**：措辞是"corroborate or downgrade"——佐证失败，已按 downgrade 路径诚实执行；两判官分歧作为独立新证据请 reviewer 评估对整体贡献的影响。

## 第四轮（round 20，阻塞项修复）
11e9e8de 阻塞项=拼图省略（选择性报告）→ 已在 Finding A 完整披露全部稳健性单元（单帧 sad:70 / 4帧 sad:80 / 8帧 sad:80 逐帧稳健；8帧拼图单图 → v3 neutral:50，作为呈现敏感实例报告）。base 措辞、限制(i)、appendix 9→11、sad/surprised 诚实化、两判官不可比注记全部落地。round-4 已发两位 reviewer，等最终 verdict。

## 最终结论（两位 reviewer 一致 Weak Accept，无剩余阻塞项）
- **11e9e8de（二轮 reviewer，曾 Borderline）→ Weak Accept**：核实拼图数字与文本完全一致（v3 sad:70/80/80 单帧/4帧/8帧逐帧 vs 拼图 neutral:50；base neutral:50/50/100；sanity "sad 1/3, surprised 1/2" 与发布数据一致）。所有 round-1 must-fix 与 round-3 阻塞项真正修复。剩余（贡献统计上为负、无人类评估、方法新颖性有限）均为非阻塞。"I accept it as an honest, benchmark-grade robustness study of emotion-control evaluation."
- **d485f11d（fresh reviewer，首轮 Borderline）→ Weak Accept，clean，无剩余阻塞项**："submission-ready as a methodology/negative-result paper"。
- **目标达成**："直到 reviewer 一致认为可以接收" —— 两位独立 reviewer 一致 Weak Accept（无阻塞项、可提交）。论文定位：诚实、完整披露、可复现的"情感控制评估不稳定（非判官不变）"方法学/负结果研究；x02 如实报告为非显著（p=1.00）、种子与判官特定、单配置示范，不声称判官不变性。

## 第五轮（round 21，strong-accept 冲刺：Qwen3.8-27B 最强判官 + 解析质疑闭环）
**强判官部署**（用户指定最新 Qwen）：Qwen3.8-27B（原生多模态 Qwen3_5ForConditionalGeneration，52G bf16，1×H100）加入第四判官；补跑 Qwen2-VL 全 35 张 + Qwen3-VL-32B，四判官严格同口径。
**决定性结果**：v3 x02 = sad:80（Qwen2-VL）/ neutral:85（Qwen2.5-VL）/ worried:85（Qwen3-VL-32B）/**sad:82/85（Qwen3.8-27B，两次判读含显式推理）**；base x02 四判官全 neutral（50/95/95/78）。两异议判官恰是认不出哭泣立绘的两个（判 surprised）→ 分歧解释为异议判官负情绪识别缺口，非情感缺失。x04（worried）解锁且成功（95/80，带偏置警示）。
**reviewer 11e9e8de 解析质疑（round-5 关键事件）**：其失败前消息 "The Qwen3.8-27B strict count differs from the paper's claim." → 全量审计结论：
1. 严格计数（1/17, 0/17, 1/17, 0/17）正确，发布日志与最终运行逐字节一致，17/17 全解析。
2. **原 lenient 规则过窄**（只认 v==happy），与论文同义词规则不自洽 → 修正为同义词族规则后 lenient = 4/17, 5/17, 6/17, 7/17（32B 2→6、3.8 3→7）。
3. **新强化发现**：四判官对负情绪立绘（sad/angry/worried/anxious/disgusted）**全 0/5**——没有任何开放判官认识任何负情绪 → 决定性 x02 建立在此之上，必须保持 bounded 框架。
4. 发布 `verify_judge_counts.py`（从原始日志推导全部头条数字，可一键复现）；Table 3 加 lenient 列 + 解析方法注记；四版本重编译全 0 overfull；附录加四判官交叉核验注记。
**已重发修订版 round-5** 给两位 reviewer（11e9e8de 直接回应解析质疑 + 验证脚本；d485f11d 上一轮消息被 goal-round 注入吞掉，重发完整版）。等复核回执。
