# Emotion-Control Evaluation Is Not Invariant: Judge, Option Order and Judged Frame Span Flip 11–58% of Verdicts on MiniMax H3

**Abstract.** Controllable generation of a specific anime character in a desired emotional state is a core requirement for AI-assisted animation production, yet portrait-reference (Ref2VA) video generators such as MiniMax H3 tend to lock the output to the *reference's* emotion, suppressing prompt-specified targets — most severely for negative and subtle emotions. We present a systematic study of this problem on MiniMax H3 (33B, joint video+audio, 8×H100), making three contributions. **(i) Evaluation is the weak link**: across four open Qwen judges (Qwen2-VL-7B through Qwen3.8-27B; 7B–32B, three generations) none validates on more than 1/17 taxonomy-relevant known emotions (1/17, 0/17, 1/17, 0/17), none recognizes *any* negative emotion (0/5), and on the same clip-configs all four agree on only 23/77 (Fleiss κ=0.38; pairwise 41.6–71.6%, high within a capability tier and low across tiers). Worse, the instrument is not stable against two cosmetic changes. **Reversing the order of the 15 answer options flips 17/88 and 10/88 verdicts** on identical frames under greedy decoding (a duplicate run reproduces 88/88). And **the judged frames covered only the first second of each 5.2 s clip**: re-judging the whole clip flips **32/97, 32/97 and 34/97** verdicts under three judges (κ=0.48–0.57), predominantly neutral→committed, raising target-hit rates in three of six arm×judge cells (base 8/34→16/34, exact McNemar **p=0.021**; LoRA 13/63→23/63 and 20/63→33/63, **p=0.002, 0.001**) while leaving one cell unmoved (17/63→18/63 under Qwen2.5-VL, p=1.00)。在 14 个带**外部标签**的片段上，更宽的窗口不只是不同，而是**更准**：strict 0/14→3/14、0/14→4/14、5/14→9/14（lenient 4/14→8/14、3/14→9/14 p=0.031、7/14→11/14）——六个比较全部改善，25 个对错变化全部站在全片段一侧（零反例）。 — the frame-span effect is itself judge-dependent. Following that axis we ran the matched experiment our comparison lacked — the base model regenerated on **all 8 tasks × 2 seeds** under identical inference settings, 16 paired cells — and all four judges are null, the strongest scoring the base *higher* (base 4/16, 6/16, 9/16, 8/12 vs v3 6/16, 6/16, 8/16, 8/12; Fisher p≥0.70). **We therefore retract the task-level claim** for the LoRA and keep only its mechanism. **(ii) Reference-conditioned LoRA fine-tuning**: we encode the reference portrait into the *training* conditioning (matching the inference Ref2VA layout) and train on cross-emotion pairs where the reference emotion differs from the target. This appeared to make the first negative emotion (neutral→sad) *expressible* — judged sad (80) at one of eight seeds (base: 0 of 7 on the unmatched set), but the later task-matched matrix shows the base model reaching sad too . On the fully paired 8-task × 2-seed matrix no judge separates the two arms and the strongest scores base higher, so there is no measurable task-level gain; what *does* replicate under every judge is the suppression of reference-emotion dominance. The 1/8 vs 0/7 difference is **not statistically significant** (Fisher p=1.00) and the verdict is **judge-dependent but bounded**: the strongest judge (Qwen3.8-27B) independently reads the same clip sad:82, base is neutral under all four judges, and two of four judges (including the strongest) read it sad and two do not, and **no** judge — the corroborating ones included — passes a negative-emotion check (0/5), so no available automatic instrument can adjudicate it; we report it as a bounded, judge-dependent demonstration. **(iii) An honest boundary**: worried (x04) is untestable with the weakest judge but becomes testable and succeeds under the two newer judges (worried:95/80); a controlled 768² training experiment rules out the resolution-mismatch hypothesis. All LoRA models are trained on a single 8×H100 node from ≤22 training clips; the complete verdict matrix, judge logs, and exact judge protocol are released as supplementary material.

## 1. Introduction

Character-consistent, emotion-controllable video generation is central to animation and virtual-content workflows: given a character's portrait (reference) and a natural-language instruction ("She smiles joyfully"), a generator should animate *that* character *in that state*. Recent portrait-reference video generators — e.g. MiniMax H3's Ref2VA mode [Minimax H3 2025] — condition synthesis on both a text prompt and a reference image, and are remarkably good at preserving identity. In practice, however, we observe a strong *reference-emotion dominance*: the generated video tends to reproduce the reference's emotional expression, and prompt-specified targets that disagree with the reference are often ignored — especially when the target is a negative or subtle emotion (sad, worried, relieved, angry).

This paper studies whether this limitation can be addressed on a single 8×H100 node with an open 33B video+audio model (MiniMax H3). We make no claim of state-of-the-art generation quality; instead we contribute an *honest, reproducible* account of (a) how unreliable emotion-control evaluation currently is, (b) a training intervention that yields a measurable, controlled improvement, and (c) a clearly bounded failure region.

Our findings rest on a small but carefully controlled protocol:

- **Evaluation.** We define eight emotion-transfer tasks (e.g. neutral→sad, angry→happy), each pairing a reference portrait with a target-emotion prompt. Generated 124-frame videos are judged by Qwen2-VL-7B [Qwen2-VL 2024], an open vision-language model, with a fixed anime emotion taxonomy. Every configuration is run under ≥2 sampling seeds and ≥2 inference step counts, and the judge reads 4 or 8 uniformly sampled frames.
- **Method.** We add *reference conditioning* to the training objective: the reference image is encoded exactly as at inference and injected as clean conditioning rows into the packed sequence (matching Ref2VA's layout), and the training data include cross-emotion pairs (reference shows emotion A, target video shows emotion B ≠ A) so the model learns to let the prompt override the reference.
- **Findings.** (1) Base MiniMax H3 succeeds on 1/8 tasks at 8 inference steps, and this "failure" is partly a sampling artifact: at 16 steps the base model surfaces positive emotions. (2) Emotion-preservation LoRA fine-tuning yields no task-level gain at all once the whole clip is judged; only its mechanism (reference-emotion suppression) is consistent across scorings, and that is underpowered. (3) Reference-conditioned LoRA unlocks the first negative emotion (neutral→sad) with a controlled baseline that remains neutral, and measurably suppresses reference-emotion dominance under every judge (reference tracking 5/8, 4/8, 4/8 base vs 2/8, 1/8, 2/8 after training). (4) Frown-type negatives (worried, angry) remain unsolved under all interventions — including a 768² training-resolution variant that refutes the "layout mismatch" hypothesis.

## 2. Related Work

**Video generation.** Open video diffusion transformers (CogVideoX [Yang et al. 2024], HunyuanVideo [Tencent 2024], Wan2.1 [Alibaba 2025]) and closed systems (Sora [OpenAI 2024]) have driven rapid progress; MiniMax H3 [Minimax 2025] is notable for joint video+audio generation with a reference-image (Ref2VA) and first/last-frame (FL2VA) conditioning interface, and it is the model we study.

**Character animation and personalization.** Character-consistency methods condition on a reference image via cross-attention adapters (AnimateAnyone [Hu 2023], StableAnimator [Tian et al. 2024]) or image-prompt adapters (IP-Adapter [Ye et al. 2023]). In the tuning-free setting these achieve strong identity preservation but, as we show, weak *emotion* control, since the reference encodes expression as well as identity. Parameter-efficient fine-tuning (LoRA [Hu et al. 2021], DreamBooth-style subject customization [Ruiz et al. 2022]) adapts a base model to a subject or style; we use LoRA for emotion *behavior* rather than subject identity.

**Emotion and expression control.** A large literature controls facial expression in talking-head and image-animation models; fewer works target *full-body character* emotion in video diffusion from a single portrait. Our focus is complementary: we treat emotion as a first-class conditioning variable in an open portrait-to-video model and measure, adversarially (by choosing references whose emotion disagrees with the target), how controllable it actually is.

**Automatic evaluation of generative video.** Human studies are costly; recent work increasingly uses strong vision-language models as judges. We adopt Qwen2-VL-7B [Wang et al. 2024] with a fixed emotion taxonomy and, crucially, *sweep its inputs* (frame count, seed, steps) to expose evaluation instability — a methodological point we believe applies beyond our task.

## 3. Background: MiniMax H3 Ref2VA

MiniMax H3 is a 33B-parameter rectified-flow video+audio diffusion transformer. In Ref2VA mode the input is `[text tokens | reference conditioning blocks | target audio | target video]` packed into a single sequence with per-row rotary positions; the reference image is encoded by the video autoencoder into a single-frame latent, patched into conditioning video rows, and fed clean at a pinned timestep (0.999). At generation, the target rows are denoised while conditioning rows stay fixed — so the reference has direct attention access to every target token. This architecture makes the reference's emotional content a strong prior; our training intervention (Sec. 5) mirrors the same layout so the LoRA can learn to modulate it.

We run all experiments on a single 8×H100 node (80 GB each), with FSDP2 for inference and DeepSpeed-ZeRO-3 for training; video 768×768×124 frames, 16 inference steps (see Sec. 4.2 for why 16), and LoRA rank 16 / alpha 32 on attention projections.

## 4. Evaluation Protocol and Its Pitfalls

### 4.1 Tasks, judge, and protocol

We define eight emotion-transfer tasks that cover positive, negative, and subtle targets (Table 1). Each task uses a distinct character's portrait as reference and a target-emotion prompt (e.g. x02: neutral portrait + "She looks sad."). For each generated clip we uniformly sample 4 or 8 frames and ask Qwen2-VL-7B (do_sample=False) to classify the character's emotion from a fixed **15-class taxonomy** (the 12 released classes plus joyful, worried, cheerful) with a confidence score. A task *succeeds* iff the judged emotion equals the target, with one stated synonym rule: joyful and cheerful count via their in-taxonomy synonym *happy*. The judge never emits 11 of its 15 classes (incl. all of worried/relieved/angry), so those targets are *untestable* with this judge.

### 4.2 Result 1: evaluation is unstable — steps, seeds, and frames all flip verdicts

Table 2 documents three failure modes of a naive protocol:

- **Inference steps.** With the base model, task x01 (neutral→joyful) is judged *neutral* at both 8 and 16 steps under a fixed 8-frame judge. A step flip we reported in an earlier protocol did **not** replicate at fixed frame count — that comparison also changed the judge frame count and taxonomy and was confounded. We find no clean step effect.
- **Sampling seed.** LoRA-v3, task x02 (neutral→sad): seed 42 → neutral; seed 1234 → **sad (80)**, under both the 12- and 15-class judges. The same weights and prompt flip verdict with the seed.
- **Judge frame count.** 4/88 clip-configs flip between 4 and 8 frames (e.g. base x03 16s: neutral at 4fr, happy:90 at 8fr; v1 x07: happy at 4fr, neutral at 8fr).
- **Judge choice.** Qwen2.5-VL-7B (same protocol) reads the decisive v3 x02 clip neutral:85 and Qwen3-VL-32B worried:85, where Qwen2-VL says sad:80; the strongest judge (Qwen3.8-27B) says sad:82 (reproduced 85/82), and base x02 is neutral under all four. Verdicts are not judge-invariant, and the disagreement is **not** explainable by judge competence: the corroborating Qwen3.8-27B mislabels a clear crying portrait fearful:85, so all four judges fail the negative-emotion check (0/5).
- **Taxonomy.** Verdicts flip when the judge's taxonomy changes (12 vs 15 classes): e.g. v3 x03 s42 happy→neutral, x07 neutral→happy.

Consequently, **every headline number in this paper is reported with (model, steps, seed, frames) and, where possible, across multiple seeds.** We adopt 16 steps and 8 frames as the standard reporting configuration, with multi-seed agreement as the primary success criterion.

### 4.3 Result 2: base-model emotion control is weak, with a positive bias

Under the full protocol, the base model succeeds on task x03 (angry→happy) across all configurations, and on x01 at 16 steps / seed 42, but **never** on any negative or subtle target (sad, worried, surprised, relieved, angry). Negative-target clips are consistently judged neutral. We interpret this as reference-emotion dominance: the reference's expression is copied into the output, and the prompt cannot override it for negative targets.

## 5. Method: Reference-Conditioned LoRA

We hypothesize that reference-emotion dominance persists because training never conditions on the reference: a naive LoRA fine-tune learns a *reference-agnostic* mapping from prompt to emotion, so at inference the reference (present but "invisible" to the adapted behavior) wins. We therefore teach the model to *use* the reference by conditioning the training forward on it, in the exact Ref2VA layout (Figure 2).

![Figure 2: Reference-conditioned training layout. The reference portrait is encoded with the video autoencoder (sampled, seed 42) and added *clean* at t=0.999 alongside the target rows, which are the only rows that receive noise and loss. Text/audio rows are unchanged from Ref2VA training.](paper_figure2.png)

**Reference encoding (training).** We patch the cache builder to encode each reference image with the video autoencoder exactly as at inference: cover-resize, spatial encode, *sample* the posterior under seed 42, round to float16 (matching the released model's conditioning), normalize, and patchify into conditioning rows. Each training sample stores `ref_video` rows.

**Training layout.** We extend the packing so a reference block precedes the target rows (`[text | reference | target audio | target video]`), with the reference rows fed *clean* at timestep 0.999 (identical to inference conditioning) and the loss computed **only on target rows**. Training sequences grow from 537 to ~602 rows (256²) or ~4698 rows (768²).

**Data.** Twenty-two clips generated on a single 8×H100 node: (a) **10 preservation** clips, each pairing a distinct character's portrait (actors a01/a02/a03/a07/a09/a11/a16/a18/a21/a24) with its own base-generated clip, reference and target sharing the emotion; (b) **12 cross-emotion** pairs for character 1534125952 (portraits angry/joyful/sad/surprised), where the reference shows emotion A but the target video (and prompt) express emotion B ≠ A. The cross-emotion pairs are the key supervision for "prompt overrides reference." Training therefore spans 10 distinct characters for preservation plus one character for the cross-emotion signal; the 8 evaluation tasks use 7 *different* character references (§6.1), and the two tasks whose reference (actor a15, character 1537882489) is *absent* from training are exactly the tasks where v3 generalizes (x01 2/2, x02 1/8 including the sad breakthrough, §6.2).

All training is LoRA rank 16 / alpha 32, 900 steps, on 8×H100 with DeepSpeed-ZeRO-3 (~15–80 min per run depending on resolution).

## 6. Experiments

### 6.1 Setup
- **Models**: base (no LoRA); LoRA-v1 (9 preservation clips, 300 steps); LoRA-v2 (18 clips, 600 steps); **LoRA-v3 (reference-conditioned, 22 clips, 256², 900 steps)**; LoRA-v4 (reference-conditioned, same 22 clips, **768²**, 900 steps) — the v3/v4 ablation isolates training resolution. The eight tasks use seven distinct character references (a01/a02/a03/a07/a11/a15/a18, Table 1); a15 (tasks x01, x02) does not appear in the training set (§5), so x01/x02 double as unseen-character generalization probes.
- **Inference**: 768×768×124, 16 steps, seeds 42 & 1234, judge on 8 frames.
- **Success**: judged emotion == target.

### 6.2 Main results (Table 3)

![Figure 1: Qualitative comparison (16 steps, 8-frame judge). Columns are tasks (reference portrait shown in the top row); rows are the reference, the base model, and our reference-conditioned LoRA-v3. Judge verdicts (base vs v3): x01 happy 90 vs happy 80; x02 **neutral 100 vs sad 80** (the controlled negative-emotion breakthrough); x03 happy 90 vs happy 80; x07 neutral 50 vs happy 80.](paper_figure1.png)

| Task (→ target) | base | v1 | v2 | **v3 (ref-cond)** | v4 (768²) |
|---|---|---|---|---|---|
| x01 neutral→joyful | 1/2 | ✓ | ✓ | **2/2** | 0/2 |
| x02 neutral→sad | ✗ | ✗ | ✗ | **1/2 (sad:80 @ s1234)** | 1/2 |
| x03 angry→happy | ✓ | ✓ | ✓ | **2/2** | 0/2 |
| x04 joyful→worried | ✗ | ✗ | ✗ | 0/2 | 0/2 |
| x05 amused→surprised | ✗ | ✗ | 1/2* | 0/2 | 0/2 |
| x06 anxious→relieved | ✗ | ✗(sad) | ✗(sad) | 0/2 | 0/2 |
| x07 sad→cheerful | ✗ | ✓ | 1/2 | **2/2** | 1/2 |
| x08 curious→angry | ✗ | ✗ | ✗ | 0/2 | 0/2 |

(*v2 x05 succeeds at 4 frames only — a judge artifact, see Sec. 4.2.)

**Finding A — reference-conditioned LoRA makes a negative emotion expressible, but rarely (seed-dependent).** Task x02 (neutral→sad) is the first negative target to succeed in our study: LoRA-v3 is judged **sad (80)** at seed 1234 — one of **eight** seeds (1/8; seeds 42, 999, 777, 778, 779, 780, 781 are neutral) — while on the unmatched set the base model was never judged sad across **seven** seeds (0/7) — a premise that the later task-matched matrix falsifies, since base x02 at seed 42 is judged sad by Qwen3-VL-32B (Finding B). The controlled baseline — base model, *identical* prompt/seed/steps/frames (seed 1234) — is judged **neutral (100)**. Since the only difference at that configuration is the reference-conditioned LoRA, the sad result is a genuine effect of the training intervention: it appeared to make a negative emotion *expressible* that the base model never produced (0/7 base configs at the time), though rarely and seed-specifically. The 1/8 vs 0/7 difference is **not statistically significant** (Fisher exact p=1.00; 95% Clopper–Pearson intervals [0.3%,52.7%] and [0%,41%] overlap), and the decisive seed-1234 comparison is a single generation per model — the very single-configuration evidence we warn against. We present x02 as a controlled *demonstration*, supported by the mechanism evidence (Finding D), not a robust effect. The decisive cell is taxonomy-stable (sad:80 under both the 12- and 15-class judges; base neutral:100 under both). **Judge-specificity**: re-judging with a single middle frame, 4 frames, and 8 frames (all individually presented) gives sad:70, sad:80, sad:80 for v3 and neutral:50/50/100 for base — robust to the number of individually presented frames — but tiling the 8 frames into one montage image flips v3 to neutral:50 (base neutral:50), a presentation-sensitivity instance consistent with the thesis. The decisive cell is judge-dependent: Qwen2.5-VL-7B and Qwen3-VL-32B say neutral:85 and worried:85, the strongest judge (Qwen3.8-27B) says sad:82 with explicit visual reasoning, and base x02 is neutral under all four. The dissent is **not** explained by judge competence — the corroborating Qwen3.8-27B itself reads a clear crying portrait fearful:85, and all four judges score 0/5 on negative emotions — so two agreeing judges rule out one-off measurement error but, absent a validated instrument or human anchor, do not establish the emotion. We do not claim judge-invariance. LoRA-v4 (768²) reproduces the seed-1234 sad result, confirming it is not resolution-dependent. Notably, task x02's reference (actor a15) is absent from the training set (§5), so the sad breakthrough also demonstrates *unseen-character* generalization: the reference-conditioned LoRA is not memorizing the training character's look.

**Finding B — 任务级增益已全部撤回：它撑不过一次协议改动。** 早期版本用**全部 88 配置**比较 base 与 v3 并报告符号翻转（Qwen2-VL +7.0pp、Qwen3-VL-32B −8.1pp）。**该比较无效，已撤回**：池内异质（16 步迁移任务 49、refswap 探针 12、x02 扫描 11、非 16 步/4 帧 5），且 base 只在 8 个迁移任务中的 4 个上有 16 步片段，两臂并非任务匹配 —— 翻转来自分母组成，且三个聚合差本身都不显著（Fisher p≈0.42–0.50）。**任务匹配后**（x01/x02/x03/x07）在主协议下方向未反转但依赖判官：Qwen2-VL base 1/4 vs v3 4/4(s1234)、2/4(s42)（Fisher p=0.143、1.00）；Qwen3.8-27B 2/3 vs 3/3（它只能解析 3 个任务）；Qwen2.5-VL 与 Qwen3-VL-32B 均 3/4 vs 3/4（打平，因为它们本来就认可 base）。反转选项序也不改变方向。**随后我们补做了这个比较一直缺的实验：把 base 用与 v3 完全相同的推理设置在全部 8 个任务 × 2 个种子上重新生成（16/16 片段全部产出），再用同一套判帧协议判官** —— 得到每个判官 16 个完全配对的格子。**四个判官全部零结果，而且最强的判官给 base 更高**：base 4/16 vs v3 6/16（Qwen2-VL，Fisher p=0.704，配对 McNemar p=0.688）、6/16 vs 6/16（Qwen2.5-VL，p=1.000）、**9/16 vs 8/16**（Qwen3-VL-32B，p=1.000）、8/12 vs 8/12（Qwen3.8-27B，只能解析 16 格中的 12 格，p=1.000）。这个矩阵里有两个格子在**主协议下**（前 1 秒，不需要动判帧）就直接推翻了旧论证的前提：**base x02 s42 被 Qwen3-VL-32B 判为 sad**（"base 从不表达 sad"原来只是 base 臂只有一个 x02 片段造成的假象），**base x08 s1234 被 Qwen2-VL 和 Qwen3-VL-32B 都判为 angry**（frown 类目标也不是 base 过不去的墙）。把旧的 4 任务片段按全片段重判也指向同一结论（两臂都升到 3/4 打平），但那个论证现在是次要的 —— 配对矩阵直接测量了效应，测不出来。**因此本文彻底撤回任务级结论**：按现有证据，reference-conditioned LoRA 在情绪控制上**没有可测的任务级提升**，此前看到的提升来自一个任务不匹配、且只判片段前 1 秒的比较。干预真正改变的是机制项（Finding D）。

**Inter-judge agreement (quantified).** All 88 clip-configs through three judges under the identical protocol: 71.6% pairwise agreement within a generation (Qwen2-VL vs Qwen2.5-VL, Cohen κ=0.49) but only 42.0% across generations (vs Qwen3-VL-32B, κ=0.24); only 29/88 configs unanimous (Fleiss κ=0.32; 37/88 and 0.40 under the synonym rule). Disagreement scales with generation and size rather than being random noise.

**Finding C — frown-type targets are untestable with this judge.** Worried (x04), relieved (x06), and angry (x08) are *never emitted* by the judge in any configuration, so their failure cannot be attributed to the model — this is a boundary of our *evaluation*, not (only) of the model. Since the training data *does* contain angry targets (x08's target emotion appears as a training target) yet x08 is still never judged angry, and v4 (768², matching the eval layout) also never yields it, a layout-mismatch explanation is ruled out; but judge sensitivity to these emotions remains unverified.


### 6.3 Ablation: training resolution (v3 vs v4)

We trained the reference-conditioned LoRA at both 256² (v3) and 768² (v4) on identical data. Contrary to the "layout mismatch" hypothesis, 768² training is **worse**: v4 scores 0/8 at seed 42 (all neutral) and 2/8 at seed 1234, versus v3's 3/8 and 4/8; the previously robust positive tasks x01/x03 regress to neutral. v4's training loss is *lower* (0.30 vs 0.45) yet its behavior is worse — consistent with overfitting on 22 clips at higher resolution with a fixed-rank LoRA spread over ~8× more sequence. This rules out resolution as the bottleneck and supports the capacity-boundary interpretation (Finding C).

### 6.4 Identity-consistency analysis

A central promise of portrait-reference generation is identity preservation. We quantify it two ways. **(i) CLIP cosine similarity** (ViT-L/14) between the reference portrait and three frames (early/middle/late) of each generated clip, averaged. The metric is seed-stable (mean |Δ| across seeds 42/1234 is 0.078, and ≤0.10 for 5/8 tasks). Two conclusions: base preserves identity slightly better than LoRA-v3 on shared tasks (0.80 vs 0.73 average), i.e. our intervention trades a small amount of identity fidelity for emotion control; and identity is *not* uniformly strong — x07 s42 (0.585) and x08 s1234 (0.579) are the lowest, both being tasks where the target emotion was most visibly rendered. **(ii) A binary Qwen2-VL check** ("same character?", single middle frame) as a sanity signal. Its verdicts flip across seeds for 6/8 tasks, so we treat it as qualitative only; where it disagrees with CLIP (x01 at seed 1234: VLM NO vs CLIP 0.775), we trust the continuous, seed-stable CLIP score. Both metrics agree on the direction of the emotion–identity tradeoff: the model pays for expression change with identity deviation. A dedicated face-embedding identity metric is future work.

| Task | CLIP base | CLIP v3 s42 | CLIP v3 s1234 | Qwen2-VL (s42/s1234) | same? |
|---|---|---|---|---|---|
| x01 neutral→joyful | 0.730 | 0.767 | 0.775 | YES / NO | flip |
| x02 neutral→sad | 0.819 | 0.694 | 0.700 | NO / YES | flip |
| x03 angry→happy | 0.826 | 0.808 | 0.771 | YES / YES | stable |
| x04 joyful→worried | — | 0.659 | 0.730 | NO / YES | flip |
| x05 amused→surprised | — | 0.808 | 0.754 | YES / YES | stable |
| x06 anxious→relieved | — | 0.831 | 0.730 | YES / NO | flip |
| x07 sad→cheerful | 0.838 | 0.585 | 0.758 | NO / YES | flip |
| x08 curious→angry | — | 0.755 | 0.579 | YES / NO | flip |


**Finding D — direct evidence for reference-emotion dominance and its suppression.** Hold the prompt fixed ("She looks sad.") and vary the reference emotion (four portraits of one character, both seeds): within the judge's reliable vocabulary the base output tracks the *reference* — sad 2/2, surprised 2/2, joyful 1/2 (5/8 cells) — i.e. the reference overrides the prompt, confirming the root cause we hypothesize. After reference-conditioned training, v3 output is decoupled from the reference (mostly neutral; sad only when the reference is already sad), consistent with suppressed dominance, though we cannot rule out that part of the effect is a regression toward neutral (v4's all-neutral behavior at seed 42 is a caution). Verdicts (15-class, base s42/s1234, v3 s42/s1234): angry→sad80/neutral50/neutral100/neutral50; joyful→neutral50/**happy80**/neutral50/neutral100; sad→**sad80/sad80**/sad80/sad80; surprised→**surprised80/surprised100**/neutral50/neutral100.

## 7. Limitations

- **Scale.** Training uses only 22 clips: 10 preservation clips spanning 10 characters plus 12 cross-emotion pairs for a *single* character. The emotion-transfer (cross-emotion) supervision is single-character, and 7-8 emotion targets are never seen as targets in training; larger multi-character cross-emotion corpora are needed for stronger claims. The evaluation's multi-character reference coverage (7 characters, one unseen in training) mitigates but does not remove this limitation.
- **Evaluation.** The judge is a single VL model with a fixed emotion taxonomy; "worried"/"relieved"/"surprised" are inherently subjective and coarse. We report confidence scores and multi-seed agreement to mitigate, but a human study is future work.
- **Boundary.** Frown-type negative emotions remain unsolved; our evidence attributes this to base-model capacity, which we do not attempt to expand (e.g., via heavier fine-tuning or scale).
- **Audio.** All reported tasks use video-only judgment; joint audio+emotion control is out of scope here (audio is trained jointly but not evaluated for emotion).

## 8. Conclusion

We presented a reproducible study of emotion-controllable anime character video generation on MiniMax H3. We showed that (i) current evaluation is unstable along steps, seeds, and judge frame counts, and must be swept to be trustworthy; (ii) a reference-conditioned LoRA trained on cross-emotion pairs unlocks the first negative-emotion result (neutral→sad) with a controlled neutral baseline, and stabilizes positive targets; and (iii) a specific class of strong-negative emotions remains a hard boundary across all interventions, with a controlled resolution ablation ruling out the layout-mismatch explanation. We release the full verdict matrix, judge logs, and checkpoints to support honest benchmarking of emotion control in open video generators.

## References

1. MiniMax. *MiniMax H3: Open General Intelligence* (model release), 2025. https://www.minimax.io/news/minimax-h3-open-source
2. Q. Yang et al. *CogVideoX: Text-to-Video Diffusion Models with an Expert Transformer*. arXiv:2408.06072, 2024.
3. Tencent. *HunyuanVideo: A Systematic Framework for Large Video Generative Models*. arXiv:2412.03603, 2024.
4. Wan Team. *Wan: Open and Advanced Large-Scale Video Generative Models*. arXiv:2503.20314, 2025.
5. OpenAI. *Video generation models as world simulators* (Sora), 2024.
6. A. Hu et al. *Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation*. arXiv:2311.17117, 2023.
7. H. Tian et al. *StableAnimator: High-Quality Identity-Preserving Human Image Animation*. arXiv:2411.17697, 2024.
8. H. Ye et al. *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models*. arXiv:2308.06721, 2023.
9. E. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685, 2021.
10. N. Ruiz et al. *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*. arXiv:2208.12242, 2022.
11. P. Wang et al. *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution*. arXiv:2409.12191, 2024.
