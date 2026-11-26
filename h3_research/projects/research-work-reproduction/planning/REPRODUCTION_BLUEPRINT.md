# Video-generation reproduction blueprint

**Version:** 0.1  
**Date:** 2026-09-16  
**Scope:** general video generation; public data; training configurations no
larger than 8×H100-class GPUs

## Purpose

最终 proposal 的文献数量与准入不由本蓝图或候选 BibTeX 数量决定。逐篇主源、相关性、
资源状态与段落映射必须通过
[`CITATION_ADMISSION_GATE.md`](../refs/CITATION_ADMISSION_GATE.md)；当前已批准最终引用为
`0/30`。

This blueprint is not a leaderboard copied from papers. It records where
published progress is genuinely comparable and where it is not.

For the compact current-state view linking each paper claim to its released
resources, local outcome and evidence limit, see
`../dashboard/PAPER_REPRODUCTION_TRACKER.md`. That dashboard is a derived
index; this document remains authoritative for evidence definitions and
protocol requirements.

For every paper, it links:

`public data → fixed protocol → paper claim → released checkpoint → local
checkpoint evaluation → local training reproduction`

The blueprint has two complementary views:

1. **benchmark lineage:** papers evaluated under an identical public protocol;
2. **method lineage:** training methods applied to the same public base model.

## Epistemic rule: evaluate claims, not papers

A paper is not assigned one global credibility label. Each quantitative or
directional statement is an independent claim.

The atomic claim key is:

```text
(paper, model/config, training data, evaluation dataset/split,
 generation protocol, evaluator revision, metric, claimed value or ordering)
```

For example, “method X improves VBench total over base Y” and “method X
improves temporal consistency” are two claims even when they appear in the same
table. One may be supported while the other is not.

Paper-reported numbers are **claims**, not observations trusted by default.
Possible fabrication, selective reporting, hidden retries, data leakage,
undocumented post-processing and evaluator manipulation are treated as risks
to be tested, not as accusations inferred from a discrepancy.

### Claim-evidence ladder

| state | evidence reached | interpretation |
|---|---|---|
| C0 — claimed | a number or ordering appears in the paper | unverified statement |
| C1 — artifacts traced | code, data references, checkpoint and evaluator identities are recorded and hashed | claim is inspectable, not yet supported |
| C2 — evaluation replayed | the official evaluator reproduces the authors' released outputs or score files | evaluation path is partly supported; model path is not |
| C3 — released checkpoint reproduced | independently generated outputs from the matching released checkpoint reproduce the metric under the exact protocol | checkpoint-level support |
| C4 — full process reproduced | public raw data → preprocessing → training → checkpoint → generation → evaluation → aggregation is reproduced | process-level support; the minimum state called “credible reproduction” |
| C5 — independently replicated | an independent implementation or independently reconstructed dataset also recovers the claim within uncertainty | stronger external support |
| X — contradicted | the exact available protocol produces a stable incompatible result and plausible implementation causes have been excluded | evidence against the claim |
| U — untestable | required artifacts are missing or cannot legally/practically be reconstructed | credibility unresolved |

Only C4 or C5 permits the phrase **“the paper claim was reproduced.”** C2 and
C3 must be described as evaluator-level or checkpoint-level support.

### Full-process path

```text
public raw media
  → immutable source manifest and hashes
  → documented filtering/splits
  → preprocessing and cached features
  → training batches and optimization trace
  → checkpoint
  → generation protocol and all attempts
  → raw outputs
  → evaluator inputs
  → per-sample scores
  → published aggregation
```

Every node and edge must be evidenced for C4. Reusing an author checkpoint
skips the training segment and therefore cannot exceed C3.

### Partial credibility is multidimensional

Each claim separately records:

- **data fidelity:** exact raw data, split, filtering and licenses;
- **training fidelity:** code revision, initialization, optimizer, schedule,
  batch construction, random seeds and compute;
- **generation fidelity:** checkpoint, prompts, sampling, retries and media
  post-processing;
- **evaluation fidelity:** evaluator code/weights, preprocessing and failures;
- **statistical fidelity:** sample count, repeated seeds, variance, confidence
  interval and aggregation;
- **numerical agreement:** absolute/relative discrepancy and whether the paper's
  ordering survives uncertainty.

A claim can therefore be checkpoint-supported but data-unresolved, or
evaluation-reproduced but statistically unsupported.

### Adversarial verification checks

For every headline claim, explicitly test:

- train/test contamination and benchmark leakage;
- cherry-picked prompts, seeds, checkpoints or generated attempts;
- hidden rejection/regeneration and omitted failures;
- mismatch between released and paper-used weights;
- metric code changes, evaluator-version drift and favorable preprocessing;
- aggregation changes, selective subcategories and missing denominators;
- manual editing, transcoding or post-processing before evaluation;
- reward/evaluator overfitting when the same model is optimized and judged;
- improvements smaller than seed/run variance;
- paper tables that cannot be derived from released per-sample records.

All generation attempts, including failures, must remain in the manifest.
Disagreement triggers diagnosis and repetition; it does not by itself prove
misconduct.

## Central method lineage

```text
VideoCrafter2 checkpoint
├── T2V-Turbo       consistency distillation + mixed reward
├── VADER           differentiable reward-gradient LoRA
├── VideoDPO        pairwise preference alignment
└── MotionCtrl      camera/object motion control modules
```

This lineage is useful because model-family differences are reduced. It allows
questions such as:

- Does a reported gain survive evaluation under one pinned VBench revision?
- Does better semantic alignment reduce motion or temporal quality?
- Is the gain reproduced from the released checkpoint?
- Does the official training run recover that gain?
- Which method still improves an H3-like flow/video model after controlling for
  data and evaluator changes?

VideoCrafter2 itself is a public base checkpoint, not a first-priority full
pretraining reproduction. Its role is to anchor affordable downstream training
comparisons.

## Public benchmark lanes

| lane | public protocol | what it measures | first papers/models to place | comparability rule |
|---|---|---|---|---|
| G: general quality | VBench | 16 decomposed quality/semantic dimensions | VideoCrafter2, T2V-Turbo, VideoDPO, H3 | same VBench commit, prompt suite, seeds, samples/prompt and evaluator weights |
| E: broad diagnostics | EvalCrafter | visual, content and motion quality through 17 objective metrics on 700 prompts | the same generated samples where supported | no comparison to a paper row unless its prompt and evaluator revisions match |
| C: compositionality | T2V-CompBench | 7 categories including binding, spatial relation, action, interaction and numeracy over 1,400 prompts | T2V-Turbo, VideoDPO, H3 | retain category scores; never collapse only to one mean |
| P: physical behavior | PhyGenBench/PhyGenEval | 160 prompts covering 27 physical laws | H3 and the strongest reproducible open baseline | pin all MLLM/evaluator versions and preserve raw judgments |
| T: long change | ChronoMagic-Bench-150 | persistence, variation and metamorphic/time-lapse behavior | open video baselines and H3 | treat this as a specialist lane, not a replacement for general quality |
| M: motion control | MotionCtrl protocol | camera motion, object trajectory, CLIP similarity, FID/FVD | released MotionCtrl then reproduced training | exact camera paths, trajectories, frame count and base checkpoint must match |

The initial primary lane is **G**. Lanes C, P and T are diagnostic extensions
used to locate a research gap; they are not mixed into one synthetic score.

## Reproducibility levels

| level | requirement | permitted claim |
|---|---|---|
| A | paper, official code, matching public weights, exact public evaluation data/protocol, reconstructable public training data, full training within budget | checkpoint metrics and full training reproduced |
| B | checkpoint evaluation is exact, but training data or preprocessing is only approximately reconstructable | checkpoint reproduced; training only approximate |
| C | official code and evaluation protocol exist, but the paper-matching checkpoint is absent | train-first reproduction; no released-weight claim |
| D | weights run, but exact paper protocol/checkpoint identity is missing | deployment or independent benchmark only |
| E | code, weights, data or evaluator is missing | literature claim only |

A failed reproduction does not lower the level by itself. Missing artifacts do.
Reported and local values remain visible side by side.

These A–E levels describe **artifact availability for a work**. They do not
replace the C0–C5/X/U evidence state assigned to every individual claim.

## Initial paper ledger

| work | main role | public weights | public training path | paper compute ≤8 H100-class | initial level | first verification |
|---|---|---:|---:|---:|---|---|
| VideoCrafter2 | common public base model | yes | incomplete for exact pretraining | not established | D | freeze checkpoint and baseline VBench |
| T2V-Turbo | fast consistency distillation and reward feedback | yes | code only; exact WebVid identity unavailable | yes, 8×A100 | B checkpoint / E exact training | C2 released-weight result complete; training artifact-gated |
| VADER | reward-gradient video post-training | yes, but released VC2 LoRAs do not match Table 1 | partial | yes, paper 2×A6000; repo 4×A6000 | D for Table 1; B for released LoRAs | released-LoRA effect audit; reconstruct Table 1 identity |
| VideoDPO | preference alignment | no clearly identified paper-final checkpoint | official 20k-MP4 archive fully checked; pair scores absent and config conflicts with paper | yes, 4×A100 | C unweighted release / E reweighted process | seek OmniScore weights; do not substitute unweighted DPO for the paper |
| MotionCtrl | camera/object motion control | yes | no paper-complete path | yes, 8×V100 | D examples / E paper metrics+training | C2 one-condition deployment smoke complete; paper protocol unavailable |
| AnimateDiff | motion module and MotionLoRA | yes | partial | full-run compute not sufficiently pinned | B/D | audit full training configuration; keep MotionLoRA separate |
| MM-Diffusion | joint audio-video baseline | yes | no exact C4 path: released Landscape archive is retained, but upstream YouTube sources/licenses and exact reconstruction manifest are absent | yes, 8×V100 | B artifacts / C3 terminal checkpoint metrics: two clean-ish runs plus one affected observation | original 64×64 FVD/KVD miss the paper aggregate in all three local observations; run 3 has post-launch GPU-6 interference; preserve evidence and do not train |
| MiniMax-H3 | target foundation model | yes | no public training reproduction | inference verified on 8×H100; training not established | C2 author-media evaluator replay + official frozen deployment cases; local 150-case checkpoint lane interrupted / E training | Sci-VBench author-media evaluator replayed; local released-Base run reached 128/150 before the identical 8-GPU service vanished. Restore TP=2/Ulysses=4/Ring=1/no-LoRA without external-GPU contention, then audit all media and run six-VT; the earlier 4×80GB OOM remains configuration-specific |
| JavisDiT++ / JavisDiT-v1.0 | released joint audio-video checkpoint and public benchmark evaluator | yes | partial only; core Stage-2 YouTube media not redistributed | mini inference demonstrated; full training not established and data-blocked | C2 checkpoint generation and four-sample evaluator-interface smoke; 1,000-prompt mini run not started / training E | weights/dependencies and four official CSV rows are hash/audit closed; a separately preregistered 1,000-prompt mini run is required before any mini-level observation, and mini remains distinct from the full paper table |

## First comparable metric chain

### MM-Diffusion public joint-audio-video lane

Landscape is the first complete public-media path: the official archive has
900 train and 100 retained test MP4s, all decode-audited; base/SR/I3D/AudioCLIP
artifacts are content-hashed. A four-rank released-checkpoint smoke has 16/16
original 64×64 MP4, SR 256×256 MP4 and WAV outputs with full decode/hash audit.
Two clean-ish and one affected 2,048-sample DDPM-1000/SR-DDIM25 checkpoint
metric runs are terminal.
Run 1
with each rank's GPU bound before Python imports PyTorch. It retained 2,048
original MP4, 2,048 SR MP4 and 2,048 WAV files; all passed `ffprobe`, with no
byte-identical media duplicates. Standalone evaluation with explicit fake media
directories gives original 64×64 FVD/KVD/FAD `249.0004/15.4862/10.7107` versus
the paper's six-run `117.20/5.78/10.72`; independently scored SR is
`149.9356/7.6905/10.1019`. This is C3 evidence from **run 1/6**, not a table
reproduction. The public DPM-Solver default path
crashes on multimodal dictionary input in both the paper-era and current code,
so that paper row remains separately blocked pending an explicitly labelled
compatibility lane. No MM-Diffusion training starts before preregistered repeat
runs distinguish variance from a stable checkpoint/protocol mismatch. Run 2 used the same
public artifacts and official unseeded sampler behavior (physical GPUs 4/5/6/7 only);
it independently retained and audited 2,048 original MP4, 2,048 SR MP4 and 2,048 WAV,
then scored original `231.8951/14.3733/10.9198` and SR diagnostic
`154.7460/7.8836/10.5998`. Across the two original runs, the descriptive mean is
`240.4477/14.9298/10.8152` and the range is `17.1053/1.1130/0.2091`. These two
observations are C3 checkpoint evidence only: neither a six-run estimate nor a paper
contradiction, and they do not authorize training. Run 3 used the same public
artifacts and official unseeded sampling path on GPUs 4/5/6/7, retained exactly
2,048 original MP4, 2,048 SR MP4 and 2,048 WAV, and passed the full audit with
zero decode failures and byte-identical duplicates. Its independently evaluated
original value was `222.6008/13.1663/10.8026`; the separately evaluated SR
diagnostic was `137.6547/7.0204/10.9443`. However, GPU 6 acquired a separate
external allocation of about 37.7 GiB after the run started. No external
process was changed, but run 3 is consequently a terminal **affected C3
observation**, not a clean repeat: it must not be pooled with runs 1--2 for a
variance estimate, used to reject the paper's six-run aggregate, or used to
justify training.

The first ledger uses VideoCrafter2-related models under one pinned VBench
protocol.

| model/run | source | checkpoint status | total | quality | semantics | HPS(V) | PickScore | local status |
|---|---|---|---:|---:|---:|---:|---:|---|
| VideoCrafter2 baseline | VideoDPO paper table | public base checkpoint | 80.44 | 82.20 | 73.42 | 0.258 | 20.65 | pending exact evaluator reproduction |
| VideoDPO full | VideoDPO paper table | paper-final weight not clearly released | 81.93 | 83.07 | 77.38 | 0.261 | 20.65 | C1 audit: gated by archive provenance and release/config conflicts |
| T2V-Turbo VC2 (paper) | original paper Table 1 | released checkpoint | 81.01 | 82.57 | 74.76 | — | — | reported claim |
| T2V-Turbo VC2 (local) | paper-day VBench revision, 4,720 videos | released checkpoint | 82.29 | 84.44 | 73.69 | — | — | C2: all 16 dimensions complete; profile mismatch |
| VADER VC2 | local paired VBench temporal subset | released PickScore LoRA | separate partial lane | Dynamic Degree 0.6750→0.5667; Motion Smoothness 0.96556→0.97166 | separate partial lane | — | 21.2102→21.8925 | checkpoint effect reproduced; training command mismatch unresolved |
| MiniMax-H3 (Sci-VBench) | author-released H3 testmini videos | author media, not locally generated | VT 0.73460 released | — | — | — | — | C2 evaluator replay: 150/150, VT median absolute delta `6.15e-06`; seed-stability evidence separate |
| MiniMax-H3 (local released Base) | author-settings-matched Sci-VBench testmini, 150 prompts | released | — | — | — | — | — | three frozen official cases passed; the formal local lane is incomplete at 128/150 after its 8-GPU service vanished. No local VT has been calculated; four-GPU OOM is retained as a separate historical configuration limit |

Paper-reported numbers are provenance, not yet verified facts. The ledger gains
a `local` row only after the full protocol key below is frozen.

## Protocol identity key

Two values may be compared as one metric lineage only when these fields match:

- benchmark name and commit;
- prompt-list hash and excluded prompts;
- checkpoint hash and model configuration;
- resolution, aspect ratio, frame count, FPS and duration;
- sampler, inference steps, guidance and negative prompt;
- seed list and samples per prompt;
- evaluator code commit and evaluator-weight hashes;
- decoding and media preprocessing;
- aggregation formula and treatment of failed generations.

Any mismatch creates a new protocol column instead of silently extending the
same curve.

## Evidence files for each reproduction

```text
reproductions/<paper-id>/
├── REPRODUCTION_CARD.md
├── source_manifest.jsonl
├── protocol.json
├── reported_vs_reproduced.json
├── environment/
├── commands.sh
└── outputs/README.md
```

`reported_vs_reproduced.json` records:

- atomic `claim_id` and full claim key;
- exact table/figure source;
- current C0–C5/X/U evidence state;
- released checkpoint result;
- reproduced-training checkpoint result;
- absolute and relative discrepancy;
- repeated-run uncertainty and decision tolerance;
- whether the paper claim is supported, unsupported or not testable;
- data/training/generation/evaluation/statistical fidelity fields;
- every protocol deviation.

## Execution waves

### Wave 1 — shared evaluator and base checkpoint

1. Pin VBench and evaluator weights.
2. Freeze the VideoCrafter2 checkpoint and generation protocol.
3. Generate one shared baseline sample set.
4. Run VBench end to end and retain per-video/per-dimension outputs.

### Wave 2 — affordable post-training lineage

1. VADER released LoRA → official reward metrics → pinned VBench. **Completed
   for the released PickScore LoRA on 360 paired videos:** reward improves while
   Dynamic Degree falls.
2. VADER training → same evaluations. Exact released-weight reproduction is
   blocked because the repository's 450-step example matches neither released
   LoRA; first reproduce and label the disclosed example process separately.
3. T2V-Turbo released checkpoint → VBench. **Completed:** all 16 dimensions,
   local quality/semantic/total 84.44/73.69/82.29 versus
   82.57/74.76/81.01 reported.
4. T2V-Turbo official 8×A100 training → same evaluations. **Artifact-gated:**
   exact WebVid media/manifest unavailable, required InternVideo2 f4 reward
   checkpoint gated, and the release environment/launcher is incomplete.
5. VideoDPO official archive audit is complete: 20,000/20,000 MP4 decode and
   10,000 pairs are preserved. **Headline training remains gated:** `pair.json`
   lacks the OmniScore weights needed for reweighting; checked-in paths are
   broken, `max_epochs: 10` conflicts with 3,000 steps, and `dupbeta: 0`
   disables the headline method. Do not use a reconstructed unweighted run as
   paper reproduction.

### Wave 3 — control and failure-oriented benchmarks

1. MotionCtrl released weights and data-reconstruction audit. **Protocol audit
   completed:** checkpoint and demo conditions exist, but the paper's 407/283
   evaluation manifests, metric code and VideoCrafter1 training program do not.
   A deterministic camera-L example smoke now passes with zero checkpoint-key
   mismatch and a fully decodable video. Next is an independently frozen public
   control protocol, not a paper-metric claim.
2. T2V-CompBench on the strongest two reproducible models plus H3.
3. PhyGenBench and ChronoMagic-Bench only after evaluator revisions are pinned.

### Wave 4 — H3 transfer question

Use the reproduced lineage to choose one falsifiable H3 training question.
Candidates must arise from a stable public deficit, not from private examples.

## Stopping rules

- A released-checkpoint failure does not automatically stop training. First
  determine whether the released weight is the paper-used checkpoint. If exact
  public data/code and budget remain available, training reproduction may be
  the necessary experiment that distinguishes a release mismatch from an
  unsupported process claim.
- Stop an expensive training path only when the missing data/code prevents C4,
  the resource ceiling is exceeded, or the resulting experiment cannot resolve
  the claim's evidence state.
- Stop calling a result comparable when any protocol-key field differs.
- Do not substitute a private judge for a missing official evaluator.
- Do not complete an expensive run until a short run reproduces data loading,
  loss scale, checkpointing and evaluation plumbing.
- Do not turn a smoke run into a paper result.
- Do not upgrade a claim from C3 to C4 merely because author weights match the
  paper table.
- Do not accept a claimed improvement whose margin is not distinguishable from
  repeated-seed or repeated-training variance.
