# Reproduction-first plan

**Date:** 2026-09-16  
**Goal:** build public, inspectable evidence before designing a new H3 training
method.

The benchmark/method lineage and reproducibility levels are maintained in
`REPRODUCTION_BLUEPRINT.md`; the machine-readable inventory is
`../reproduction_registry.jsonl`.

## Rules

- Reproduce before exploring.
- Prefer official code, weights, datasets and evaluation scripts.
- Candidate priority is:
  1. paper + official code + public weights;
  2. paper + code but no public weights;
  3. paper with missing code or non-reconstructable data.
- For priority-1 candidates, reproduce released-weight metrics before starting
  training. Training reproduction is admitted only after the checkpoint result
  has been compared against the paper's main claim.
- If released weights, code, evaluation data or reported settings cannot support
  a paper claim, record the mismatch as a result rather than silently replacing
  the protocol.
- For papers without public weights, reproduce training first only when the
  official code, exact public training data and evaluation protocol are
  available.
- Public datasets are mandatory for exploratory training.
- Private data is limited to visualization, diagnostics and smoke tests.
- GPU utilization is subordinate to a defined reproduction question.
- Every cached tensor must have an inspectable source manifest.

## Candidate classification

## Compute-first video expansion

Pure video generation and video post-training papers are in scope. The first
queue is restricted to paper training configurations using no more than eight
H100-class GPUs (older A100/V100/A6000 configurations count as within this
ceiling). A method is not promoted merely because checkpoint inference fits.

The primary research scope is general video generation:

- motion and temporal consistency;
- camera and object trajectory control;
- prompt following and visual quality;
- efficient generation, distillation and post-training;
- reusable control mechanisms that could later transfer to H3.

Talking-head generation, lip synchronization, speaker identity and digital
human rendering are not primary targets. They are retained only when a metric
or method is demonstrably reusable for general video generation.

### V1 — VADER: first low-cost video-training reproduction

- Paper, official code and released VideoCrafter2 LoRA weights are available.
- Artifact audit found that the released VideoCrafter2 LoRAs use ChatGPT animal
  and instrument prompts and are not identified as the checkpoints behind
  Table 1. They cannot directly verify the paper's core numeric table.
- Released-weight pass: first perform a deterministic checkpoint-effect audit
  for PickScore and HPSv2+Aesthetic on the released prompt domains. Label this
  as independent checkpoint evidence, not Table 1 reproduction.
- Released PickScore checkpoint result: on 360 paired VBench-standard temporal
  videos, PickScore improves by 0.6823 while Dynamic Degree falls from 0.6750
  to 0.5667. This is C2 checkpoint evidence, not Table 1.
- Training pass: the public 450-step command is HPSv2.1+Aesthetic on instrument
  prompts with rank 16. It does not match the released PickScore LoRA
  (animal prompts, rank 16) or HPS+Aesthetic LoRA (instrument prompts, rank 8).
  Smoke-test this disclosed example as a process reproduction, but do not call
  it a released-checkpoint reproduction. Exact training waits for the missing
  hyperparameters or an explicitly approximate protocol.
- Paper configuration: 2×A6000 for about 12 hours; official repository also
  documents 4×A6000 and a single-16GB-GPU mode.
- This is the first execution target because both checkpoint and exact training
  are affordable, and it directly tests a training method relevant to H3.
- Before claiming exact training reproduction, identify the Table 1 base model,
  checkpoint, sample count, seeds, checkpoint-selection rule and aggregation.

### V2 — T2V-Turbo: second checkpoint-to-training reproduction

- Paper, official training code and four released checkpoint variants exist.
- Base model/data: VideoCrafter2 or ModelScopeT2V with WebVid-10M.
- Released-weight pass is complete on the paper-day VBench evaluator: all 16
  dimensions over 4,720 generated videos, with local quality/semantic/total
  84.44/73.69/82.29 versus paper 82.57/74.76/81.01. This is C2 evidence:
  total is higher, but the dimension profile is not numerically recovered.
- Do not start the official 10k-step training run yet. The paper-used WebVid
  manifest is absent, official URLs/captions are no longer distributed, the
  required InternVideo2 Stage2 1B f4 reward checkpoint is gated, and the
  release has no dependency lock or complete launcher configuration.
- If those artifacts become available, first pass a one-batch
  decode/reward/backward/resume smoke, then reproduce the 8-process run.

### V3 — VideoDPO: train-first candidate, headline process artifact-gated

- The official OneDrive archive is acquired, SHA-256 verified and fully
  container-checked: 20,000/20,000 MP4 files decode successfully, with 20,000
  metadata rows and 10,000 preference pairs. Original media is retained beside
  the tar, manifests and audit outputs.
- The repository provides base VideoCrafter2/T2V-Turbo checkpoints, but does
  not provide a clearly identified final VideoDPO checkpoint matching the paper
  table.
- Do not run the checked-in command as “official 3k-step reproduction.” README
  and shell paths are broken, the config uses 10 epochs rather than 3,000
  steps, and `dupbeta: 0` disables OmniScore pair reweighting.
- The checked-in `pair.json` has no OmniScore/probability field, while the
  paper's headline method uses OmniScore pair reweighting. Its `metadata.json`
  also retains stale absolute author-machine paths. The released media supports
  a documented unweighted-DPO variant only; it cannot presently support the
  headline reweighted process claim.
- Seek the missing per-pair scores and exact evaluator identity. Do not invent
  reweighting or train the unweighted variant as VideoDPO paper reproduction.
- Training starts only after one-batch decode, pair ordering, loss, backward,
  checkpoint and resume smokes pass.
- Primary VideoCrafter2 targets: VBench total/quality/semantics
  81.93/83.07/77.38, HPS(V) 0.261 and PickScore 20.65.
- T2V-Turbo and CogVideo branches remain secondary until their training data
  and released implementation are complete.

### V4 — MotionCtrl: controllable-generation reproduction

- The paper checkpoint and example inference code are public, but the full
  paper evaluation and training protocols are not.
- The 407-camera/283-object manifests, 1,000-video FID/FVD reference list and
  CamMC/ObjMC/CLIPSIM/FID/FVD implementations are absent. Released examples
  cannot support a paper-metric claim.
- Public training code targets AnimateDiff, not the paper's VideoCrafter1
  lineage. Shipped scripts also overwrite 8 GPUs with one debug GPU, route
  sparse training to the dense config, and use 1M rather than 50k/20k/20k
  steps.
- Paper configuration: 8×V100, batch size 128; 50k CMCM steps followed by
  20k dense-trajectory and 20k sparse-trajectory OMCM steps.
- Next run only a deterministic released-example checkpoint smoke. Full
  training remains gated on raw media, exact sample manifests and a separately
  frozen public evaluation protocol.

### V5 — conditional candidates

- AnimateDiff: paper, code and weights are public. Full motion-module training
  compute is not sufficiently specified in the paper; do not schedule it until
  the official configuration is reconstructed. MotionLoRA alone is cheap
  (20–50 videos, 2k steps, about 1–2 hours) and may be reproduced separately,
  but must not be described as full AnimateDiff training reproduction.
- MM-Diffusion has cleared the public-data and compute audit: its paper uses
  8×V100 for base/SR training, and official Landscape media, weights and
  evaluators are retained. Its released-checkpoint 2,048-sample DDPM/DDIM
  metric lane is active; do not start base or SR training until that lane
  completes and is compared with the paper claim.

### Excluded from the first queue by compute or release state

- CameraCtrl: original paper reports 16×80GB A100, above the ceiling.
- OmniForcing: official minimum is 8×H200 141GB, not comparable to the
  available 8×H100 envelope.
- JavisDiT: paper training uses stages with up to 256 H100 GPU-days and the
  released checkpoint is a limited-budget preview rather than the paper's main
  model.
- Foundation-model pretraining such as H3, CogVideoX, Open-Sora and
  VideoCrafter2 is not considered training reproduction merely because its
  checkpoint can be fine-tuned on eight GPUs.

### P1 — paper, official code and public weights

#### P1.1 MM-Diffusion, CVPR 2023 — first full reproduction

Public assets:

- official training, inference and evaluation code;
- released Landscape and AIST++ base/SR checkpoints;
- Landscape and AIST++-crop download recipes;
- official I3D/AudioCLIP-based FVD, KVD and FAD evaluators.

Weight-first targets:

| dataset | protocol | FVD | KVD | FAD |
|---|---|---:|---:|---:|
| Landscape | paper default sampler | 229.08 | 13.26 | 9.39 |
| Landscape | paper 1000-step DDPM | 117.20 | 5.78 | 10.72 |
| AIST++ | paper default sampler | 176.55 | 31.92 | 12.90 |
| AIST++ | paper 1000-step DDPM | 75.71 | 11.52 | 10.69 |

The released checkpoint's intended row must be established from its config and
sampling script; these values must not be mixed with later papers' re-evaluation
numbers. Landscape DDPM-1000 + SR-DDIM25 passed a complete four-rank
16-sample media/evaluator smoke (16 original MP4 + 16 SR MP4 + 16 WAV, all
decode/hash-audited). A fresh 2,048-sample run 1 of 6 is active with pre-import
per-worker CUDA binding; it is not a metric result until complete media and a
separate evaluator audit close.

Training reproduction order:

1. reproduce one released Landscape checkpoint's generated samples and metrics;
2. reproduce the Landscape base model training configuration;
3. reproduce or resume the 64→256 super-resolution training;
4. repeat on AIST++ only after the first dataset closes.

The paper specifies 100k base-model steps and 270k SR steps. A reduced run is
only an engineering smoke test; it cannot be reported as training reproduction.

#### P1.2 OmniForcing — weight evaluation eligible, training resource-blocked

- Paper, training code and causal checkpoint are public.
- Reproduce the released checkpoint on JavisBench with FVD, FAD, CLIP, AV-IB,
  DeSync and runtime.
- Full training currently requires at least 8×H200 141GB and is therefore
  recorded as resource-blocked rather than approximated on a toy run.

### P1 with a claim-support risk

#### JavisDiT, ICLR 2026

- Official code, JavisBench/JavisBench-mini, evaluation code and preview weights
  are public.
- The released model is explicitly described as a limited-budget preview, while
  the paper reports a three-stage model trained on 788k/611k examples with
  64+8+256 H100 GPU-days.
- First evaluate the exact released preview checkpoint on JavisBench-mini using
  FVD, KVD, FAD, TV-IB, TA-IB, CLIP, CLAP, AV-IB, CAVP, AVHScore and JavisScore.
- Do not treat a preview-checkpoint discrepancy as a failed reproduction of the
  final paper model. Record it as “released weights do not instantiate the
  paper's main training configuration” unless the authors provide the matching
  checkpoint.
- Exact training is presently data-blocked because the 611k training videos are
  reconstructed from YouTube-derived MMTrail/TAVGBench sources rather than
  redistributed as a fixed public corpus, and is also compute-prohibitive.

### P2/P3 — missing code, weights or reconstructable data

- AV-DiT: paper states that code and pretrained models will be released, but no
  verified official release is currently linked; no metric or training
  reproduction starts.
- SyncFlow: keep in the missing-release audit until an official code/checkpoint
  package is verified.
- MiniMax-H3: official weights and inference examples exist, but there is no
  paper main table or official training release to reproduce. Treat it as model
  deployment plus independent public-benchmark evaluation, not paper
  reproduction.

## Track R0: MiniMax H3 official deployment validation

Deliverables:

- immutable copies of official T2VA, FL2VA and Ref2VA inputs and prompts;
- environment lock, model revision and inference command;
- generated MP4/WAV outputs;
- runtime, peak memory and basic media metadata;
- comparison notes against official reference outputs.

Success means all three official 768p cases run without local modifications to
the generation semantics. This is not a paper-metric or training reproduction.

**Negative provenance audit (2026-09-18):** the retained `official_requests/`
directories do contain three completed, hashable 768p API outputs (T2VA,
FL2VA and Ref2VA), with matching request IDs, polling history and media
metadata.  They do **not** close R0: all three final service responses identify
the model as `sora-2`, not MiniMax-H3, and they contain no H3 revision,
environment lock or local inference command.  Retain them as auditable service
API artifacts only; do not score, compare, or describe them as H3 generations.
R0 still requires a separately identified H3-serving or local-inference path.

## Track R1: H3 public-benchmark evidence

### R1a — Sci-VBench evaluator reproduction (completed on released media)

- Pinned Sci-VBench commit `e8cb3988631d0c22d31c02fc5b1cc3693bfb7031`.
- Re-evaluated 150 released MiniMax-H3 `testmini` videos; continuous VT
  components have median absolute discrepancies at or below `2.44e-05`, and
  Dynamic Degree matches exactly for 149/150 videos.
- This is C2 evaluation evidence for author-released media, **not** a local-H3
  generation or training reproduction.

### R1b — VABench (only after resource verification)

Deliverables:

- pinned VABench commit and dependency lock;
- public test-data manifest with source, license and checksums;
- H3 generation adapter that emits the exact filenames required by VABench;
- evaluator smoke on official example data;
- H3 result JSON with a coverage report for all 15 dimensions.

Start with a small official subset only to validate plumbing. A subset result
must be labelled as a smoke test, not as a benchmark score.

**Audit update (2026-09-18):** official code is pinned at
`12b9617a900a9eca1691da6b7115ce3d58f53a16`; its 521-image I2AV archive is
downloaded, hash-verified and retained with per-image hashes. The 38.38-GB
evaluator cache is only pointer-audited, not downloaded. A mapping-path defect
and unpinned evaluator dependencies keep this route at protocol-preparation,
not score reproduction.

The first component acquisition further found a concrete release blocker:
the two official DNSMOS cache weights hash-match their LFS pointers, but the
pinned evaluator imports an absent `third_party/DNS-Challenge/DNSMOS/dnsmos_local.py`.
Do not use a substitute implementation as an official VABench result.

The official I2AV JPEG archive and annotations have no published pairing key.
All 521 media hashes were independently rechecked, but equal per-category
counts and category names do not determine image-to-prompt identity; archive
and lexical order conflict with JSON order. Formal VABench I2AV scoring is
therefore blocked pending an author pairing manifest.

## Track R2: MM-Diffusion reproduction

Why first:

- official CVPR 2023 implementation;
- public Landscape and AIST++-crop download recipes;
- released Landscape/AIST++ base and super-resolution checkpoints;
- official FVD, KVD and FAD scripts;
- documented inference and training commands.

Milestones:

1. clone and pin the official repository;
2. audit dataset availability, license, file counts and preprocessing;
3. download one official checkpoint and evaluation models;
4. reproduce generated samples;
5. reproduce one dataset's FVD/KVD/FAD protocol;
6. compare against published and reproduced values;
7. only then estimate whether full or partial training reproduction is
   feasible on the available GPUs.

Current state: milestones 1--5 are complete twice for Landscape: runs 1/6 and
2/6 are terminal 2,048-sample released-checkpoint C3 results with retained
media, decode/hash audits and separate original/SR evaluations. Original 64×64
FVD/KVD/FAD×10k are `249.0004/15.4862/10.7107` and
`231.8951/14.3733/10.9198`; two runs do not estimate stable variance, reproduce
the paper's six-run aggregate, or authorize training. The official DPM-Solver default branch fails on a multimodal
dictionary in both inspected public revisions, so the scored lane is explicitly
the paper's `Ours*` DDPM-1000 row; any DPM-Solver fix is a separate labelled
compatibility lane.

Primary targets are the original paper rows listed in P1.1. Values reported by
later papers are retained only as secondary cross-checks because preprocessing,
sampling and evaluator versions may differ.

## Deferred targets

- DreamID-Omni: benchmark and training data are not currently established as
  public; literature-only.
- Talker-T2AV: code and weights may be reproducible, but the task is centered
  on talking heads, speech and lip synchronization; defer unless a general
  video-generation question specifically requires it.

## Documentation contract

Each reproduction gets:

- `REPRODUCTION_CARD.md`;
- `environment/`;
- `source_manifest.jsonl`;
- `commands.sh`;
- `reported_vs_reproduced.json`;
- `outputs/README.md`.

The card must distinguish:

- paper-reported;
- author-code reproduced;
- locally modified;
- incomplete or failed.

## Immediate order

The operational priority and all training stop conditions are maintained in
`REPRODUCTION_EXECUTION_MATRIX.md`. MM-Diffusion's two 2,048-sample checkpoint
runs are terminal C3 evidence, with original FVD/KVD not matching the paper
aggregate; preserve the individual values and only consider a separately
preregistered run 3, never a training decision. The fresh H3 ownership gate
now passes for GPUs 4--7, but a sequential frozen deployment case still requires
explicit human confirmation of ownership before launch.
T2V-Turbo, VADER, MotionCtrl and VideoDPO remain evidence/audit paths, not
authorized substitute-training paths, until their respective missing public
data, protocol or weight gates close.
