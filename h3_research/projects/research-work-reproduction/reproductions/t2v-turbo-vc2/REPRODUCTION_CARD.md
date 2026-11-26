# T2V-Turbo (VideoCrafter2) reproduction card

**Prepared:** 2026-09-16  
**Status:** released LoRA inference, 4,720-video generation and all 16
paper-day VBench dimensions complete; training reproduction not started  
**Paper:** *T2V-Turbo: Breaking the Quality Bottleneck of Video Consistency
Model with Mixed Reward Feedback*  
**Paper revision:** arXiv:2405.18750v2 / NeurIPS 2024

## Reproduction question

Does the released T2V-Turbo VC2 LoRA reproduce the paper's four-step VBench
Table 1 row under the official five-samples-per-prompt protocol, and can the
10K-step public training process be reconstructed from WebVid-10M within
8×H100?

The sequence is:

1. load the released LoRA and generate a deterministic one-prompt smoke;
2. freeze the paper-matching VBench generation protocol;
3. generate five videos per official prompt and run all 16 pinned dimensions;
4. compare every local dimension, Quality, Semantic and Total score with Table 1;
5. audit WebVid shards and reward checkpoints before any training reproduction.

## Paper-reported claims

The four-step T2V-Turbo (VC2) Table 1 row reports:

| aggregate | value |
|---|---:|
| Total Score | 81.01 |
| Quality Score | 82.57 |
| Semantic Score | 74.76 |

Quality dimensions:

| Subject | Background | Flicker | Smoothness | Aesthetic | Dynamic | Imaging |
|---:|---:|---:|---:|---:|---:|---:|
| 96.28 | 97.02 | 97.48 | 97.34 | 63.04 | 49.17 | 72.49 |

Semantic dimensions:

| Object | Multiple | Action | Color | Spatial | Scene | Appearance | Temporal style | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 93.96 | 54.65 | 95.20 | 89.90 | 38.67 | 55.58 | 24.42 | 25.51 | 28.16 |

The paper states that five videos were generated for every VBench prompt.

## Pinned public artifacts

Official code audit checkout:

`/gluster_hdd_cv/user/jinzili/code/research-work-reproduction/t2v-turbo`

- commit `eaae323f10d136796a33e8f5304ed50e40def570`;
- repository contains both T2V-Turbo v1 and v2; this card covers v1 VC2 only.

Paper-day VC2 inference revision:

- commit `10b1c53f1e444b88e9440e5e5da21bed20486f7c`, dated 2024-05-29;
- this predates the later v2 merge and is the primary weight-evaluation code;
- the later main branch changed fp16 to bf16, changed GUI defaults, explicitly
  passed 200 origin steps, and introduced a v1/v2 motion-condition branch.

Released VC2 LoRA:

`/gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/t2v-turbo/unet_lora.pt`

- Hugging Face revision `2c7a5a90188ed147b2a38a779c4331a69ba06b07`;
- 468,885,008 bytes;
- 1,150 fp32 tensors with 117,142,176 parameters;
- SHA256 `f1b223501f3fbbd491a9797c37f224ba031520cb442a10c07e4613b70b203845`.

The VideoCrafter2 base checkpoint is the same pinned artifact used by the VADER
reproduction.

## Protocol known from paper and release

- output: 512×320, 16 frames, FPS 16;
- release sampler: T2V-Turbo scheduler, four inference steps, 50 origin steps;
- demo guidance scale: 7.5;
- release dtype: fp16;
- VBench: official prompts, five samples per prompt, all 16 dimensions;
- training: WebVid-10M, 10K gradient steps, 6–8 A100 GPUs, no accumulation,
  per-GPU batch 1, 16 frames, learning rate 1e-5, guidance range [5, 15],
  DDIM skip 20, HPSv2.1 scale 1 and InternVideo2 Stage-2 scale 2.

The paper specifies four-step generation but does not state `lcm_origin_steps`.
The paper-day release uses the pipeline default of 50; the current main branch
overrides it to 200. The repository contains no VBench generation script or
seed list. Therefore 50 is frozen for the released-weight verification, while
the missing seeds remain a strict-reproduction limitation.

The current merged Gradio app is not used as the primary protocol because its
function defaults (4 steps, 16 FPS) conflict with its widgets (8 steps, 8 FPS)
and its v1 motion-condition branch did not exist in the paper-day release.

## VBench protocol audit

The paper-day VBench revision is
`03e7b948b0e8b9f87333c3f133b9f2a988a08f6e`. Its README requires five
randomly seeded, non-cherry-picked videos per prompt and filenames of the form
`$prompt-$index.mp4`, but it does not publish the seeds.

The paper-day and current prompt/full-info files are byte-identical. The
all-dimension file has 946 non-empty lines but only 944 unique prompts because
two strings are duplicated across dimensions. Deduplicating while preserving
order therefore yields 4,720 unique video files at five samples per prompt.

This reproduction freezes deterministic independent seeds 0–4,719 using
`prompt_index * 5 + sample_index`. All filenames, prompts and seeds were checked
for uniqueness before generation. This is a checkpoint verification under a
fixed public protocol, not an exact recovery of the authors' unavailable seeds.

## Environment compatibility finding

The paper-day app calls `pipeline.to(torch_device=..., torch_dtype=...)`.
Diffusers 0.30 uses `device` and `dtype`; the legacy names were silently ignored
in the first smoke, leaving xFormers attention on CPU. The failed run is kept at
`release-smoke-v1/run.log`. The local runner changes only those keyword names;
sampling parameters and weights remain unchanged.

The corrected device migration exposed a second release-code fp16 issue:
`timestep_embedding` and `fps_embedding` remain fp32 after the UNet is converted
to fp16. Official commit `80cb2ec` later added `.to(self.dtype)` at both sites.
That two-line official fix is isolated in
`patches/release-fp16-compat.patch`; the paper-day worktree remains clean and a
separate patched worktree is used for the next smoke.

That patch alone is insufficient because the old `UNetModel.dtype` is a plain
attribute initialized to fp32 and is not changed by `nn.Module.to(fp16)`. The
current official app explicitly assigns `unet.dtype`; the auditable runner now
does the same.

The next failure occurred in `GroupNormSpecific`, which forced fp16 activations
to fp32 while its converted parameters remained fp16. Official commit
`80cb2ec` also fixed this by removing the forced cast. That exact one-line
change is retained in `patches/release-groupnorm-fp16-compat.patch`.

The paper-day path then reached the old pipeline's forced-fp32 text context.
Rather than keep backporting later official fixes one by one, the successful
checkpoint verification uses clean current official revision `eaae323` with
the paper-day sampling parameters explicitly restored.

## Released-checkpoint smoke result

The released VC2 LoRA successfully generated:

`/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/t2v-turbo/outputs/current-code-release-protocol-smoke-v2/smoke_0000__seed_0.mp4`

- prompt `An astronaut riding a horse.`, seed 0;
- 4 steps, 50 origin steps, guidance 7.5, fp16;
- 512×320, 16 frames, 16 FPS, H.264, duration 1.0 s;
- complete ffmpeg decode passed;
- MP4 SHA256
  `9396aebacfb9b5683d09f940e8551a0d6fa1a50437e0f7c64ced1c4c48b7710d`;
- denoising runtime after model load: 0.903 s on one H100;
- raw MP4, run log, environment freeze and media manifest are retained.

This proves the released checkpoint is executable (C1), not that Table 1 is
reproduced. Full C2 evaluation still requires all VBench prompts under a fixed
independent seed protocol because the paper seeds are unavailable.

## Batch-generation preflight and active run

The four-shard preflight generated one official prompt on each of GPUs 4–7.
All four outputs are 512×320 H.264 videos with 16 frames at 16 FPS; complete
ffmpeg decoding passed, and each shard retained its raw video, generation
configuration, run log and per-video manifest.

The completed run is retained at:

`/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/t2v-turbo/outputs/vbench-standard-independent-seeds-v1`

It uses four shards of 1,180 videos. Every shard configuration directly records
the code revision, runner hash, base-checkpoint hash, LoRA hash, prompt-file
hash, generation parameters and seed rule.

The post-run validator matched all 4,720 expected filenames to 4,720 unique
manifest records and seeds 0–4,719. It recomputed every media SHA256 and fully
decoded every file. All videos are H.264, 512×320, 16 FPS and 16 frames; the
5,027,869,882 retained media bytes produced zero validation issues. The remote
validation report SHA256 is
`14e079e93f5d5f92c6de5afc770119d8c1ad0fc163b68b80115ae55928f2da78`.

Generation is therefore closed at C1. The released-checkpoint evaluation below
raises the checkpoint evidence to C2; it does not reproduce training.

## Complete paper-day VBench results

The first evaluator attempt pointed VBench at the mixed generation directory.
Paper-day VBench infers the media suffix from the first directory entry; it
selected `.json`, found zero videos and still wrote `NaN` result files. Those
failed outputs are retained. A second attempt used a videos-only hardlink view
of the same validated MP4 files.

| dimension | paper | fixed-seed result | difference |
|---|---:|---:|---:|
| Subject Consistency | 96.28 | 96.95 | +0.67 pp |
| Background Consistency | 97.02 | 97.88 | +0.86 pp |
| Temporal Flickering | 97.48 | 95.85 | −1.63 pp |
| Motion Smoothness | 97.34 | 96.07 | −1.27 pp |
| Aesthetic Quality | 63.04 | 66.61 | +3.57 pp |
| Dynamic Degree | 49.17 | 80.28 | +31.11 pp |
| Imaging Quality | 72.49 | 72.36 | −0.13 pp |
| Object Class | 93.96 | 95.00 | +1.04 pp |
| Multiple Objects | 54.65 | 47.64 | −7.01 pp |
| Human Action | 95.20 | 94.40 | −0.80 pp |
| Color | 89.90 | 87.34 | −2.56 pp |
| Spatial Relationship | 38.67 | 40.70 | +2.03 pp |
| Scene | 55.58 | 53.30 | −2.28 pp |
| Appearance Style | 24.42 | 24.44 | +0.02 pp |
| Temporal Style | 25.51 | 25.42 | −0.09 pp |
| Overall Consistency | 28.16 | 28.36 | +0.20 pp |

The later official aggregation script was introduced on 2024-06-28. Applied
to the paper's 16 dimension values it yields 82.5627 Quality, 74.7681 Semantic
and 81.0037 Total, exactly recovering the published rounded values. Applying
the same script here yields:

| aggregate | paper | fixed-seed result | difference |
|---|---:|---:|---:|
| Quality | 82.57 | 84.44 | +1.87 pp |
| Semantic | 74.76 | 73.69 | −1.07 pp |
| Total | 81.01 | 82.29 | +1.28 pp |

Aggregate agreement is misleading: Dynamic Degree is 31.11 points higher and
Multiple Objects is 7.01 points lower. The released checkpoint therefore does
not numerically recover the paper's dimension profile under the independent
fixed seeds, even though its recomputed Total is higher.

All dimensions have full expected coverage except Color. The paper-day Color
implementation drops a video whenever GRIT detects no target object, so only
386 of 425 expected videos contribute (90.82%). This makes Color a
detector-conditioned score with a changing denominator. The structured result
file records every result hash, coverage count and a 10,000-resample
prompt-cluster bootstrap interval.

This is C2 checkpoint evidence, not an exact contradiction or C4 process
reproduction: author seeds remain unavailable, and the raw WebVid training
shards, preprocessing and reward-model revisions are not yet reconstructed.

## Current gates

- Do not compare this row with VADER's 24-frame temporal subset: model,
  scheduler, frame count and prompt coverage differ.
- Do not start WebVid training. The official project no longer distributes
  URLs/captions, no paper-used sample or shard manifest exists, and no local
  WebVid media was found.
- The exact InternVideo2 Stage2 1B f4 reward checkpoint exists but is gated;
  current access returns HTTP 403. The README's f8 link is a different model.
- Use the clean 2024-06-24 training-release worktree, not the current v2 tree,
  if the data and reward gates are later cleared.
- Treat the unmodified released command as an eight-process protocol: ranks
  6–7 alone execute the video-reward branch.
- Retain every generated MP4 and prompt/seed/checkpoint manifest.
- Do not call a VBench result an exact Table 1 reproduction unless the authors'
  evaluation seeds and evaluator revision are recovered; otherwise report it
  as an independent fixed-protocol checkpoint verification.
