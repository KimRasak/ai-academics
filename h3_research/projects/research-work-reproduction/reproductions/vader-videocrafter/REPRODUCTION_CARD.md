# VADER / VideoCrafter reproduction card

**Prepared:** 2026-09-16  
**Status:** artifacts/environment pinned; released PickScore LoRA audit and
full paired VBench temporal subset evaluation completed. This is C2 checkpoint
effect evidence only; VADER Table 1 and training reproduction remain unresolved.  
**Paper:** *Video Diffusion Alignment via Reward Gradients*  
**Primary claim source:** arXiv:2407.08737v1, Table 1 and Figure 7

## Reproduction question

Does the public VADER release support the paper's claim that reward-gradient
fine-tuning improves train and unseen-test reward more efficiently than DPO and
DDPO?

This question is split into:

1. **released-checkpoint audit:** independently generate with the two released
   VideoCrafter2 LoRAs and verify that they change their named rewards;
2. **Table 1 process reproduction:** reconstruct the paper's Base/DDPO/DPO/VADER
   train/test protocol and reproduce its numeric rows;
3. **training reproduction:** rerun the matching VADER training from the public
   base checkpoint and prompt sets.

Only item 3 can raise the Table 1 claims to C4.

## Paper-reported claims

Table 1 reports:

| task | method | train | test |
|---|---|---:|---:|
| Aesthetic T2V | Base | 4.61 | 4.49 |
| Aesthetic T2V | VADER | 7.31 | 7.12 |
| HPS T2V | Base | 0.25 | 0.24 |
| HPS T2V | VADER | 0.33 | 0.32 |
| Action prediction T2V | Base | 0.14 | not reported |
| Action prediction T2V | VADER | 0.79 | not reported |
| Aesthetic I2V | Base | 4.91 | 4.96 |
| Aesthetic I2V | VADER | 7.83 | 7.64 |

The paper states that experiments used 2×A6000 48GB and averaged about 12
hours. Figure 7 claims VADER begins converging within at most 12 GPU-hours and
is more query/compute efficient than DPO and DDPO.

The paper numbers are claims rather than verified results. Artifact tracing has
raised the four ledger entries to C1, but none has reached evaluator-,
checkpoint- or process-level support.

## Pinned public artifacts

Remote code:

`/gluster_hdd_cv/user/jinzili/code/research-work-reproduction/VADER`

- paper-release tag `v1.0`:
  `d2da3433c58473c69525c17b0182e66bc4580ca0`;
- audited current commit:
  `c63a39ee370a7374814cd069989a144863fb6095`;
- the current commit fixes boolean argument parsing and Accelerate device
  placement and adds DPO/DDPO implementations after the paper release.

Remote weights:

`/gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/vader`

- VideoCrafter2 base checkpoint, 7,404,653,244 bytes;
- PickScore LoRA, 20,220,491 bytes;
- HPSv2.1+Aesthetic LoRA, 10,189,387 bytes.

Hashes and immutable source revisions are in `source_manifest.jsonl`.

## Artifact audit findings

The released VideoCrafter2 LoRAs do **not** instantiate Table 1:

- PickScore LoRA: trained on `chatgpt_custom_animal.txt`, rank 16;
- HPSv2.1+Aesthetic LoRA: trained on
  `chatgpt_custom_instruments.txt`, rank 8;
- Table 1 describes HPSv2 Action prompts for training and Activity prompts for
  testing; it does not identify these released LoRAs as the evaluated weights.

The repository contains plausible Table 1 prompt ingredients:

- `hps_custom.txt`: 50 action-containing prompts;
- `simple_animals.txt`: 45 animals;
- `activities.txt`: three activities, producing 135 combinations.

However:

- Table 1 does not explicitly identify the base model/configuration used for
  every column;
- exact seeds, number of samples per prompt, checkpoint-selection rule and
  aggregation code are not specified;
- prompt helper functions still reference several files absent from all current
  release tags;
- three action/evaluation files existed briefly in repository history and were
  deleted before the public release tag;
- the current VideoCrafter training script uses 4×A6000 in the README, 450
  steps, rank 16 and ChatGPT instrument prompts, whereas the released
  HPS+Aesthetic LoRA is rank 8.

Therefore:

- released weights can support a checkpoint-effect audit, not Table 1;
- Table 1 is currently C1 (artifacts traced) with unresolved protocol identity;
- no claim is yet C2, C3 or C4.

## Released PickScore LoRA smoke audit

On 2026-09-16, the repository's inference path was run on one H100 using the
official VideoCrafter2 base and released rank-16 PickScore LoRA. Seed 200
deterministically selects line 1 of the three-record
`chatgpt_inference.txt` file.

The same prompt, seed, latent-generation settings and 24-frame output protocol
were used for a base control and the released LoRA:

| output | SHA256 | PickScore mean over 24 decoded frames |
|---|---|---:|
| base | `7ef3d8d94f265702efb698125e4e9dedac0a8c04116f6b47423e21e9a65da540` | 20.9564 |
| PickScore LoRA | `39064b7422b64a4c2634c3a89c8d7a1f25f12a07a2cdb16e750d46a3e7d09315` | 21.8594 |

Observed single-sample delta: **+0.9030**. Both files are 512×320, 24 frames,
10 fps and 2.4 seconds. Contact-sheet inspection shows a clear appearance and
composition change, but little motion in either sample.

This is a non-paper, single-seed checkpoint-effect smoke result. It is recorded
as C2 supporting evidence for the released artifact only; it does not support
any Table 1 row and is not a reproducibility claim.

The pilot was then expanded without modifying upstream code. Seeds 3–11 select
the three official inference prompts exactly three times each, yielding nine
paired base/LoRA videos:

| metric | base mean | LoRA mean | paired delta | positive pairs |
|---|---:|---:|---:|---:|
| PickScore | 21.8806 | 23.1838 | +1.3032 | 9/9 |
| HPSv2.1 | 0.32191 | 0.33046 | +0.00854 | 8/9 |
| Aesthetic | 5.4036 | 5.7361 | +0.3325 | 8/9 |

The 95% paired t intervals are `[0.8833, 1.7231]` for PickScore,
`[0.00068, 0.01641]` for HPSv2.1 and `[0.1539, 0.5110]` for Aesthetic.
Thus the released PickScore LoRA shows a stable effect in this small protocol
and does not exhibit an immediate trade-off on the two cross-frame metrics.
The same nine pairs were therefore evaluated with the official VBench custom
input implementations:

| VBench diagnostic | base | LoRA | paired change |
|---|---:|---:|---:|
| Dynamic Degree | 2/9 dynamic | 1/9 dynamic | one true→false pair |
| Motion Smoothness | 0.97722 | 0.97972 | +0.00250 |

The Motion Smoothness 95% paired t interval is
`[-0.00020, 0.00520]`; it does not establish a change. Dynamic Degree has only
one discordant pair. The supported conclusion is that both conditions are
mostly static on these three demo prompts, not that the LoRA caused a temporal
degradation.

The checkpoint audit remains C2 until a paper-matching protocol is identified;
Table 1 remains C1.

On 2026-09-16, a manifest-driven wrapper was smoke-tested for the official
VBench prompt protocol. The base and PickScore LoRA runs used the same prompt
(`a person swimming in ocean`) and seed 30000; both produced 512×320, 24-frame,
10-fps videos and distinct media hashes. After this gate passed, the full 72
prompt intersection of Dynamic Degree and Motion Smoothness was launched with
five samples per prompt and paired seeds: 360 base and 360 LoRA videos across
four H100s. All four shards completed in about 32.4–32.6 minutes. An independent
validator checked all 720 hashes and media specifications, all 360 prompt/seed
pairs, the standard filenames and hard-link provenance with no errors. Official
`vbench_standard` evaluation is being run separately for the base and LoRA
conditions. This remains a public-benchmark checkpoint audit, not a
reproduction of VADER Table 1.

The completed standard-mode temporal evaluation gives:

| metric | base | PickScore LoRA | paired change |
|---|---:|---:|---:|
| Dynamic Degree | 0.6750 (243/360) | 0.5667 (204/360) | -0.1083 |
| Motion Smoothness | 0.96556 | 0.97166 | +0.00610 |

Dynamic Degree has 54 dynamic→static and 15 static→dynamic pairs
(exact McNemar `p=2.61e-6`). Its prompt-level paired 95% interval is
`[-0.1582, -0.0585]`. Motion Smoothness has a prompt-level paired interval of
`[0.00307, 0.00913]`. The supported interpretation is a measurable temporal
trade-off: the released reward-aligned checkpoint is smoother but substantially
less dynamic. Smoothness must not be presented as motion preservation, because
the dynamic→static subset itself has a positive mean smoothness change.

The same 360 pairs were scored on all 24 frames with the pinned PickScore
processor and checkpoint:

| metric | base | PickScore LoRA | paired change |
|---|---:|---:|---:|
| PickScore | 21.2102 | 21.8925 | +0.6823 |

The prompt-level paired 95% interval is `[0.5383, 0.8263]`; 294/360 video pairs
improve. Therefore the reward effect and motion loss coexist under one fixed,
public and paired protocol. This is stronger evidence than the earlier
three-prompt pilot, but it remains C2 checkpoint evidence rather than a VADER
training or Table 1 reproduction.

Two failed evaluator attempts are retained:

1. decoding with Decord in the same CUDA process exited 139;
2. FFmpeg-decoded frames initially failed because fp32 processor output was
   passed to fp16 evaluator weights.

The successful replay uses FFmpeg-decoded PNG frames, the official PickScore
processor/checkpoint revisions in `source_manifest.jsonl`, fp16 model weights,
explicit fp16 `pixel_values`, and batches of four frames.

Additional retained failures:

- `from_str` cannot be used through the official inference entry point because
  it accepts one positional argument while the caller always passes two;
- importing the training script as a module initially failed because it assumes
  `scripts/main` is on `PYTHONPATH`.
- a four-H100, one-step smoke of the disclosed 450-step training example
  continued to seven optimizer steps until manually terminated. The official
  loop checks `max_train_steps` only inside its 100-iteration inner loop, then
  enters the next epoch; it also multiplies checkpoint/validation intervals by
  gradient accumulation even though `global_step` already counts optimizer
  steps, and it does not create `project_dir` when W&B is disabled. The failure
  and termination log are retained. A minimal patch is stored under `patches/`
  and must be applied only in a separate worktree.

The patched-worktree replay completed exactly one optimizer step in 88.31
seconds, exited normally and saved a 396-tensor, 5,015,552-parameter LoRA
checkpoint. This establishes that the disclosed HPS+Aesthetic example can
execute through prompt sampling, differentiable generation, reward evaluation,
backpropagation and checkpoint saving after the three loop/output fixes.

On 2026-09-18, an additional single-H100 one-step attempt reached model
loading, LoRA injection and HPSv2.1/Aesthetic weight loading, then failed before
sampling: the training branch unconditionally accesses `peft_model.module`, but
Accelerate does not wrap a single-process model in DDP. The failed log is
`/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vader/process-smoke-20260918/stdout.log`.
This is an environment/launcher compatibility finding only. The prior four-H100
patched smoke remains the authoritative successful process evidence; do not
silently treat the single-GPU adaptation as an author-protocol reproduction.

The 450-step example is not being expanded into a full run: no paper metric or
released checkpoint is identified with that rank-16 HPS+Aesthetic configuration,
so an approximately 11-hour run would have no claim-level endpoint. Exact
PickScore training reproduction remains blocked by undisclosed hyperparameters.

## Execution gates

Before expanding checkpoint inference:

1. move from repository demo prompts to a fixed public VBench prompt subset;
2. use the full VBench sample count where a benchmark number is claimed;
3. do not begin training until the released checkpoint's relation to Table 1 is
   resolved or the training run is explicitly labelled approximate.

Before training:

1. identify which base model produced Table 1;
2. reconstruct the exact train/test protocol or mark it as approximate;
3. reproduce the base checkpoint reward first;
4. define repeated seeds and confidence intervals;
5. do not call the repository's 450-step example an exact paper reproduction
   unless its relation to Table 1 is established.
