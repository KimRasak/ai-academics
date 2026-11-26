# VideoDPO (VideoCrafter2) reproduction card

## Question

Can the public VideoDPO release reproduce the CVPR 2025 VideoCrafter2
preference-alignment result from raw, inspectable media within 4×A100-class
compute?

## Paper claim

The paper reports VideoDPO training on 10,000 VidProM prompts, with four
VideoCrafter2 samples per prompt. It selects preference pairs with OmniScore,
trains for 3,000 steps with global batch size 8 on 4×A100 GPUs, and evaluates
with VBench, HPS(V) and PickScore.

| model | VBench total | quality | semantics | HPS(V) | PickScore |
|---|---:|---:|---:|---:|---:|
| VideoCrafter2 | 80.44 | 82.20 | 73.42 | 0.258 | 20.65 |
| VideoDPO | 81.93 | 83.07 | 77.38 | 0.261 | 20.65 |

These remain paper claims at C1; no matching final checkpoint is released.

## Public artifacts

- Official code revision:
  `1febdb42db147824ba4c96eb5e787bbd1295afc7`.
- Official dataset acquired at
  `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/videodpo/official/vidpro10k-vc2-dataset.tar`:
  12,124,170,240 bytes, SHA-256
  `5379c1a25d2f00c04c2d75a89b0d5fcf26f62417ae4788cb323c0cfe19a9266b`.
- VideoCrafter2 base checkpoint and a script that derives the frozen reference
  UNet from it.
- VBench prompt file with 945 lines, not a complete pinned evaluator
  environment.
- No clearly identified paper-final VideoDPO checkpoint.

A third-party Hugging Face derivative,
`CodeGoat24/VideoDPO@39a557ee8f69dbc7da98d83c49c32427d91967d3`,
contains approximately 32 GB of winner/loser frame archives and 22 MB metadata.
It was repackaged for UnifiedReward, is not the official VideoDPO training
release, and does not establish original-video provenance or the official
`metadata.json`/`pair.json` schema.

## Code-to-paper audit

The checked-in release cannot be run as the paper protocol without explicit
reconstruction:

- README says `bash configs/vc_dpo/run.sh`, but the repository directory is
  `configs/vc2_dpo/`.
- `configs/vc2_dpo/run.sh` points to nonexistent
  `configs/dpo/config.yaml`, not the checked-in VC2 configuration.
- It passes `--auto_resume True` although `--auto_resume` is a flag.
- The configuration uses `max_epochs: 10` and no `max_steps: 3000`.
- With 10,000 pairs, four ranks, per-rank batch 1 and accumulation 2, ten
  epochs would nominally produce about 12,500 optimizer steps, not 3,000.
- The headline OmniScore pair reweighting is disabled by `dupbeta: 0`; the
  loader then emits a constant factor of 1.0.
- Dependencies leave Torch, torchvision, xFormers, diffusers and CUDA
  unresolved; `tenosrboard` is misspelled.
- The repository has no complete pinned VBench/HPS(V)/PickScore evaluator
  package or paper generation seeds.

The official tar contains 20,000 MP4 files, 20,000 metadata rows and 10,000
unique pairs, with no `.pt`/latent files. Every pair is `(2i, 2i+1)`, has
`label=0`, and shares one caption. However, `pair.json` contains no `score` or
OmniScore probability field at all. It therefore cannot instantiate the
paper's advertised pair reweighting. Metadata also stores stale absolute paths
from the authors' machine, so the released loader cannot read a normal
extraction without a documented path rewrite.

The extracted original media was fully probed with `ffprobe`: **20,000 passed,
0 failed**. The raw tar, extracted MP4 files, `metadata.json`, `pair.json`,
archive SHA-256, member inventory, metadata audit and media-probe outputs are
retained at `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/videodpo/official/`.

## Evidence decision

- Paper numeric claims: **C1 / artifacts traced, process unsupported**.
- Matching released checkpoint: **absent**.
- Official media/pair archive: **acquired and structurally audited, but missing
  the reweighting scores required by the headline method**.
- Checked-in training command: **non-executable as written**.
- Checked-in VC2 configuration: **does not match 3,000-step reweighted
  VideoDPO**.
- Exact evaluation protocol: **incomplete**.

Do not launch a full training run yet.

## Next action

1. Recover the unreleased per-pair OmniScore/probability values.
2. Recover the exact VBench/HPS(V)/PickScore revisions, generation seeds and
   aggregation protocol.
3. Only if those artifacts arrive, freeze a separate 3,000-step,
   global-batch-8 reweighted config and pass decode/pair/loss/backward/
   checkpoint/resume smokes.
4. Then run 4-GPU training and pinned VBench/HPS(V)/PickScore.

If the official archive lacks original videos or source provenance, the run may
test the released training package, but cannot reach C4 full-process
reproduction.
