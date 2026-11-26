# MM-Diffusion reproduction card

**Status:** paper/code/config、Landscape source media 与 released checkpoints
均已审计；三次 2,048-sample checkpoint metric runs 已终态。run 1/2 为干净重复，
run 3 完整但受 GPU-6 后发外部占用影响；均非论文六次聚合或训练复现。  
**Authority:** CVPR 2023 paper, official `researchmm/MM-Diffusion` repository,
and the Google Drive folders linked by its README.  
**Scope:** first reproduce released-checkpoint inference and the pinned
objective evaluation protocol on Landscape; only then attempt the published
training process. This is a public joint audio-video baseline relevant to H3
comparison, not an H3 training method.

## Immutable sources

- Paper: arXiv `2212.09478`, retrieved from `https://arxiv.org/pdf/2212.09478`;
  SHA-256 `39c9d8378d90700ecca0240d19d766551b920aaa2dea91d5a4af592bc39a9b3c`.
- Official code, latest reachable commit:
  `7250222114ee1eca36e32d90e8a41f12d92651e2` (2024-06-05).
- Paper-era release baseline used for metric comparison:
  `c895b2993c471dd93fe4411f46af6c46c6eed4af` (2023-04-23). It is the earliest
  reachable release containing the complete evaluator and disclosed shell
  commands; it is not proven identical to the authors' private paper runtime.
- The official repository is MIT licensed.

## Paper claims to test

The paper evaluates 2,048 generated samples per model, calculates all visual
metrics at 64×64, and averages six runs in the main objective comparisons.
Its Table 1 / Table 2 claims for the released joint model are:

| dataset | sampler | FVD ↓ | KVD ↓ | FAD ↓ |
|---|---|---:|---:|---:|
| Landscape | default DPM-Solver | 229.08 | 13.26 | 9.39 |
| Landscape | full 1,000-step DDPM (`Ours*`) | 117.20 | 5.78 | 10.72 |
| AIST++-crop | default DPM-Solver | 176.55 | 31.92 | 12.90 |
| AIST++-crop | full 1,000-step DDPM (`Ours*`) | 75.71 | 11.52 | 10.69 |

The paper defines FAD as the AudioCLIP Fréchet distance multiplied by `1e4`.
The official repository's 2023 release code used `×1000`; commit
`3766a2fbfd97a7d133f048a99151fd5568f247bc` (2024-06-04) corrects it to
`×10000`, and current `7250222` retains that correction. FVD/KVD and FAD must
therefore be recorded with their evaluator revision and scaling factor.

## Published training envelope

The supplementary implementation table and official shell scripts agree on an
8×V100 process, within the available 8×H100 ceiling but not yet authorized to
run until exact public data identity closes:

| stage | paper steps | per-rank batch | world size | global batch | model |
|---|---:|---:|---:|---:|---|
| Coupled U-Net | 100,000 | 4 | 8 | 32 | 115.13M parameters |
| 64→256 SR | 270,000 | 6 | 8 | 48 | 311.03M parameters |

Both use a linear 1,000-step diffusion schedule. The released sampling script
uses 4 GPUs, batch 4, 64 base samples per launch, DPM-Solver for base sampling
and DDIM-25 for super-resolution. A smaller run is a process smoke only and
must not be presented as training reproduction.

## Dataset identity and provenance gates

The paper states that Landscape is reconstructed from 928 YouTube source videos
via URLs from Sound-guided Semantic Video Generation, split into 1,000
non-overlapping 10-second clips (about 2.7 hours / 300k frames). AIST++ has
1,020 clips, 5.2 hours / about 560k frames, with a center 1024×1024 crop.

The official README links one public Google Drive folder for `Landscape` and
`AIST++_crop`, but it does not publish an in-repository file manifest, URL list,
checksums, train/eval split, preprocessing command, or license statement. The
paper says only the training set is used for both training and evaluation.
Before running anything that claims paper fidelity, retain:

1. exact Drive file listing, ownership/access state and hashes;
2. raw MP4s, directory inventory and decode report;
3. any source URL/clip manifest and applicable licenses;
4. the `VideoClips` metadata cache provenance (which source files and runtime
   created it);
5. checkpoint hashes, evaluator-weight hashes and sampling seeds.

## Official artifact listing (audited 2026-09-18)

The linked Drive folder HTML was retained remotely before acquisition and its
displayed names, file IDs and anonymous download response headers were read
directly. This resolves the earlier listing gate; it does not yet validate file
contents or dataset semantics.

| role | official filename | Drive ID | bytes | first-lane decision |
|---|---|---|---:|---|
| Landscape media | `landscape.zip` | `10zrHOWMOtaMSCNi7qfQP5FN5qRlMvywa` | 654,090,471 | acquire |
| Landscape base | `landscape.pt` | `1PTl-ChGHbIAR0jNTEYCao8AkArKDUNsh` | 534,946,983 | acquire |
| Landscape SR | `landscape_SR.pt` | `1WNuSqwk9ng7AaFSnCiiyvSLjhFz-7f1I` | 1,244,225,895 | acquire |
| I3D evaluator | `i3d_pretrained_400.pt` | `1N9tAlQzhNnICExaye1jU1K3jTIkpq3G8` | 50,939,526 | acquire |
| AIST++ media | `AIST++_crop.zip` | `1Rx_-vtnqPmIVgOpd3YWrGi0Xf5cOBPf8` | 1,265,534,333 | defer |
| AIST++ base/SR | `AIST++.pt`, `AIST++_SR.pt` | `1toHyAGF-mM8zwmfKlqtn1CfcnIyC2ICg`, `1ZFJaqH2ig9l-ng0ERUAYbYCd0gcwmqn7` | 534,946,983 / 1,244,225,895 | defer |
| SR training init | `guided-diffusion_64_256_upsampler.pt` | `1hqTit_q9IwM98Y20Mpvc6sgLKm66vV00` | 1,247,310,648 | defer |

All eight headers returned attachment content type and the listed filename
without authentication. The first lane is intentionally Landscape only, so
there is one data and metric lineage rather than two partially audited ones.
Downloads target `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/mm-diffusion/`;
they will be renamed from `.part` only after byte-size and SHA-256 checks.

### Acquisition and raw-media audit

The four Landscape-lane artifacts now pass the Drive-header byte check and have
these local SHA-256 values: `landscape.zip`
`32df45107f5ab1ba6fff0d3382373d31490df9ccdde3b72c4aa71c9513cca6d5`,
`landscape.pt` `f21b7adf250bc4daa049c9331c7526c8af25579cf44cf3cf4bae299da6901f16`,
`landscape_SR.pt` `a8cb31ed06e749a92ce13cb93ad140f2714ad0eb5ea1326d6b9f9d2f2448f14e`,
and `i3d_pretrained_400.pt`
`55095f049e706479d48e221adcdb145b2b9dc930ba28b081ed72367ffaa32343`.

`unzip -t` passes. The preserved archive extracts to exactly 1,000 MP4s,
explicitly split as 900 `train` and 100 `test`; all 1,000 pass `ffprobe`.
They are H.264 video plus 16-kHz stereo MP3 audio, principally 512×288 at
10 fps, with durations 10.01--10.08 seconds. The raw archive, extracted MP4s,
per-file SHA-256 inventory and `ffprobe` summary remain together remotely.
This corrects the earlier wording: the archive contains 1,000 clips, not 1,020
(the latter was its count of ZIP directory entries). The loader's exact choice
of split for the paper metric is still a gate, not an assumption.

The gate is now resolved from the pinned release scripts: `multimodal_train.sh`
sets `DATA_DIR=.../landscape/train`, and `multimodal_eval.sh` sets
`REF_DIR=.../landscape/train` with `--sample_num 2048`. Thus the first metric
pass must use those 900 train clips as reference, while retaining the 100 test
clips untouched. This is an unusual train-reference protocol, but it is what
the authors released and must not be silently replaced by conventional testing.

### Runtime compatibility status

The current base image has PyTorch 2.8.0 / CUDA 12.8 and initially had neither
`mpiexec` nor a loadable MPI library. MM-Diffusion imports `mpi4py` before
constructing its models. A dedicated MPICH 5.0.1 runtime plus an isolated Python
environment were therefore used. Its legacy `av==10.0.0` cannot build against
the absent FFmpeg development headers on Python 3.11; the compatible installed
PyAV is 17.0.0. The repository's declared `decorator==4.4.2`, `pytorch-ignite`
0.4.11, `termcolor` 2.2.0 and `visdom` 0.2.4 were also necessary because the
sampling script imports the complete AudioCLIP evaluator even when no metric is
requested. No source code patch was used.

### Checkpoint execution result (C2 smoke, not a table result)

Under the complete official Landscape sampling architecture, both checkpoints
strict-load with **all keys matched**: base 133,675,524 parameters and SR
311,027,910 parameters. The base count differs from the paper supplement's
115.13M figure and remains an unresolved release/paper discrepancy.

On 2026-09-18, a single deterministic Landscape sample (`seed=42`) successfully
ran on GPU 4 through the official `ddpm` base branch (1,000 steps) and the
official `ddim` SR branch (`ddim25`). This is the paper's `Ours*` sampler family,
but with one sample only, hence it tests execution and media provenance rather
than any reported score. It retained a 64×64 original MP4, 256×256 SR MP4,
16-kHz WAV and all 16 PNG frames. All outputs pass `ffprobe`:

| output | SHA-256 | decoded streams |
|---|---|---|
| original MP4 | `0a17dbf4c7b283c5ccba02e8cfc61518ce06e3bbc4330fd0b3719ca4d2910102` | H.264 64×64/10 fps + 16-kHz stereo MP3, 1.70 s |
| SR MP4 | `bf48bb03c43b6bb35a8a6d226b1b4ef68fe47793b51bdbc8ac29ce02e4e5da04` | H.264 256×256/10 fps + 16-kHz stereo MP3, 1.70 s |
| WAV | `0836df9d211c8f1b4ea9714a85af6af2a8d9e58ee1c28e1b59d6e1917cea619b` | 16-kHz stereo PCM, 1.60 s |

Outputs and command-effective configuration are at
`/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/mm-diffusion/outputs/smoke-ddpm-retry1-20260918/`
and `logs/smoke-ddpm-retry1-20260918.*`.

The nominal default DPM-Solver path is independently blocked. In both
paper-era `c895b299` and current `7250222`, the multimodal sampler passes a
`{video,audio}` dictionary into `model_wrapper.model_fn`, which calls `x.shape`.
It crashes at the first solver step. This is a release defect, not a result
about model quality. Do not report the paper's default DPM-Solver row unless an
explicitly scoped compatibility patch is separately reviewed and labelled.

### Evaluator execution result (C2 self-smoke, not a benchmark result)

The exact I3D evaluator checkpoint and the upstream official AudioCLIP release
checkpoint are now pinned and have successfully passed the public evaluator
end-to-end. AudioCLIP is `537,302,068` bytes with SHA-256
`2441d35b353352c8b1bbfb8f7c687f46314c3d2909e940eaf763b8c17f632c44`;
I3D remains SHA-256
`55095f049e706479d48e221adcdb145b2b9dc930ba28b081ed72367ffaa32343`.

The official evaluator was run on the same two 16-frame clips extracted from
the single generated original MP4 as both reference and fake directories. It
completed I3D feature extraction, AudioCLIP video/audio feature extraction and
the released aggregation, yielding `FVD=-0.005901336669921875`,
`KVD=-1.350311279296875`, `FAD=3.055902197957039e-05` (current `×10000`
scaling). These near-zero self-comparison values validate execution only; a
two-sample covariance/KVD estimate is not statistically meaningful and must
never be compared with Table 1. The preceding one-sample run correctly failed
because a Fréchet covariance cannot be formed from one sample.

For the working runtime, the declared legacy versions needed explicit isolation:
NumPy 1.23.5, SciPy 1.10.1, scikit-learn 1.2.1, pandas 1.5.3, MoviePy 1.0.3,
decorator 4.4.2, pytorch-ignite 0.4.11, termcolor 2.2.0 and visdom 0.2.4.
PyAV 17.0.0 is the sole unavoidable deviation because the declared PyAV 10.0.0
cannot build under the available Python 3.11 without FFmpeg development headers.

## Execution order

1. Complete the official Landscape archive, base/SR checkpoints and I3D weight;
   verify byte sizes and SHA-256 hashes.
2. Inspect the archive and checkpoints without modifying their originals.
3. Reproduce one deterministic, explicitly non-table generation smoke using
   the official release config; retain MP4, WAV, input/noise provenance and
   `ffprobe` output.
4. Audit and acquire the exact Landscape source media; run complete decode and
   inventory checks.
5. Reproduce released-checkpoint samples and the 2,048-sample evaluator first
   under both `×1000` legacy and paper/current `×10000` FAD reports, labelled
   separately.
6. Only after the released-weight and data protocol are closed, run the
   100k/270k public-data training reproduction.

### Invalidated metric-run attempt; restart gate

An attempted run on 2026-09-18 (`seed=1001`, four MPI ranks nominally assigned
to GPUs 4--7, four samples per rank per group, `all_save_num=2048`, base DDPM
1,000-step and SR DDIM-25) is **invalid and terminated**. The official
`dist_util.setup_dist()` exposes one GPU per rank through `CUDA_VISIBLE_DEVICES`
but does not select logical device 0 before NCCL initialization. On the current
PyTorch/NCCL runtime all ranks bound to physical GPU 0; the run later stalled.
It produced 16 base MP4/WAV files and only 4 SR MP4 files. These raw files,
the log and immutable command provenance are retained at
`outputs/ddpm_ddim25_run01_seed1001_20260918/`,
`logs/ddpm_ddim25_run01_seed1001_20260918.log`, and
`commands/ddpm_ddim25_run01_seed1001_20260918.provenance.txt` as failure
evidence; they must never be evaluated, resumed or mixed into a result.

The initial restart gate also failed. A minimal, separately versioned patch
that calls `torch.cuda.set_device(0)` after restricting each rank's visible
device was applied to the immutable official checkout using `git apply`; the
binding script reported ranks 0--3 with visible devices 4--7 and logical device
0. That report is insufficient: its 16-sample DDPM/DDIM smoke ran through base
generation, retained 16 original MP4s and 16 WAVs, but produced only four SR
MP4s before rank 0 raised `torch.OutOfMemoryError` during SR group normalization.
The initial OOM reports logical `cuda:0`, which cannot identify a physical GPU.
A later UUID audit proves that, when `CUDA_VISIBLE_DEVICES` is set by the MPI
launcher *before* Python imports PyTorch, ranks 0--3 map one-to-one to physical
GPUs 4--7. It also shows GPU 6 already has about 40 GiB of external occupancy;
the smoke's rank 2 was routed there, so its logical `cuda:0` OOM is explained
by contention, not an incorrect physical-GPU mapping.

The next smoke must therefore exclude GPU 6. At the audit point GPUs 0, 1 and
2 each had only about 2.8 GiB external use and GPU 4 about 4 MiB, so the next
candidate set is 0, 1, 2 and 4, subject to a fresh `nvidia-smi` check directly
before launch. This is a runtime allocation choice, not a change to the paper
protocol or model configuration.

That replacement smoke completed on 2026-09-18 with the official Landscape
checkpoints, base DDPM-1000, SR DDIM-25, four ranks and batch 4/rank. It has
16 original 64×64 MP4s, 16 SR 256×256 MP4s and 16 WAVs; every one of the 48
media files passed `ffprobe` and is listed in the retained SHA-256 inventory.
The script's automatic 16-sample evaluator output was FVD `1658.1383`, KVD
`44.2481` and FAD `15.7497`. It is a process check only—not a benchmark result:
the paper calls for 2,048 generated samples and six runs. The next permitted
action is a new, separately named 2,048-sample DDPM/DDIM run 1 of 6 using the
same pre-import per-worker GPU binding, followed by independent evaluator audit.

### Active formal metric lane

Run 1 of 6 started on 2026-09-18 as
`ddpm_ddim25_metric_run01_seed42_preimport_gpu0124_20260918`. It uses the
same official base/SR checkpoints and sampler settings as the complete smoke,
but `all_save_num=2048`; four MPI workers bind physical GPUs 0, 1, 2 and 4
before Python imports PyTorch. The versioned launcher is
`commands/run_ddpm_ddim25_metric_lane.sh` (SHA-256
`e082df8d2e6c0de9ce367ac7ba231d4de43a632b1873f85bd8a53037d16e3c56`).
Its output directory and log are remote under `outputs/` and `logs/` with the
same run ID. It is not a result until the process completes, counts reach
2,048 original MP4 + 2,048 SR MP4 + 2,048 WAV, all media pass decode/hash
audit, and a separately invoked evaluator completes.

The current sampling script contains a second evaluation defect: after its
generation loop it iterates over `original` and `sr_mp4` labels, but calls
`eval_multimodal(..., multimodal_save_path, ...)` in both iterations. It thus
evaluates original 64×64 media twice. The official standalone `eval.py` script
instead accepts an explicit fake directory and the checked-in evaluation shell
passes `sr_mp4`. Once the run's media audit closes, this study will invoke the
standalone evaluator separately on `original` and `sr_mp4`, preserving both
input paths and never treating the sampling script's duplicated output as two
independent measurements. The evaluator itself decodes/reformats video to
16×64×64; this resampling is recorded for either input path.

The failed smoke is retained as `outputs/ddpm_ddim25_16sample_binding_smoke_20260918/`
with log, 36-file SHA-256 inventory, and successful `ffprobe` audit for every
retained file. It is invalid and must never be scored or mixed into a result.
Before any new 16-sample or 2,048-sample attempt, the replacement gate must
set each worker's `CUDA_VISIBLE_DEVICES` through the MPI launcher before Python
import and record each worker's CUDA hardware UUID against `nvidia-smi` GPU
UUIDs. It must additionally select four GPUs with enough unoccupied memory for
SR. Only a complete 16-original/16-SR/16-WAV, decode-audited run then unlocks a
fresh 2,048-sample run 1 of 6.

### Formal metric lane results — runs 1--2 of 6

The run reached terminal closeout on 2026-09-18. The conservative watcher first
counted exactly 2,048 original MP4, 2,048 SR MP4 and 2,048 WAV files (it counts
only the expected media suffixes; the evaluator's derived `.pkl` cache is not
media). It retained all-media SHA-256 inventory, found zero byte-identical
duplicate hashes, and `ffprobe` had zero failures across all 6,144 media files.
It then invoked the official standalone evaluator separately with explicit
`fake_dir` values; this avoids the released sampling script's original-twice
defect.

| independently evaluated fake media | FVD ↓ | KVD ↓ | FAD (`×10^4`) ↓ | relation to paper row |
|---|---:|---:|---:|---|
| paper Landscape `Ours*` DDPM-1000 | 117.20 | 5.78 | 10.72 | paper's six-run aggregate claim |
| local `original` (64×64), run 1/6 | 249.0004 | 15.4862 | 10.7107 | direct visual-resolution counterpart; FVD/KVD do **not** recover paper values in this run; FAD is close |
| local `sr_mp4` (256×256), run 1/6 | 149.9356 | 7.6905 | 10.1019 | separately evaluated diagnostic; evaluator reformats video to 16×64×64, so not a distinct paper table row |

This is C3 released-checkpoint evidence for the declared one-run protocol, not
C4 training evidence and not a reproduction of the paper's six-run claim. The
paper's generation seeds/sample sets are unavailable; run-to-run variance,
the current-runtime deviation (PyAV 17 versus unavailable pinned PyAV 10),
and paper/runtime evaluator alignment remain open explanations. Do not train
yet. First preserve this terminal evidence, audit the result discrepancy, and
decide whether additional preregistered DDPM/DDIM runs can distinguish seed
variance from a stable protocol mismatch.

Run 2, `ddpm_ddim25_metric_run02_officialrng_gpu4567_20260918`, subsequently
repeated the public checkpoint/data/sampler/evaluator protocol with the
official code's unseeded RNG behavior. Its only preregistered operational
difference was physical GPUs 4/5/6/7, selected because the original cards were
externally occupied. Closeout retained exactly 2,048 original MP4, 2,048 SR
MP4 and 2,048 WAV; `ffprobe` failures and byte-identical media duplicates were
both zero. The independently invoked evaluator returned:

| independently evaluated fake media | FVD ↓ | KVD ↓ | FAD (`×10^4`) ↓ | status |
|---|---:|---:|---:|---|
| local `original` (64×64), run 2/6 | 231.8951 | 14.3733 | 10.9198 | direct counterpart; again does not recover the paper FVD/KVD aggregate |
| local `sr_mp4` (256×256), run 2/6 | 154.7460 | 7.8836 | 10.5998 | diagnostic only, not a second paper row |

For the two original-media runs, descriptive means are FVD `240.4477`, KVD
`14.9298`, FAD `10.8152`; observed two-run ranges are `17.1053`, `1.1130`,
and `0.2091`. They are not a reliable variance estimate and cannot establish
a contradiction with the authors' six-run aggregate. Both runs are C3
released-checkpoint evidence only. A third run, if justified, must be
separately preregistered with the same public artifacts and protocol; neither
result authorizes training.

## Current non-claims

### Formal metric lane result — run 3 (affected observation; do not pool)

Run 3, `ddpm_ddim25_metric_run03_officialrng_gpu4567_20260920`, reached
terminal closeout on 2026-09-20. It retained exactly 2,048 original 64×64
MP4s, 2,048 SR 256×256 MP4s, and 2,048 WAV files. The all-media audit recorded
zero `ffprobe` failures and zero byte-identical duplicate hashes. The official
standalone evaluator was independently invoked once per explicit fake-media
directory, giving:

| independently evaluated fake media | FVD ↓ | KVD ↓ | FAD (`×10^4`) ↓ | status |
|---|---:|---:|---:|---|
| local `original` (64×64), run 3/6 | 222.6008 | 13.1663 | 10.8026 | direct counterpart; it again does not recover the paper's six-run FVD/KVD aggregate |
| local `sr_mp4` (256×256), run 3/6 | 137.6547 | 7.0204 | 10.9443 | diagnostic only; evaluator reformats input to 16×64×64 |

This is terminal C3 released-checkpoint evidence only. Crucially, after launch
GPU 6 acquired a separate external allocation of about 37.7 GiB; this work did
not stop or modify that process, and it later disappeared. Consequently run 3
is a **post-launch-GPU-6-interference observation**, not a clean four-GPU
repeat. Do not pool it with runs 1--2 for variance, use it to reject the paper
six-run aggregate, or treat it as training evidence. The terminal count,
audit, metric values, and SHA-256 hashes of the status/audit/evaluator logs are
frozen in `run_manifests/ddpm_ddim25_metric_run03_officialrng_gpu4567_20260920.json`.

- A released-checkpoint DDPM/DDIM media smoke and a two-clip evaluator
  self-smoke are locally reproduced; neither is a 2,048-sample or six-run
  metric reproduction.
- The paper table is not validated by the current public repository alone.
- No approximate dataset rebuild, toy training or private-data experiment is
  permitted as a substitute for this reproduction.
