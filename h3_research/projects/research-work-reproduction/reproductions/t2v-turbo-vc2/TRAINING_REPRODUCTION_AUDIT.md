# T2V-Turbo VC2 training-reproduction audit

**Date:** 2026-09-17  
**Decision:** do not start the 10,000-step run

## What is reproducible

The first official training release is commit
`ca61d382aaaa91b9d526a912b1749096d166c0e3`, dated 2024-06-24. A clean detached
worktree is retained at:

`/gluster_hdd_cv/user/jinzili/code/research-work-reproduction/t2v-turbo-training-release-20240624`

The initial files were named `train_t2v_turbo_vc2.py` and
`train_t2v_turbo_vc2.sh`. They were renamed to the current v1 names on
2024-10-09 without changing the training program.

The release fixes the main optimization settings: fp16, seed 453645634, LoRA
rank 64, 10,000 steps, batch one per process, 16 frames at 16 FPS, learning rate
1e-5, guidance range [5, 15], 50 DDIM timesteps, skip 20, HPSv2.1 scale 1 and
InternVideo2 scale 2.

The VideoCrafter2 base and HPSv2.1 checkpoint are present and hashed. The HPS
checkpoint also passes `torch.load`.

## Why training cannot start

### WebVid identity is unrecoverable

The paper names WebVid-10M but does not publish the selected samples, ordering,
download date, failed-download policy or shard hashes. The release command
contains only:

`/path/to/webvid_data/train_data/{00000..99999}.tar`

The loader requires `mp4` and `txt` keys, decodes 16 frames at 16 FPS and
resizes/crops to 320×512. Neither the shard builder nor the exact
`video2dataset` version is pinned.

The official WebVid project now states that URLs and captions are no longer
available. No WebVid raw media, metadata or shards exist in the in-scope remote
roots. A newly assembled substitute could support a declared-data reproduction,
but not the paper's original-data claim.

Official sources:

- `https://github.com/m-bain/webvid`
- `https://github.com/m-bain/webvid/blob/main/TERMS.md`

### Exact video reward is gated and missing

The code requires:

`OpenGVLab/InternVideo2-Stage2_1B-224p-f4/InternVideo2-stage2_1b-224p-f4.pt`

at repository revision
`4362e1f88a992e7edbfd7696f7f78b7f79426dfd`. The current account receives HTTP
403 from this gated repository. The README instead links
`InternVideo2-CLIP-1B-224p-f8`, which is a different gated model and must not be
substituted silently.

### Released environment is incomplete

The training release contains no requirements file, environment lock, container
or Accelerate configuration. It imports the external `video2dataset` package
without a version. Reconstructing a runnable modern environment is engineering
work, not proof of the authors' environment.

### Six-GPU wording conflicts with code defaults

The paper states 6–8 A100 GPUs. The released defaults assign distillation and
image reward to ranks 0–5 and video reward to ranks 6–7. With six processes, no
rank executes the video-reward branch. The unmodified official command is
therefore effectively an eight-process protocol.

## Gate for a future training run

Training may start only after:

1. raw videos, captions, source URLs, license state and media hashes are
   retained;
2. deterministic shard and sample manifests are frozen;
3. authorized access to the exact InternVideo2 f4 checkpoint is available;
4. the environment and `video2dataset` implementation are locked;
5. one-batch decode, both reward paths, backward, optimizer and resume smokes
   pass on the clean release worktree.

Unless the authors' exact WebVid manifest is recovered, any eventual run must
be labelled a declared-public-data training reproduction, not an exact
paper-data reproduction.
