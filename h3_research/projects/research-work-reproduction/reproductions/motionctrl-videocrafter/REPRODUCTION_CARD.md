# MotionCtrl (VideoCrafter1/LVDM) reproduction card

## Question

Can the released `motionctrl.pth` reproduce the SIGGRAPH 2024 paper's
VideoCrafter1/LVDM control metrics, and can the reported three-stage training
process be reconstructed from public artifacts within eight GPUs?

## Paper claims

The paper uses VideoCrafter1/LVDM, not the later VideoCrafter2 CMCM checkpoint.
Its main comparison reports:

| metric | MotionCtrl |
|---|---:|
| CamMC, basic poses | 0.0289 |
| CamMC, complex poses | 0.0735 |
| ObjMC | 28.877 |
| CLIPSIM | 0.2319 |
| FID | 124.09 |
| FVD | 852.15 |

Training uses 8×V100, batch size 128 and Adam:

1. CMCM on caption-augmented RealEstate10K for about 50k iterations;
2. OMCM on 243k ParticleSfM-augmented WebVid clips with dense trajectories for
   20k iterations;
3. OMCM sparse-trajectory fine-tuning for another 20k iterations.

## Public artifacts

- Paper and supplement: complete enough to state the claims and dataset sizes.
- Main code revision: `6242d55414d3f4ad690bf0934889660f78e81b6e`.
- First public VideoCrafter/LVDM implementation: commit
  `bb00d47a017e55ae0931434e76656edfcde2b9b8`.
- Public training branch: `animatediff` at
  `08cae00c7020f1115aecfca352af899e23cc3f19`.
- Paper-model checkpoint: `motionctrl.pth`, 4,016,205,312 bytes, Git-LFS SHA-256
  `efe225096946c804841c2a78c3624d77cade3ac021a2519a2f0f85708a0fe46c`.

The checkpoint is now present remotely and its file hash matches the upstream
Git-LFS object. It contains 2,106 FP16 state tensors and 2,007,751,564
parameters. The official configuration accepts it with zero missing and zero
unexpected keys.

## Evaluation audit

The paper evaluates:

- 407 camera-control samples: 80 basic, 200 RealEstate10K, 100 WebVid and
  27 HD-VILA;
- 283 object-control samples from 74 trajectories and 77 prompts;
- FID/FVD against 1,000 WebVid reference videos;
- CamMC/ObjMC by rerunning ParticleSfM on generated videos.

The repository does not release that complete protocol:

- main inference lists 22 camera prompt/pose pairs and 6 object prompt/trajectory
  pairs, intended as examples rather than the paper benchmark;
- only 17 basic and 20 RealEstate10K camera JSON files and 8 trajectory examples
  are present;
- the 100 WebVid and 27 HD-VILA camera cases, the 283 object pair manifest, and
  the 1,000 reference-video manifest are absent;
- no CamMC, ObjMC, CLIPSIM, FID or FVD evaluation implementation is included.

Therefore the released checkpoint can support a deployment smoke test and a
new, explicitly independent benchmark, but not a faithful reproduction of the
paper's Table 1 numbers.

## Training audit

The released training code targets AnimateDiff rather than the paper's
VideoCrafter1/LVDM experiment. It is useful implementation evidence but is not
the paper-matching training program.

Additional release defects prevent direct execution as disclosed:

- all three shell scripts first set 8 GPUs and then overwrite the setting to one
  GPU;
- `train_omcm_sparse.sh` invokes `omcm_dense.yaml`;
- all released configs specify 1,000,000 steps, not 50k/20k/20k;
- CMCM's published batch-128 protocol is not instantiated by the released
  batch-1, accumulation-4 config;
- the RealEstate10K caption/list files and the exact 243k WebVid selection and
  `motion_seg_list_v2.txt` are not released;
- WebVid's original URL/caption distribution is no longer available;
- evaluator code, checkpoint-selection rule and paper seeds are absent.

## Evidence decision

- Paper metric claims: **C1 / unverified protocol**.
- Released VideoCrafter1 checkpoint: **C2 deployment smoke complete** under an
  explicitly H100-adapted environment.
- Exact paper evaluation availability: **E**.
- Exact paper training availability: **E**.
- AnimateDiff training release: a separate approximate/deployment lineage; it
  must not be presented as reproduction of the paper's VideoCrafter1 tables.

## Next action

Do not allocate a full benchmark or training run yet. First:

1. extend the completed one-condition camera smoke to the released
   22-camera/6-object example set only if qualitative coverage is useful;
2. decide whether to construct a new public camera-control benchmark with a
   frozen evaluator, clearly separated from the unavailable paper protocol.

Training starts only if a legally usable raw-data corpus and complete
provenance manifests are available. Any newly reconstructed corpus is an
approximate reimplementation, not exact paper reproduction.

## Completed checkpoint smoke

On 2026-09-17, `test_camera_L` was generated with the released coastline prompt,
seed 1234, 50 DDIM steps, eta 1.0 and guidance 7.5:

- H.264, 256×256, 16 frames at 10 fps;
- full decode passed;
- generated video SHA-256
  `3c7fdb135ce411513d08ade0d0e79ce6e2745c6c541b1bfc9c7842c2a4434b09`;
- four-frame inspection shows a coherent coastline and visible lateral camera
  motion, with minor boundary deformation.

The original release environment cannot execute xFormers attention on H100.
The smoke therefore uses Torch 2.3/CUDA 12.1 and xFormers 0.0.26.post1, and is
explicitly labelled a compatibility adaptation.
