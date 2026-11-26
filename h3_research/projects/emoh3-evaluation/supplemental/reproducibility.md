# Reproducibility (EmoH3) — exact software, data, and checkpoints

## Software environment
- GPU: single node, 8 × H100 (80 GB), CUDA 12.8
- Python env `h3ft` (conda):
  - torch 2.11 (cu128)
  - **diffusers pinned at commit `abc5e9b`** (the `MiniMaxH3Ref2VAModularPipeline` API used in inference)
  - transformers 5.16.1, peft 0.20.0, deepspeed 0.19.6
- Inference parallelism: FSDP2 (`torch.distributed.fsdp.fully_shard`) over 8 ranks;
  manual 8-proc launch (torchrun rendezvous is unavailable on the IPv6-less host);
  `FileStore` init (`H3_STORE_FILE`).
- Training parallelism: DeepSpeed-ZeRO-3, 8×H100, per-sample packing (see `Background`).
- ffmpeg/ffprobe from env `sglang310` for frame extraction and audio muxing.

## Base model
- MiniMax H3, weights at `/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3`
- Ref2VA mode; reference encoded by the video autoencoder (cover-resize → spatial encode →
  posterior sample under seed 42 → float16 → normalize → patchify), fed clean at timestep 0.999.
- Scheduler shifts: video 12.0, audio 3.0.

## Training data (`traindata3`, 22 clips, all generated on 8×H100)
- **10 preservation clips**: distinct character portraits (actors a01/a02/a03/a07/a09/a11/a16/
  a18/a21/a24) + their base-generated clip; reference and target share the emotion.
- **12 cross-emotion pairs** for character 1534125952 (portraits angry/joyful/sad/surprised);
  reference emotion A, target emotion B≠A — the supervision for "prompt overrides reference".
- All pairs use the Ref2VA layout: `[text | reference | target audio | target video]`, reference
  rows fed clean, loss on target rows only. Cache entries store `ref_video` rows.
- Cache dims (256² training): video (448,96), audio (74,32), prompt (15/17,5120), ref_video rows.

## Training hyperparameters (LoRA v3 = best model)
| Setting | Value |
|---|---|
| LoRA rank / alpha | 16 / 32 |
| Target modules | attention projections (q/k/v/o, gate) — LoRA applied to both transformer and text encoder |
| Steps | 900 |
| Training resolution | 256² (v3) / 768² (v4, ablation) |
| Optimizer | AdamW (DeepSpeed-ZeRO-3) |
| Node | 8×H100, ~15–80 min per run |

## Models evaluated
| Model | Data | Steps | Notes |
|---|---|---|---|
| base | — | — | no LoRA, 16-step inference |
| v1 | 9 preservation clips | 300 | reference-agnostic |
| v2 | 18 clips | 600 | reference-agnostic |
| **v3** | **22 clips (12 cross-emotion + 10 preservation), 256²** | **900** | **reference-conditioned; best config** |
| v4 | same 22 clips, 768² | 900 | resolution ablation (worse) |

Checkpoints: `runs/lora3/checkpoint-900.pt` (v3, 42.6 MB) and `runs/lora4/checkpoint-900.pt` (v4)
are on the training node; public hosting is in progress (see `RELEASE_UPLOAD.md`).

## Evaluation configuration
- Generation: 768×768×124 frames, Ref2VA, 16 inference steps, seeds 42/1234(/999),
  deterministic generator per seed.
- Judge: Qwen2-VL-7B-Instruct, 8 frames, greedy (see `judge_protocol.md`).
- Identity: CLIP ViT-L/14 cosine similarity (reference vs 3 frames: early/middle/late), mean;
  binary Qwen2-VL "same character?" on the middle frame (qualitative only).

## Reference portraits
- Eval tasks: `inputs/expressions_0825/` (v1), e.g. `a15_neutral_1537882489_v1.png` (tasks x01/x02).
- Cross-emotion training: `inputs/expressions_0825_v2/` (raw), e.g.
  `angry_1534125952_v1_raw.png`, `joyful_1534125952_v1_raw.png`, `sad_...`, `surprised_...`.
- Character IDs are the timestamps in filenames (e.g. a15 = character 1537882489).

## Full results
- `DEFINITIVE_RESULTS.md` — complete per-config verdict matrix (the paper's numbers).
- `release/judge_logs/*.jsonl` — raw judge outputs for every generation used in the paper.

## Hosting plan (in progress)
- HF repo (checkpoints + judge logs + verdict matrix) and GitHub (code + data prep) —
  see `RELEASE_UPLOAD.md` for the exact commands once a token is available.
