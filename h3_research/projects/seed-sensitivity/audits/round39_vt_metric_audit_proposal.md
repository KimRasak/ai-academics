# Round 39 proposal: Sci-VBench VT reproducibility & seed-power audit

Status: proposal + Round-39 offline evidence. No training, no judge endpoint, no new benchmark.
Evidence script: `analyze_vt_dominance.py` → `audits/round39_vt_dominance.json`.

## Why this experiment

The Judge Gate is blocked (Rounds 36/38: no Sci-VBench Qwen3.5 judge endpoint; Qwen2-VL must not be
substituted). The VT Gate is passed (Round 35: 32/32 fresh H3-Base videos, six metrics, 0 errors).
So the only dimension we can currently measure locally is exactly the one the Sci-VBench paper uses
as its automatic proxy for LPF — and it turns out to be measurably fragile. That makes a metrology
paper reachable in the one-month window with the assets already on disk.

## Round-39 offline findings (already computed, zero GPU)

All from released artifacts (`vt_scores.jsonl`, 11,216 rows) plus our 32 fresh local VT records.

1. `VT` is the unweighted mean of the six VBench Video-Quality metrics (max residual 1.1e-16).
2. On testmini across 16 models, **dynamic degree carries 84.9 % of the between-model variance** of
   VT (between-model SD: DD 0.1503, imaging 0.0420, subject 0.0365, aesthetic 0.0275,
   background 0.0125, motion smoothness 0.0029). Motion smoothness is saturated (≈0.99 for every
   model) and contributes 0.03 %.
3. Dropping DD and ranking the same 16 models by the other five metrics gives **Kendall τ = 0.05**
   against the published VT order — the leaderboard order is essentially the order of one binary
   "is there motion" detector.
4. Adjacent-model VT gaps on testmini are tiny: median 0.0049, max 0.0129.
5. Our fresh local H3-Base reproduction on the same 16 verbatim prompts (2 seeds) differs from the
   released MiniMax-H3 rows by **VT +0.0440, bootstrap 95 % CI [0.0068, 0.0808]**, and the delta is
   almost entirely DD (+0.281); all five continuous metrics agree within ±0.023.
6. Seed instability on our own outputs: mean |seed0−seed1| VT gap 0.0721 (max 0.2167), and **DD
   flips for 7/16 prompts (43.75 %) between just two seeds**.

Items 4–6 together are the core tension: the released protocol scores one video per prompt per model,
yet a single seed change moves per-prompt VT by an order of magnitude more than the gaps that
separate adjacent leaderboard entries.

## Proposed protocol

### E1 — Human-reference and settings join (offline, ~2 h, 0 GPU)

Download text-only artifacts from `Sci-VBench/Sci-VBench-Videos` (HF reachable, API 200):
`human_scores.jsonl` (2,392 expert testmini ratings), `index.jsonl`, and
`settings/<video_id>/MiniMax-H3.json` (1,253 sidecars). Record SHA-256 for each, as in Round 2.

Then compute:
- per-video Spearman of VT and each of the six components against the expert `lpf` rating, pooled
  and within-model;
- model-level Kendall τ of VT order vs expert-LPF order, with and without DD;
- the released MiniMax-H3 generation envelope (resolution, frames, fps, sampler steps, guidance,
  seed) versus our pilot envelope (5.175 s, 1344×768, 24 FPS).

E1 is a prerequisite for interpreting finding 5: until the sidecars confirm the envelope, the
+0.0440 delta cannot be attributed to seed noise rather than a duration/resolution difference.

### E2 — Evaluator reproduction check (≈1 GPU-hour)

Download the 150 testmini `videos/<video_id>/MiniMax-H3.mp4` (~450 MB, `allow_patterns`), re-score
them with our pinned stack — Sci-VBench commit `e8cb3988631d0c22d31c02fc5b1cc3693bfb7031`,
`/ssd2/jinzili/envs/scivbench_vt`, the verified 1.9 GB asset root, physical GPU 3 exposed in-process
as `cuda` — and compare per-video against the released rows.

### E3 — Seed-power measurement (≈1.5 h of the existing 034 Base service, +~20 min VT)

Extend the frozen 16-prompt pilot from 2 seeds to 6 (seeds 0–5): 64 new generations through the
already-healthy 034 FL2VA Base service, identical raw `posted_prompt`, identical 5 s / 768p / 16:9
envelope, `lora_path=null`, sequential single-request submission via the existing
`run_scivbench_pilot.py` and per-case evidence layout. Score all 96 videos with the E2-validated VT
stack. Then report per-prompt DD flip rate, model-level VT standard error as a function of seed
count k, and the minimum detectable difference at 150 prompts × k seeds against the observed
median adjacent-model gap of 0.0049.

### E4 — Generality check (only if E2 passes; ≈1 GPU-hour)

Re-score the 150 testmini videos of two more open models (Wan2.2-5B, HunyuanVideo-1.5, opposite
ends of the DD range) so the claim is about the metric, not about H3.

## Go / No-Go

| Gate | Go | No-Go / re-scope |
|---|---|---|
| E1 | Released MiniMax-H3 sidecars show the same 768p / 24 FPS / ~5 s envelope as our pilot | Envelope differs → the paired delta is confounded; report it as a protocol-sensitivity result and drop the "same-configuration reproduction" framing |
| E2 | Median per-video \|Δ\| ≤ 0.005 on the five continuous metrics **and** DD exact match ≥ 95 % of 150 videos | Otherwise stop comparing our numbers to released numbers; re-scope to an implementation-sensitivity report and do not run E3 |
| E3 | DD flip rate ≥ 25 % across 6 seeds **and** single-seed MDD > 0.0049 (median adjacent gap) | Flip rate < 10 % **and** single-seed MDD < 0.0049 → VT is adequately powered; abandon this direction and return to the modality-aware cache candidate (README §6) |
| E4 | DD variance share > 50 % and the seed instability reproduces on ≥ 1 other model | H3-specific → weaken to a model-level observation, not a benchmark claim |

Whole-experiment kill switch: if E2 fails, the total spend is one download plus one GPU-hour and no
paper claim is made.

## Cost and schedule

No training, no judge endpoint, no new benchmark, ≈3–4 GPU-hours total on already-provisioned
resources, ~600 MB of downloads. E1 and E2 fit in one working day; E3's generation is ~80 minutes of
sequential service time; E4 is optional. This leaves the remaining window for writing.

## Venue fit

A 4-page measurement paper matches WVAQ @ WACV 2027 (tentatively Oct 12 — the official CFP must be
opened and verified before committing, per README §8). Fallbacks: an ICASSP 2027 satellite (tentative
Nov 11) or a reproducibility venue (ReScience C / TMLR) if the window slips.

## Stated limitations (must appear in any write-up)

- 16 prompts × 6 seeds is a power study on a small prompt set, not a re-run of the 1,253-video
  benchmark; per-prompt DD is Bernoulli, so flip rates carry wide intervals.
- VT is only the paper's LPF proxy. PG/SCC/SC require the released Qwen3.5 judge, which we do not
  have; no claim may be made about them, and the released SCC≈1.63 remains an author-reported number.
- Our fresh videos come from the local open-weight H3-Base with the raw prompt; hosted Context-IR
  and Regenerate-2K are not reproduced.
- Dynamic degree is a thresholded RAFT-flow statistic; showing that it dominates VT is not evidence
  that it is wrong, only that VT inherits its binary, seed-sensitive behaviour. The human-score join
  in E1 is what decides whether the dominance helps or hurts agreement with experts.
