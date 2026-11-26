# EmoH3 — Release Package

**Paper** (current submission target: **IEEE ICASSP 2027**): *Emotion-Control Evaluation Is Not
Invariant: Judge, Option Order and Judged Frame Span Flip 11--58% of Verdicts on MiniMax H3*.

This is a measurement paper. Its central result is that three essentially free scoring choices each
move a large fraction of verdicts on identical clips: judge identity (28--58%), answer-option order
(11--19%) and how much of the clip the judge sees (33--35%, the only axis with significant effects).
Following that third axis led us to run the matched experiment our own earlier comparison lacked --- the
base model regenerated on all 8 tasks x 2 seeds under identical inference settings --- and it returns a
null result under all four judges (base 4/16, 6/16, 9/16, 8/12 vs LoRA-v3 6/16, 6/16, 8/16, 8/12; Fisher
p >= 0.70; Qwen3.8-27B parses 12 of the 16 cells).
The paper's earlier task-level claim is therefore **retracted**, and the released artifacts include every
log needed to check that retraction. Reproduce the three axes with:

```
python analyze_agreement88.py    # judge identity: pairwise/Fleiss agreement, option-order control
python analyze_framespan.py      # judged frame span: 0-1 s vs full clip, 3 judges x 97 clips
python analyze_basefull.py       # paired base-vs-v3 matrix (8 tasks x 2 seeds) + all-pairs reliability
python analyze_gt_accuracy.py    # ACCURACY of both frame spans against external labels (14 clips)
```

`analyze_gt_accuracy.py` is the one check that speaks to correctness rather than agreement: the 14 source
clips referenced by the training manifest come from the same production batch as the known-emotion
portraits and carry their intended expression in the filename, so both protocols can be scored with no
annotation of ours. Judging the first second recovers the label 0/14, 0/14 and 5/14 (strict) against 3/14,
4/14 and 9/14 for the whole clip; all six judge x scoring-rule comparisons improve and all 25 clip-level
correctness changes favour the whole clip. Caveats, also stated in the paper: the labels are the batch's
intended expressions rather than our own annotation, and emotions no judge can emit (anxious, curious,
relaxed) fail under both protocols.

Generation and judging drivers used for the paired matrix are included as-is:
`run_basefull4.sh` (4-GPU resumable base generation, 8 tasks x 2 seeds, with the ownership-checked lock
and the dynamic GPU-quad fallback needed on a shared node), `run_judge_full3.sh` (full-span pass for the
three fast judges), `run_judge_bf38.sh` (bounded Qwen3.8-27B pass) and `run_gt.sh` (stages the 14
externally labelled clips, extracts both frame spans, judges both).

Judge logs live in `judge_logs/`. `judge_full_{q2,q25,q3vl}.jsonl` are the full-clip re-scorings (97
configs each), `judge_basefull_{q2,q25,q3vl,q38}.jsonl` the 0--1 s scorings of the newly generated paired
base matrix (16 cells each; q38 parses 12), and `judge_all88_*_rev.jsonl` the reversed-option-order runs.
Qwen3.8-27B has no full-span log: in plain `transformers` it emits a long deliberation before its
`WORD:SCORE` line and runs at minutes per clip, so no full-span number depends on it.

Venue-specific builds: `icassp2027/` (primary; 4 pp of content + references), `icme2027/` (7 pp of
content), `neurips2026/` (9 pp of content + references). Generic source: `paper_emoh3.md`,
`paper_emoh3.tex` (9 pp).

This directory accompanies the paper with the artifacts used to compute every number.
- `supplemental/` — judge protocol (exact prompt + 12-class taxonomy), reproducibility sheet
  (software versions, data, hyperparameters), human spot-check kit, human eval protocol.

## Contents
- `paper_emoh3.pdf` — **compiled 9-page 2-column paper** (pdflatex-compatible; built with Tectonic).
- `paper_emoh3.md` / `paper_emoh3.tex` — paper source (LaTeX uses only standard packages).
- `paper_figure1.png` — Figure 1: qualitative base-vs-v3 comparison (4 tasks × reference/base/v3).
- `DEFINITIVE_RESULTS.md` — the complete verdict matrix (all models × seeds × steps × frames).
- `judge_logs/` — raw Qwen2-VL-7B judge outputs, one JSONL per configuration:
  - `pilot_judge_results.jsonl` — base, 8 steps, seed 42, 4 frames
  - `lora_judge_results.jsonl` — LoRA-v1, 8 steps, seed 42, 4 frames
  - `lora2_judge_results.jsonl` — LoRA-v2, 8 steps, seed 42, 4 frames
  - `lora2s_judge_results.jsonl` — LoRA-v2, 8 steps, seed 1234, 4 frames
  - `judge8_results.jsonl` — LoRA-v2, 8 steps, seed 42, 8 frames (subset of cases)
  - `control_judge_results.jsonl` — 16-step controls (base & LoRA-v2, seeds 42/1234, 8 frames)
  - `judge_x02ctrl.jsonl` — **decisive control**: base, x02 neutral→sad, 16 steps, seed 1234 → `neutral:100`
  - `v3_judge_results.jsonl` — LoRA-v3 (reference-conditioned), 16 steps, seeds 42 + 1234, 8 frames
  - `v4_judge_results.jsonl` — LoRA-v4 (reference-conditioned, 768²), 16 steps, seeds 42 + 1234, 8 frames
  - `judge_s999.jsonl` — LoRA-v3, third seed (999), x02/x07, 16 steps, 8 frames
  - `identity_lora3.jsonl` / `identity_lora3s.jsonl` — identity-consistency judge (ref vs middle frame, seeds 42/1234)
  - `clip_identity_all.jsonl` — **CLIP ViT-L/14 identity similarity** (3-frame mean, base + v3 both seeds)
  - **Four-judge cross-check (round-21)**: `judge38_results.jsonl` (Qwen3.8-27B, 53 cases, latest Qwen generation, incl. raw reasoning), `judge3vl_results.jsonl` (Qwen3-VL-32B, 49 cases), `judge_extra.jsonl` (v2raw re-judges), `judge15_35.jsonl` (Qwen2-VL full 35 portraits), `MASTER_4JUDGE.md` (cross-comparison), and `verify_judge_counts.py` (derives every headline judge number — strict/lenient sanity, negative-emotion counts, decisive x02 — directly from these raw logs; run with `python3 verify_judge_counts.py`).


## Model checkpoints (on the 8×H100 training node; 43 MB each)
- `runs/lora3/checkpoint-900.pt` — **best model**: reference-conditioned LoRA, 256² training, 22 clips, 900 steps. This is the model behind the sad breakthrough and 4/8 @ seed 1234.
- `runs/lora4/checkpoint-900.pt` — reference-conditioned LoRA, 768² training (the resolution ablation, deliberately worse).
- LoRA config: rank 16, alpha 32, targets = attention projections (`to_qkv`, `to_out.0`, `linear_1`, `linear_2` + `time_embedder`).

## Training data
- `traindata3/cache/` (256²) and `traindata4/cache/` (768²): 22 clips each — 10 emotion-preservation + 12 cross-emotion pairs (reference emotion A → target emotion B ≠ A, character 1534125952). Each cache entry: patchified target video latent rows, real audio latent rows, Qwen3-VL text embeds, and the *reference* latent rows (the conditioning added by this work).

## Evaluation protocol (all numbers reproducible; round-19 revision)
- Generator: MiniMax H3 Ref2VA, 768×768×124 frames, FSDP2, 8×H100.
- Judge: Qwen2-VL-7B-Instruct, `do_sample=False`, fixed **15-class** emotion taxonomy (round 19 primary: 12 released classes + joyful/worried/cheerful), `WORD:SCORE`. The 12-class logs remain for taxonomy-sensitivity comparison.
- **Judge validity (round 19)**: on 21 known-emotion reference portraits the judge labels 3/21 correctly (7/21 lenient), strong neutral bias; it is competent on exactly the emotions we claim (sad, surprised). Successes are conservative lower bounds.
- **Judge conservatism (round 19)**: across 100+ clip-configs the judge emits only 4/15 classes (neutral/happy/sad/surprised); worried/relieved/angry are untestable → their task "failures" are unattributable.
- Every headline is reported as (model, steps, seed, frames, taxonomy); multi-seed agreement is the success criterion (see paper §4 for why single-config reports are misleading). Statistics are reported honestly: the headline 1/8 vs 0/7 difference is not significant (Fisher p=1.00) and is presented as a controlled demonstration.
- New in round 19 (`judge15_8fr.jsonl` 88 clips, `judge15_4fr.jsonl` 88 clips, `judge15_extra.jsonl` 24 configs, `judge_sanity15.jsonl` 21 reference portraits, `judge15.py` judge script): full 15-class re-judgment of every clip-config; verdict-matrix appendix regenerated; 4/88 frame-count flips; taxonomy flips documented.

## Request for reviewers
- Reproducibility of *behavior* requires the exact generation seeds (42/1234) and judge settings above; checkpoints plus the judge script are sufficient to reproduce the verdicts.
