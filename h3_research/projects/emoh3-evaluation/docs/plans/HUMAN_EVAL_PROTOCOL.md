# EmoH3 — Human Evaluation Protocol (ready to run)

This protocol operationalizes the human study named as future work in the paper (§7, Limitations:
"a human study is future work"). It is designed to be run by 3–5 annotators with anime familiarity
in ~40 minutes, and produces a small table that can replace or corroborate the Qwen2-VL judge for
the headline claims (emotion control + identity preservation).

## 1. Stimuli
Per task × model pair, show a **12-second muted clip** (the exact generated `*_lora.mp4` /
`*_base.mp4` at 16 steps, seed 1234) with the reference portrait displayed above it.
- Tasks: the 8 emotion-transfer tasks (x01…x08), the decisive control pair (x02 base vs x02 v3,
  both seed 1234), and optionally x07 base vs v3.
- Models: **base** vs **LoRA-v3** (the two that matter for Findings A/B). ~16 clips total.
- Clips are muted because the paper evaluates video-only emotion (audio is trained but not
  emotion-evaluated).

## 2. Interface
Grid layout: one row per clip = [reference portrait | video player | three answer widgets].
Randomize clip order per annotator; hide which model produced each clip (label only "A"/"B").

## 3. Questions per clip
1. **Emotion** (single choice): 12-class taxonomy identical to the judge —
   happy, joyful, cheerful, amused, sad, worried, anxious, surprised, angry, relieved, calm/neutral, other.
2. **Emotion intensity** (1–5, integer).
3. **Identity** (single choice): "The character in the video is the same as the reference portrait?"
   — Definitely yes / Probably yes / Probably no / Definitely no. (Convert to 1–4.)
4. **Quality** (1–5): overall clip quality (motion, rendering artifacts), to guard against
   quality-driven emotion answers.

## 4. Instructions to annotators (verbatim template)
> You are rating short anime character videos. Each task shows a character portrait and a video
> that should animate that same character with a specific emotion. Answer (1) which emotion the
> character is most clearly showing, (2) how strongly, (3) whether it is the same character as the
> portrait, and (4) overall video quality. Judge what you actually see — there are no "correct"
> labels; we are measuring agreement between humans and an automatic judge.

## 5. Analysis
- **Agreement**: Cohen's κ (or simple % agreement) between (a) each annotator and the Qwen2-VL
  judge's emotion class, and (b) annotators pairwise. Report both.
- **Headline check**: % of annotators who label x02-v3 as "sad" vs % who label x02-base as "sad"
  (expect sad > neutral for v3; both neutral for base) → supports Finding A.
- **Identity**: mean identity score (1–4) for base vs v3 across shared tasks → corroborates the
  CLIP result (base 0.80 vs v3 0.73).
- **Failure region**: confirm annotators also fail to see worried/angry in x04/x08 across models.

## 6. Anticipated report
One table, ~5 rows, to include in the paper's Appendix or to replace the judge-only framing:
| metric | base | LoRA-v3 |
|---|---|---|
| human–judge emotion agreement (κ) | ? | ? |
| % annotators: x02 sad | 0–10% | ? |
| mean identity (1–4) | ? | ? |

## 7. Files needed (already on cluster / release)
- Clips: `outputs_lora3_s1234/*.mp4`, `outputs_base16x02/*.mp4`, `outputs_base16b/*.mp4`
- References: `inputs/expressions_0825/{a15_neutral,a02_angry,a11_joyful,a01_amused,a03_anxious,a18_sad,a07_curious}_*.png`
- Judge results to compare against: `release/judge_logs/v3_judge_results.jsonl`, `judge_x02ctrl.jsonl`
