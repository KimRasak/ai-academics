# Human Spot-Check of the Judge (M2 revision) — annotation kit

**Goal.** Validate the Qwen2-VL-7B judge on our anime emotion taxonomy with a small human sample,
so the paper can report judge–human agreement instead of trusting the VLM alone.

## Sample
24 clips = 8 tasks × 3 models (base, LoRA-v3, LoRA-v4), 16 inference steps, seed 42, middle frame
of each clip shown (the judge uses 8 frames; a single well-chosen frame is enough for a human label
and keeps labeling fast). For x02 add seed-1234 (the sad breakthrough case) → 26 clips total.

Frames live on the cluster as `frames_<...>/<case>/f*.jpg` (first ~8 are evenly spaced). For
labeling, export one frame per clip into `spotcheck/clip_<case>.jpg`.

## Label taxonomy (same 12 classes as the judge)
`happy, sad, angry, surprised, anxious, curious, neutral, disgusted, fearful, amused, relieved, other`

## Instructions for annotators (per clip)
1. Look at the character's face in the single shown frame.
2. Pick **exactly one** emotion word from the taxonomy that best matches.
3. Optionally give a 0-100 confidence.

## CSV template (`spotcheck/labels.csv`)
```csv
clip_id,annotator,human_emotion,human_conf,judge_emotion,judge_conf,note
x01_neutral_to_joyful_base,A,,,happy,90,
x01_neutral_to_joyful_lora,A,,,happy,80,
...
```
`judge_emotion/judge_conf` are pre-filled from the released judge logs (8-frame, 16-step, seed 42).

## Analysis (small script, ~20 lines)
- Per-annotator: judge–human exact-match accuracy over the 26 clips.
- Inter-annotator: Cohen's $\kappa$ on the 2 annotators (each annotates all 26).
- Report in the paper's evaluation section: e.g. ``judge agrees with humans on X/26 clips
  ($\kappa{=}0.xx$), validating the automated protocol''.

## Expected outcome for the paper
- If agreement is high ($\ge$0.8 exact-match), add one sentence + the numbers to the paper.
- If low, report it honestly and rely on the multi-seed/multi-frame robustness framing already in
  the paper.
