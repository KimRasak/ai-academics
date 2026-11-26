# Judge Protocol (EmoH3) — exact evaluation configuration

This document specifies the automatic emotion judge used for the verdicts in the paper, so that
the evaluation is fully reproducible. The judge is a single open vision-language model,
**Qwen2-VL-7B-Instruct**.

## Model and inference settings
- Model: `Qwen2-VL-7B-Instruct` (`/gluster_hdd_cv/user/jinzili/weights/reward_models/Qwen2-VL-7B-Instruct`)
- Loaded with `transformers`, `torch.bfloat16`, `device_map="cuda"`
- Generation: `do_sample=False`, `max_new_tokens=32` (greedy, deterministic)
- Input: the character's face is shown via the first **8** (robust protocol; round-19 primary)
  uniformly spaced frames of the 124-frame clip, each converted to RGB JPEG. Frame-count
  sensitivity is measured by re-judging on the first 4 frames.

## Emotion taxonomy (15 classes, fixed; round-19 primary)
```
happy, joyful, sad, angry, worried, cheerful, surprised, relieved, anxious, curious, neutral, disgusted, fearful, amused, other
```
The 15 classes extend the released 12-class protocol (happy, sad, angry, surprised, anxious,
curious, neutral, disgusted, fearful, amused, relieved, other) with the three task targets that
were missing (joyful, worried, cheerful), so that the success criterion is literal for every task.
**The judge never emits 9 of its 15 classes** (only neutral/happy/sad/surprised occur across the
full round-19 re-judgment of 100+ clip-configs), so worried/relieved/angry (and joyful/cheerful as
distinct words) are untestable with this judge.

## Exact prompt (verbatim, 15-class)
> You are an anime expression evaluator. Look at the anime character's face across these video
> frames (in temporal order). Which emotion is the character expressing? Answer with exactly one
> word from: happy, joyful, sad, angry, worried, cheerful, surprised, relieved, anxious, curious,
> neutral, disgusted, fearful, amused, other. Then a colon, then a confidence score 0-100. Format:
> WORD:SCORE

(The 12-class prompt used in the released 12-class logs is identical with the 12-word list.)

## Output parsing and success criterion
The model emits `WORD:SCORE` (e.g. `sad:80`). **Success** = `WORD == target`, with one stated
synonym rule: joyful and cheerful count via their in-taxonomy synonym **happy**. Scores are the
VLM's verbalized confidence (0-100); they are **not** calibrated probabilities.

## Judge-validity sanity check (round 19; results in `judge_sanity15.jsonl`)
On 21 known-emotion reference portraits (the dataset's own labeled a/b/c/d-series and v2 raw
portraits) the judge labels **3/21 correctly** (7/21 if `joyful≡happy` counts), with a strong
neutral bias. It is correct on exactly the emotions we claim (sad, surprised) and collapses the
rest to neutral/happy. Implications (disclosed in the paper): every success is a *conservative
lower bound*; untestable-target "failures" are unattributable.

## Protocol-sensitivity evidence (round 19)
- **Seed**: v3 x02 judged neutral (s42) vs **sad:80** (s1234), under both the 12- and 15-class
  judges.
- **Frame count**: 4/88 clip-configs flip between 4 and 8 frames (e.g. base x03, v2 x01).
- **Taxonomy**: verdicts flip when the judge's word list changes (12 vs 15), e.g. v3 x03 s42
  happy→neutral; x07 neutral→happy.
- **Steps**: an early 8-vs-16 step flip did *not* replicate at fixed frame count (that comparison
  was confounded by a simultaneous frame-count/taxonomy change); no clean step effect is claimed.

## Configurations used in the paper
| Protocol | Frames | Inference steps | Seeds | Used for |
|---|---|---|---|---|
| basic (legacy) | 4 | 8 | 42 | early screening (12-class) |
| **robust (standard)** | **8** | **16** | **42, 1234** | **all headline numbers (15-class primary)** |
| third-seed / sweep | 8 | 16 | 999; 777-781 | v3 x02/x07 rate estimates |

## Frame extraction
Uniformly sampled frames via ffmpeg (`/gluster_hdd_cv/user/jinzili/env/sglang310/bin/ffmpeg`),
saved as `f0000.jpg ...`; scripts `extract_lora_frames.py` (4 frames) and `extract8_frames.py`
(8 frames).

## Known limitations (disclosed in the paper)
The judge is a single, partially-blind VLM (3/21 on known emotions) with a strong neutral bias;
its verdicts are sensitive to seed, frame count, and taxonomy. We report scores, multi-seed
agreement, and both taxonomies to mitigate; a human validation protocol is provided in
`HUMAN_EVAL_PROTOCOL.md` (in this release).
