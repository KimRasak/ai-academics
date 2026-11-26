# Four-Judge Cross-Comparison (strong-accept round, round 2/256)

Judges (all open-weight Qwen VLMs, greedy, same 15-class protocol, same frames):
- **Qwen2-VL-7B** (paper's original judge; 21-portrait sanity)
- **Qwen2.5-VL-7B** (35-portrait sanity)
- **Qwen3-VL-32B** (35-portrait sanity)
- **Qwen3.8-27B** (35-portrait sanity; latest Qwen generation, native VLM, Qwen3_5ForConditionalGeneration)

## 1. Judge validity on known-emotion portraits (sanity)
Protocol: 17 portraits whose intended label is inside the 15-class taxonomy, identical set for all
four judges, same 15-class prompt/greedy/frames. Verdict parsed as the LAST `WORD:SCORE` in each raw
response (the model settles on its final answer at the end; earlier mentions are reasoning, not the
answer). All 17/17 parsed for every judge.
- **Strict** = exact 15-class match.
- **Lenient** = the paper's synonym rule: joyful/cheerful/happy treated as one happy family
  (strictly, the rule maps joyful and cheerful to their in-taxonomy synonym happy, so a judge
  answering any happy-family word for a happy-family portrait counts).

| Judge | strict | lenient | parsed | negative-emotion correct |
|---|---|---|---|---|
| Qwen2-VL-7B | 1/17 | 4/17 | 17/17 | 0/5 |
| Qwen2.5-VL-7B | 0/17 | 5/17 | 17/17 | 0/5 |
| Qwen3-VL-32B | 1/17 | 6/17 | 17/17 | 0/5 |
| Qwen3.8-27B | 0/17 | 7/17 | 17/17 | 0/5 |

Negative-emotion portraits in the 17: a02_angry, a03_anxious, a08_disgusted, a18_sad, c02_sad.
**No judge of any tested scale/generation (7B..32B, Qwen2-VL..Qwen3.8) recognizes ANY negative
emotion (0/5 for all four), and none passes even on happy-family (best 7/17=41%).** Evaluation is
the weak link at every scale.

For reference, Qwen2-VL's original 21-set report (superset incl. 4 non-taxonomy labels) was 3/21
strict, 7/21 lenient; on the shared 17-subset it is 1/17 strict, 4/17 lenient — the 17-subset is
the comparable number across all four judges.

## 2. Decisive x02 control pair (same seed s1234)
| Clip | Qwen2-VL | Qwen2.5-VL | Qwen3-VL-32B | Qwen3.8-27B |
|---|---|---|---|---|
| v3 x02 (neutral->sad) | **sad:80** | neutral:85 | worried:85 | **sad:82 / sad:85** (twice) |
| base x02 (control) | neutral:50 | neutral:95 | neutral:95 | neutral:78 |

Headline: base x02 is neutral under **all four** judges (unanimous). The v3 x02 clip is sad
under the paper's original judge **and** under the strongest available judge (Qwen3.8-27B,
two independent clip-presentations: sad:85, sad:82, with explicit visual reasoning).

## 3. Judges that "disagree" with sad on x02 also fail basic sad recognition
On the v2-raw clearly-crying portrait (sad_..._raw):
| Judge | v2raw_sad | (x02 verdict) |
|---|---|---|
| Qwen2-VL | **sad** | sad |
| Qwen2.5-VL | surprised:95 | neutral |
| Qwen3-VL-32B | surprised:95 | worried |
| Qwen3.8-27B | fearful:85 | sad |

Note: Qwen2.5-VL and Qwen3-VL-32B mislabel a crying portrait as "surprised"; their neutral/worried
x02 verdicts are consistent with a negative-emotion recognition gap. Qwen3.8-27B reads the crying
portrait as fearful but still reads x02 as sad (clip-specific judgment).

## 4. v3 tasks (8) under the two "capable-of-sad" / strongest judges
| task | target | Qwen3-VL-32B | Qwen3.8-27B |
|---|---|---|---|
| x01 neutral->joyful | joyful | cheerful:98 | cheerful:90 (happy-family) |
| x02 neutral->sad | sad | worried:85 | **sad:82** |
| x03 angry->happy | happy | cheerful:95 | cheerful:88 (happy-family) |
| x04 joyful->worried | worried | worried:95 | **worried:80** (untestable under 7B judges; now testable + success) |
| x05 amused->surprised | surprised | cheerful:95 | cheerful:90 |
| x06 anxious->relieved | relieved | happy:95 | happy:78 |
| x07 sad->cheerful | cheerful | cheerful:95 | happy:92 (happy-family) |
| x08 curious->angry | angry | neutral:95 | neutral:55 |

Qwen3.8-27B success (strict): x02=sad, x04=worried. Happy-family lenient: x01, x03, x07.
Qwen2.5-VL emitted no parseable task verdicts for v3 tasks (all verbosity-truncated) - see judge25 log.

## 5. Files
- judge38_results.jsonl (53 rows: Qwen3.8-27B, raw verdicts incl. reasoning)
- judge38_one.jsonl (2 rows: v2raw sad/joyful re-judged @1600 tok)
- judge3vl_results.jsonl (49 rows: Qwen3-VL-32B)
- judge_extra.jsonl (8 rows: Qwen2.5-VL + Qwen3-VL-32B on v2raw)
- judge25_results.jsonl (37 rows: Qwen2.5-VL)
- judge_sanity15.jsonl (21 rows: Qwen2-VL)
