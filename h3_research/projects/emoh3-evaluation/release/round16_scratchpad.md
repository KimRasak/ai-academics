# Round 16 — results scratchpad (fill from cluster when jobs finish)

## M1: x02 neutral→sad multi-seed sweep (16 steps, 8-frame judge)
Reference: a15_neutral_1537882489. Prompt: "She looks sad."
Effective seed = CLI --seed + index.

Known (rounds 7-11, 8-frame/16-step): v3 s42→neutral:100 ✗, s1234→**sad:80 ✓**, s999→neutral:80 ✗ (1/3);
base s42→neutral:50(8s) ✗, s1234→neutral:100 ✗ (0/2).

New (CLI --seed 777, indices 0-4 → eff 777,778,779,780,781):
| eff seed | base | v3 |
|---|---|---|
| 777 | ? | ? |
| 778 | ? | ? |
| 779 | ? | ? |
| 780 | ? | ? |
| 781 | ? | ? |

Combined rates: base ?/7, v3 ?/8.  → paper Finding A / Table 3 x02 cells.

## M4: reference-swap mechanism test (prompt FIXED "She looks sad.", char 1534125952)
Refs: angry/joyful/sad/surprised (v2 raw). eff seed = CLI + index (0-3).
| ref emotion | base s42 | base s1234 | v3 s42 | v3 s1234 |
|---|---|---|---|---|
| angry | ? | ? | ? | ? |
| joyful | ? | ? | ? | ? |
| sad | ? | ? | ? | ? |
| surprised | ? | ? | ? | ? |

Interpretation: base → judged emotion tracks REF (dominance); v3 → judged emotion tracks PROMPT (sad) = override.
Count how many refs each model "wins" (judged=sad) per seed.

## After data:
1. Update paper Finding A rate + Table 3 x02 (base 0/1→Y/7, v3 1/3→X/8).
2. Add M4 table + 3 sentences to ICASSP (Sec. Experiments, after Findings) — page budget ~0.3pp, trim intro/limitations if needed.
3. Copy new judge jsonls to release/judge_logs/; append to DEFINITIVE_RESULTS.md (sections I/J).
4. Rebuild all 5 papers + verify pages/leaks.
