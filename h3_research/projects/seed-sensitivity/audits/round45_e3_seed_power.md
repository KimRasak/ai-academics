# Round 45：E3 seed-power 结果

## 生成与 VT

- 16 frozen prompts
- seeds 0–5，共 96 videos
- seeds 2–5 的 64 条新视频：64/64，0 errors
- 新视频 VT：64/64，0 errors，481.3s
- 使用 E2 已验证的 evaluator/weights/GPU stack

## Seed-count mean / standard error

| k seeds | mean VT | across-prompt SD | SE (16 prompts) |
|---:|---:|---:|---:|
| 1 | 0.755956 | 0.084245 | 0.021061 |
| 2 | 0.765678 | 0.073114 | 0.018278 |
| 3 | 0.758174 | 0.070562 | 0.017640 |
| 4 | 0.753531 | 0.068678 | 0.017169 |
| 5 | 0.751112 | 0.067065 | 0.016766 |
| 6 | 0.751899 | 0.067211 | 0.016803 |

## Within-prompt instability

Mean within-prompt SD across six seeds:

- VT: 0.051963
- dynamic degree: 0.306386
- motion smoothness: 0.002525
- subject consistency: 0.033258
- background consistency: 0.023114
- aesthetic quality: 0.038437
- imaging quality: 0.049944

Across all within-prompt seed pairs, DD flip rate is `0.304167` (73/240 pairs = 30.42%). A DD change occurs for 10/16 prompts (62.5% any-flip rate). This exceeds E3 go threshold 25%. The prior 2-seed flip rate was 7/16 prompts; the six-seed pairwise statistic is a more complete summary but is not identical to prompt-level any-flip rate. The initial draft accidentally stated 146/480; the corrected denominator is 240 unique pairs for 6 seeds.

Single-seed MDD against the observed 0.0049 adjacent released-model gap is clearly exceeded by within-prompt VT variability (mean SD 0.051963). E3 Go condition is met: DD flip rate >=25% and single-seed MDD >0.0049.

## Interpretation

This is a seed-power/metric sensitivity result on 16 prompts, not a full benchmark re-run. The mean VT stabilizes somewhat with more seeds, but 6 seeds still leave substantial within-prompt uncertainty, dominated by thresholded DD. It supports reporting seed aggregation or confidence intervals when comparing close model scores. It does not establish DD is invalid; the human-score join and limitations remain relevant.


---

**NOTE (Round 70)**: 本审计的 6-seed 数字（73/240=30.42% 等）已被 12-seed 扩展（seeds 6–11）supersede；论文与后续审计以 `round70_12seed_extension.md` 的 12-seed 数字为准（290/1056=27.46%，SD 0.0509）。6-seed 保留为历史证据链。
