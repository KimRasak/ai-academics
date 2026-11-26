# Round 35：Sci-VBench VT full pilot

## Run

- Input: 32 frozen H3-Base raw-prompt outputs (`data/scivbench/pilot_full/`)
- Remote environment: `/ssd2/jinzili/envs/scivbench_vt`
- Weights: `/ssd2/jinzili/scivbench_vt_weights`, all required assets check OK
- GPU: physical GPU 3, exposed inside process as `cuda`
- Metrics: all six Sci-VBench VT metrics
- No H3 generation was changed; no training was run

## Result

The resumable runner processed 32/32 videos with 0 errors in 351.1 s. Every video received all six fields and a composite VT score.

Aggregate (`data/scivbench/vt_full/h3_base/aggregate.json`):

| Metric | Mean |
|---|---:|
| Subject consistency | 0.922163813 |
| Background consistency | 0.950839151 |
| Motion smoothness | 0.991714066 |
| Dynamic degree | 0.593750000 |
| Aesthetic quality | 0.477314484 |
| Imaging quality | 0.658284099 |
| **VT** | **0.765677602** |

The summary reports `32/32` complete and VT `0.765678`.

## Interpretation

This is a fresh local VT evaluation of the 32 new H3-Base pilot outputs, using the fixed Sci-VBench implementation and its official upstream assets. It is not the published 150-video MiniMax-H3 aggregate and does not reproduce the MLLM judge dimensions PG/SCC/SC/LPF. The high motion-smoothness value and binary dynamic-degree distribution should not be read as human quality claims; the project already documents known limitations of such metrics.
