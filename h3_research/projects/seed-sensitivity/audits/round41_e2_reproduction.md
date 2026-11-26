# Round 41：released MiniMax-H3 VT reproduction

## Protocol

- Downloaded all 150 `testmini` released MiniMax-H3 MP4s from `Sci-VBench/Sci-VBench-Videos`.
- Downloaded 150 MiniMax-H3 settings sidecars.
- Used pinned Sci-VBench evaluator commit `e8cb3988631d0c22d31c02fc5b1cc3693bfb7031`, isolated `/ssd2/jinzili/envs/scivbench_vt`, verified weights, physical GPU 3 exposed as `cuda`.
- Evaluated all six VT metrics.

## Result

150/150 videos completed with 0 errors in 3317.6 s. Local aggregate from the re-run: `VT=0.735706`.

Comparison with released `vt_scores.jsonl` on the same 150 `(video_id, model)` rows:

- Median absolute delta: subject consistency `1.45e-08`
- background consistency `2.44e-05`
- motion smoothness `1.82e-06`
- dynamic degree `0.0`
- aesthetic quality `6.17e-06`
- imaging quality `1.64e-05`
- VT `6.15e-06`
- Dynamic-degree exact match: `149/150 = 99.3333%`

Mean deltas are also near zero; only one DD cell differs. This passes the pre-registered E2 reproduction gate (continuous metrics far below 0.005; DD exact match above 95%).

The reproduction is valid at the evaluator level. It does not validate the MLLM judge or imply that VT is a sufficient scientific-quality measure.

Artifacts:

- `data/scivbench/released_vt/`
- `data/scivbench/released_settings/`
- `data/scivbench/released_vt_comparison.json`
- `compare_released_vt.py`
