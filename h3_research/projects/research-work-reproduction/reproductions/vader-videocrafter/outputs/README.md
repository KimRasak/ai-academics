# Outputs

The first deterministic checkpoint audit was run on 2026-09-16.

Remote roots:

- base: `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vader/outputs/base-smoke-seed200`;
- PickScore LoRA:
  `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vader/outputs/pickscore-smoke-seed200`;
- decoded frames, contact sheets and evaluator logs:
  `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vader/outputs/audit-seed200`.
- balanced three-prompt, nine-seed audit:
  `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vader/outputs/pickscore-paired-seed3-n9`.
- VBench manifest-wrapper paired smoke:
  `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vader/outputs/vbench-standard-smoke-v2`;
- full 72-prompt VBench Dynamic Degree/Motion Smoothness generation:
  `/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vader/outputs/vbench-standard-dynamic-v1`.

Both MP4 files are 512×320, 24 frames, 10 fps and 2.4 seconds. Their hashes,
logs, evaluator revisions, results and failed attempts are enumerated in
`run_manifest.jsonl`.

`media_manifest.jsonl` records every balanced-audit MP4 by condition, seed,
prompt record, path and SHA256. `balanced_results.json` records per-video and
aggregate PickScore, HPSv2.1, Aesthetic, VBench Dynamic Degree and VBench
Motion Smoothness results.

Every run must retain:

- the frozen prompt manifest;
- all generation attempts, including failures;
- stdout/stderr and environment lock;
- MP4 files and media metadata;
- per-sample reward values before aggregation;
- checkpoint and evaluator hashes.

Smoke outputs must be labelled as smoke tests and cannot be used as paper
metric reproduction.

The full VBench run uses all 72 prompts shared by Dynamic Degree and Motion
Smoothness, five samples per prompt and paired base/LoRA seeds. It produces 360
videos per condition. Each of four shards retains its PID, log, frozen prompt
sequence, seeds and media hashes.
