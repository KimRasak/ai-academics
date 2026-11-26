# Round 31：Sci-VBench raw-prompt pilot generation audit

## Frozen protocol

- Input: `data/scivbench/pilot16.jsonl` (16 records, four domains, stratified prompt lengths)
- Prompt: exact `posted_prompt`
- Seeds: 0 and 1
- Variant: official H3-Base FL2VA service on `osa00-h100-034:30011`, no LoRA
- Target request: 768 short edge, 16:9, 5 seconds
- Total: 32 cases

## Generation result

All 32/32 cases completed. The runner persisted request, submit response, status history, final status, source record, MP4 and runner summary for every case. No failed case or runner error was observed.

## Media audit

All 32 outputs were audited with the remote official environment's:

```text
/gluster_hdd_cv/user/jinzili/env/sglang310/bin/ffprobe
```

Results:

- 32/32 ffprobe successful
- 32/32 duration 5.175 s
- 32/32 H.264 video, 1344×768, 24 FPS
- 32/32 AAC audio, 32 kHz, 2 channels
- 0 missing, 0 malformed, 0 decode-audit failures

Structured report: `data/scivbench/pilot_full/audit.json`. Per-case outputs and ffprobe files are under `data/scivbench/pilot_full/`.

## Interpretation boundary

This is a generation and media-integrity pass only. It does not reproduce Sci-VBench's PG/SCC/SC/LPF judge scores or VT metrics. No quality or failure-mode claim is made until the evaluator checkpoint/protocol is independently available and self-tested. The complete generation pass allows the project to proceed to evaluator feasibility and controlled baseline planning without starting training.
