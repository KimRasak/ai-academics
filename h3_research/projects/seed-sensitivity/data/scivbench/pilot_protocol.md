# Sci-VBench H3-Base Pilot Protocol

## Goal

验证公开 MiniMax-H3 SCC≈1.63/5 现象能否在本地官方 H3-Base、原始 `posted_prompt` 和冻结随机种子下复现。此 pilot 不做方法训练，也不声称代表完整 benchmark。

## Frozen input

- Split file: `pilot16.jsonl`
- Source: Sci-VBench `testmini.jsonl`
- Selection: each of four domains, four prompt-length quantiles (short, lower-middle, upper-middle, long)
- SHA-256: `f5d45f56f6514d538c914defa51dbc80767d7b5f8f80a6eb0c8ab27cde81860f5`
- Prompt: exact `posted_prompt`; do not use Context-IR expansion for this test
- Seeds: 0, 1
- Target: short edge 768, 16:9, duration 5 s for the first smoke/pilot; if a prompt requires longer temporal reasoning, record the mismatch instead of silently changing it

## Generation arms

1. H3-Base FL2VA / T2VA, raw prompt, seed 0
2. H3-Base FL2VA / T2VA, raw prompt, seed 1
3. Optional diagnostic arm: same prompt through hosted Context-IR, then local Base, only after API prompt and cost are archived

Do not mix Ref2VA into this science-domain T2VA pilot. Keep model variant, checkpoint path, SGLang commit/environment and topology fixed.

## Required per-output evidence

- request JSON and exact source record ID
- submit response, all status polls, final status
- MP4 and SHA-256
- ffprobe JSON: duration, width/height, FPS, audio codec/sample rate/channels
- service log range and peak GPU memory
- model path, variant, command line, environment fingerprint
- generation wall time and inference time if returned

## Evaluation

### Automatic

For a first pilot, do not run a judge unless its checkpoint and protocol are available. Record:

- Sci-VBench dimensions: PG, SCC, SC, LPF only if using the official judge implementation and its exact judge model
- six VT metrics only if VBench weights are installed and self-test passes
- ffprobe and decode integrity for every output

### Human/MLLM scoring rule

Use one dimension per request, three rounds, exactly as Sci-VBench code specifies. Never report a single judge score as ground truth. Report per-video means, domain/length strata, seed variation, and a confidence interval or bootstrap interval.

## Go / No-Go

- **Go to expanded pilot:** all smoke outputs decode; both seeds complete; no infrastructure-specific failure; score schema valid.
- **No-Go on method:** if SCC differences are not stable across seeds or are explained by prompt-duration mismatch, do not create a method around SCC.
- **Go to Gate 4 candidate:** only if a failure survives fixed prompts, two seeds, four domains/length strata and an evaluator self-test. Then compare at least one strong baseline before proposing mitigation.
