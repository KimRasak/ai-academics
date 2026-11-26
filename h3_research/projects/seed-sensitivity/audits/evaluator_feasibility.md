# Sci-VBench evaluator feasibility audit

日期：2026-09-08

## Judge

- The local Sci-VBench judge CLI imports and exposes the expected backend options.
- `python -m scivbench_eval.judge.run_scoring --self-test` passed: offline reasoning extraction checks passed for 3 response shapes.
- The judge is not a local standalone checkpoint. It requires an OpenAI-compatible vLLM endpoint (or Gemini API), a served multimodal judge model, and the released rubric file.
- No judge endpoint was started and no paid/API request was made in this audit. Therefore the 32-case pilot has not received PG/SCC/SC/LPF judge scores.

## VT

The code exposes the six metrics and deterministic resumable runner. `python -m scivbench_eval.vt.weights --check --commands` reports all required checkpoints missing in the local weights root:

- DINO ViT-B/16
- OpenAI CLIP ViT-B/32 and ViT-L/14
- LAION aesthetic head
- AMT-S
- RAFT Things
- MUSIQ SPAQ
- DINO repository source

A bounded remote search did not find these assets in the searched `/ssd2/jinzili` and gluster paths. No weight download was started because the audit's purpose was feasibility, not an untracked large download. Running VT now would fail at checkpoint setup, so no VT score is claimed.

## Decision

Generation Gate passes: 32/32 raw-prompt H3-Base pilot outputs decode and match the media envelope. Evaluator Gate is not yet passed. Do not interpret released MiniMax-H3 VT/judge aggregates as scores of our new outputs. The next reproducible options are:

1. provision/cache the exact VT assets and run a one-video metric smoke test before all 32;
2. obtain an approved local multimodal judge endpoint and score a 1-video/1-dimension self-test before scaling;
3. if neither is available, use the generation audit plus a separately labeled human/MLLM protocol rather than silently substituting metrics.

No training is justified by the current evidence.

## Compute availability follow-up

After pilot completion, 034's H3 services used only GPUs 0–2 (about 2.8 GiB each idle/warm memory in the service snapshot); GPUs 3–7 were effectively free. Remote `/ssd2` has about 5.6 TB free and the user's allocation was about 6.8 GB. The existing `sglang310` environment did not expose a usable VT stack: `vbench`, `clip`, and `pyiqa` were absent, and probing torch/torchvision was inconclusive due to slow import/remote connection termination. Thus the next action is environment provisioning plus a bounded one-video VT smoke test, not leaving available GPU unused and not launching a 32-video run blindly.
