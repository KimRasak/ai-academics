# EmoH3 — Release Upload Instructions (checkpoint + assets)

The paper's "checkpoints are released" claim is backed by files currently on the training node.
This doc gives the exact steps to make them public (HF Hub for model weights, GitHub for the paper).

## What to upload (all verified to exist)
| Asset | Location (cluster) | Size |
|---|---|---|
| **LoRA v3 (best)** | `/gluster_hdd_cv/user/jinzili/tmp/pilot/runs/lora3/checkpoint-900.pt` | 43 MB |
| LoRA v4 (768² ablation) | `/gluster_hdd_cv/user/jinzili/tmp/pilot/runs/lora4/checkpoint-900.pt` | 43 MB |
| Judge logs (16 files) | `release/judge_logs/` | ~250 KB (incl. Qwen3.8-27B, Qwen3-VL-32B full logs) |
| Verdict matrix | `release/DEFINITIVE_RESULTS.md` | 7 KB |
| Paper (PDF + source) | `release/paper_emoh3.*`, 2 figures | ~3 MB |
| Human-eval protocol | `release/HUMAN_EVAL_PROTOCOL.md` | — |

Checkpoint format: `{"model": {<108 LoRA tensors>}, "step": 900}`; rank 16 / alpha 32, targets =
attention projections (`to_qkv`, `to_out.0`, `linear_1`, `linear_2`, `time_embedder`).

## Steps

### 1. Hugging Face repo (weights)
```bash
# from the training node (has the files + huggingface_hub)
pip install -U huggingface_hub
huggingface-cli login   # paste write token
# create repo: huggingface.co/new -> "EmoH3-RefCond-LoRA-v3"
cd /gluster_hdd_cv/user/jinzili/tmp/pilot/runs
huggingface-cli upload <your-hf-user>/EmoH3-RefCond-LoRA-v3 \
    lora3/checkpoint-900.pt lora4/checkpoint-900.pt \
    --include "*.pt"
```
Then add a `README.md` in the repo: model card (base = MiniMax H3 33B, Ref2VA), LoRA config,
training data summary (22 clips), inference snippet, license (follow MiniMax H3's license for
derived models), and this checklist URL.

### 2. GitHub repo (paper + data) — anonymous-submission note
For the **anonymous** workshop submission do NOT link a repo that de-anonymizes you. Options:
- Use the venue's anonymous submission channel (OpenReview/CMT anonymous link) for the release
  files as supplementary material.
- For the camera-ready, push `release/` to a public repo with:
```bash
git init && git add release/ && git commit -m "EmoH3 release"
gh repo create EmoH3 --public --source=. --push
```

### 3. Verify after upload
- Download the two `.pt` from HF and `python -c "import torch; d=torch.load(f,map_location='cpu'); print(len(d['model']), d['step'])"` → expect `108 900`.
- Recompute one judge verdict from `release/judge_logs/` to confirm reproducibility.

## License note
MiniMax H3 is a permissively-licensed open model; check the current license at
https://www.minimax.io/news/minimax-h3-open-source before public release of derived weights. The
training/eval data are generated outputs of the model itself (no external copyrighted assets).
