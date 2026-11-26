#!/usr/bin/env python3
"""
Exp 7B: Fast 1-Forward-Pass Gate 1 (Perception) + Gate 2 (Text-Only Anime Rule Knowledge)
for Cross-Family VLM Verification (InternVL3-8B, LLaVA-OneVision-7B, Qwen2.5-VL-7B, Qwen3-VL-32B).

Uses exact YES/NO and A/B/C/D next-token logits (zero generation latency, zero parser noise!).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

GATE2_QS = {
    "eye_highlight_off": (
        "In 2D anime visual language, if a character has a smiling mouth BUT their white eye highlights/catchlights "
        "are completely removed ('dead eyes' / matte dark irises), what emotional state does this typically signify?\n"
        "A) Genuinely cheerful and carefree\nB) Emotionally hollow, shocked, sinister, or ominous (yandere)\n"
        "C) Shy or romantically embarrassed\nD) Awkwardly nervous or sweating\nAnswer with ONLY A, B, C, or D.",
        "B",
    ),
    "forehead_shadow_on": (
        "In 2D anime visual language, if a character has a smiling mouth BUT a heavy dark vertical shadow with gloom lines "
        "covers their forehead and eyes, what emotional state does this typically signify?\n"
        "A) Genuinely cheerful and carefree\nB) Ominous, menacing, grim, or suppressing dark intent\n"
        "C) Shy or romantically embarrassed\nD) Awkwardly nervous\nAnswer with ONLY A, B, C, or D.",
        "B",
    ),
    "blush_lines_on": (
        "In 2D anime visual language, if a character has diagonal pink-red hatch lines ('///') and soft pink blush under both eyes, "
        "what emotional state does this typically signify?\n"
        "A) Genuinely neutral/cheerful without shyness\nB) Ominous or menacing\n"
        "C) Shy, flustered, or embarrassed\nD) Angry\nAnswer with ONLY A, B, C, or D.",
        "C",
    ),
    "sweat_drop_on": (
        "In 2D anime visual language, if a character has a stylized cyan-white cartoon sweat drop floating beside their temple, "
        "what emotional state does this typically signify?\n"
        "A) Genuinely carefree and confident\nB) Ominous or sinister\n"
        "C) Shy or romantically in love\nD) Awkward, flustered, or nervously uneasy\nAnswer with ONLY A, B, C, or D.",
        "D",
    ),
}

GATE1_PROMPTS = {
    "eye_highlight_off": ("Look closely inside the character's colored irises. Are bright white specular eye highlights/catchlights clearly visible inside the eyes? Answer ONLY YES or NO.", False),
    "forehead_shadow_on": ("Is there a heavy dark ominous shadow cast across the character's forehead and upper eyes? Answer ONLY YES or NO.", True),
    "blush_lines_on": ("Are there stylized anime pink-red diagonal blush lines ('///') on the character's cheeks below the eyes? Answer ONLY YES or NO.", True),
    "sweat_drop_on": ("Is there a stylized cartoon anime sweat drop floating near the character's temple/head? Answer ONLY YES or NO.", True),
}
LABELS = ["A", "B", "C", "D"]


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    tok = processor.tokenizer
    abcd_ids = [tok.encode(ch, add_special_tokens=False)[-1] for ch in LABELS]
    yes_id = tok.encode("YES", add_special_tokens=False)[-1]
    no_id = tok.encode("NO", add_special_tokens=False)[-1]

    # Gate 2: Text-only knowledge
    gate2 = {}
    for k, (q, gt) in GATE2_QS.items():
        msgs = [{"role": "user", "content": [{"type": "text", "text": q}]}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[txt], return_tensors="pt").to(model.device)
        lg = model(**inputs).logits[0, -1, abcd_ids].float().cpu().numpy()
        pred = LABELS[int(np.argmax(lg))]
        gate2[k] = {"pred": pred, "gt": gt, "correct": pred == gt}

    # Gate 1: Perception on target variant vs clean orig (computes balanced TPR & TNR!)
    gate1 = {}
    orig_recs = [r for r in records if r["variant"] == "orig"]
    for v, (q, target_is_yes) in GATE1_PROMPTS.items():
        v_recs = [r for r in records if r["variant"] == v]
        def eval_yes(sub):
            ys = []
            for rec in sub:
                img = Image.open(rec["image_path"]).convert("RGB")
                msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
                txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[txt], images=[img], return_tensors="pt").to(model.device)
                lg = model(**inputs).logits[0, -1, [no_id, yes_id]].float().cpu().numpy()
                ys.append(bool(lg[1] > lg[0]))
            return ys

        yes_on_orig = eval_yes(orig_recs)
        yes_on_var = eval_yes(v_recs)
        if target_is_yes:
            orig_ok = float(np.mean([not x for x in yes_on_orig]))
            var_ok = float(np.mean(yes_on_var))
        else:
            orig_ok = float(np.mean(yes_on_orig))
            var_ok = float(np.mean([not x for x in yes_on_var]))
        gate1[v] = {
            "orig_acc": round(orig_ok, 4),
            "variant_acc": round(var_ok, 4),
            "balanced_acc": round(0.5 * (orig_ok + var_ok), 4),
        }

    res = {
        "model": Path(args.model_dir).name,
        "gate2_acc": round(float(np.mean([x["correct"] for x in gate2.values()])), 4),
        "gate2": gate2,
        "gate1": gate1,
    }
    out_path = args.out_dir / f"gates12_{Path(args.model_dir).name}.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
