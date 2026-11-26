#!/usr/bin/env python3
"""
Exp 5C (Exact 1-Forward-Pass Logit Scaling Check across 7B vs 32B VLMs):
  Evaluates exact softmax probabilities [P(A), P(B), P(C), P(D)] for BOTH
  Direct Prompt AND 1-Token Audit Prompt across ALL 16 full-color TV-anime
  characters (112 images) with ZERO generation/parser ambiguity!

  Directly measures whether scaling from Qwen2.5-VL-7B -> Qwen3-VL-32B solves
  Biological-Override Binding Failure or whether P(A) biological override persists
  at 32B scale!
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

DIRECT_PROMPT = (
    "Look at this 2D anime character portrait and judge their TRUE emotional / psychological state.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "Answer with ONLY a single letter: A, B, C, or D."
)

AUDIT_PROMPT = (
    "Inspect this 2D anime portrait carefully for ANY non-biological anime graphic affect symbols "
    "(such as unusual dark shadows, missing/hollow eye highlights, cheek blush marks, or floating sweat drops), "
    "ignoring ordinary hair accessories. In 2D anime, non-biological graphic cues OVERRIDE a smiling mouth.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "Answer with ONLY a single letter: A, B, C, or D."
)

EXPECTED = {
    "orig": "A",
    "placebo_hairpin": "A",
    "eye_highlight_off": "B",
    "forehead_shadow_on": "B",
    "dead_eyes_and_shadow": "B",
    "blush_lines_on": "C",
    "sweat_drop_on": "D",
}
LABELS = ["A", "B", "C", "D"]


@torch.no_grad()
def run_eval(model, processor, records: list[dict[str, Any]], prompt: str, abcd_ids: list[int]) -> dict[str, Any]:
    by_var: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[txt], images=[img], return_tensors="pt").to(model.device)
        logits = model(**inputs).logits[0, -1, abcd_ids].float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = LABELS[int(np.argmax(probs))]
        gt = EXPECTED[rec["variant"]]
        by_var.setdefault(rec["variant"], []).append(
            {
                "char_id": rec["char_id"],
                "pred": pred,
                "gt": gt,
                "correct": pred == gt,
                "p_A": float(probs[0]),
                "p_gt": float(probs[LABELS.index(gt)]),
            }
        )

    summary: dict[str, Any] = {}
    for v, rows in sorted(by_var.items()):
        summary[v] = {
            "n": len(rows),
            "gt": EXPECTED[v],
            "acc": round(float(np.mean([r["correct"] for r in rows])), 4),
            "pred_A_rate": round(float(np.mean([r["pred"] == "A" for r in rows])), 4),
            "mean_p_A": round(float(np.mean([r["p_A"] for r in rows])), 4),
            "mean_p_gt": round(float(np.mean([r["p_gt"] for r in rows])), 4),
        }
    conflicts = [summary[k]["acc"] for k in ["eye_highlight_off", "forehead_shadow_on", "dead_eyes_and_shadow", "blush_lines_on", "sweat_drop_on"]]
    clean = [summary[k]["acc"] for k in ["orig", "placebo_hairpin"]]
    return {
        "macro_all_7": round(float(np.mean([s["acc"] for s in summary.values()])), 4),
        "clean_acc": round(float(np.mean(clean)), 4),
        "conflict_acc": round(float(np.mean(conflicts)), 4),
        "by_variant": summary,
    }


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
    abcd_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in LABELS]

    res = {
        "model": Path(args.model_dir).name,
        "direct": run_eval(model, processor, records, DIRECT_PROMPT, abcd_ids),
        "audit_1tok": run_eval(model, processor, records, AUDIT_PROMPT, abcd_ids),
    }
    out_path = args.out_dir / f"logit_scaling_{Path(args.model_dir).name}.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
