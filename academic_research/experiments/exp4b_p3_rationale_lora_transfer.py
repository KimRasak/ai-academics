#!/usr/bin/env python3
"""
Exp 4B (P3 Principled Optimization Test): Audit-Then-Bind Rationale LoRA
vs. Direct-Letter LoRA (Exp 4) on Strict Leave-Symbols-Out Zero-Shot Transfer!

Why Exp 4B exists (directly driven by Exp 4's empirical discovery):
  - In Exp 4, standard direct-letter LoRA (image -> "A/B/C/D") trained on
    {orig, placebo_hairpin, forehead_shadow_on, sweat_drop_on} jumped
    0% -> 100% on seen `forehead_shadow_on` and 0% -> 66.7% on seen `sweat_drop_on`,
    BUT FAILED zero-shot transfer to UNSEEN `eye_highlight_off` (stuck at 16.7%)
    and SUPPRESSED UNSEEN `blush_lines_on` (33.3% -> 0.0%) via a closed-set shortcut!

Hypothesis tested here:
  Training the exact same rank-8 LLM-only LoRA on the EXACT SAME 4 seen variants
  using an open-vocabulary 2-step Audit-Then-Bind rationale (NEVER mentioning
  eye_highlight_off, blush_lines_on, or label C during training!) teaches the
  general visual-override routing habit instead of a closed-set label lookup,
  unlocking ZERO-SHOT TRANSFER to completely unseen anime affect symbols!
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


AUDIT_PROMPT = (
    "Inspect this 2D anime portrait carefully for ANY non-biological anime graphic affect symbols "
    "(such as unusual dark shadows, missing/hollow eye highlights, cheek blush marks, or floating sweat drops), "
    "ignoring ordinary hair accessories.\n"
    "Remember: in 2D anime, non-biological graphic affect symbols OVERRIDE a smiling mouth.\n"
    "First briefly state what graphic affect symbol (if any) you see, then conclude with `FINAL: X` where X is:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating"
)

# ONLY 4 seen variants have training rationales; eye_highlight_off (B) and blush_lines_on (C) are 100% HELD OUT!
TRAIN_RATIONALES = {
    "orig": (
        "Graphic check: No non-biological anime affect symbol is present; the face is evenly lit and the eyes look bright. "
        "Binding: With no overriding graphic cue, the smiling mouth indicates genuine cheerfulness.\nFINAL: A"
    ),
    "placebo_hairpin": (
        "Graphic check: Only a small decorative hair clip is on the hair; no anime affect symbol is present on the face or eyes. "
        "Binding: Ignore ordinary accessories and follow the smiling mouth.\nFINAL: A"
    ),
    "forehead_shadow_on": (
        "Graphic check: A heavy dark anime dread shadow with vertical gloom lines covers the upper forehead and eyes. "
        "Binding: This ominous non-biological graphic cue overrides the smiling mouth.\nFINAL: B"
    ),
    "sweat_drop_on": (
        "Graphic check: A stylized cyan-white cartoon anime sweat drop floats beside the character's temple. "
        "Binding: This non-biological anxiety cue overrides the smiling mouth.\nFINAL: D"
    ),
}

EXPECTED = {
    "orig": "A",
    "placebo_hairpin": "A",
    "eye_highlight_off": "B",
    "forehead_shadow_on": "B",
    "dead_eyes_and_shadow": "B",
    "blush_lines_on": "C",
    "sweat_drop_on": "D",
}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.scaling = alpha / float(r)
        self.lora_a = nn.Parameter(torch.empty((r, in_f), dtype=base.weight.dtype, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros((out_f, r), dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + ((x @ self.lora_a.t()) @ self.lora_b.t()) * self.scaling


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    for name, module in list(model.named_modules()):
        if "visual" in name:
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in {"q_proj", "v_proj", "o_proj"}:
                wrapped = LoRALinear(child, r=r, alpha=alpha)
                setattr(module, child_name, wrapped)
                trainable.extend([wrapped.lora_a, wrapped.lora_b])
    return trainable


def _extract_final(raw: str) -> str:
    m = re.search(r"FINAL\s*:\s*([ABCD])", raw.upper())
    if m:
        return m.group(1)
    ms = re.findall(r"\b([ABCD])\b", raw.upper())
    return ms[-1] if ms else "UNK"


def eval_audit(model, processor, records: list[dict[str, Any]]) -> dict[str, Any]:
    model.eval()
    by_var: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": AUDIT_PROMPT}]}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[txt], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        raw = processor.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        pred = _extract_final(raw)
        gt = EXPECTED[rec["variant"]]
        by_var.setdefault(rec["variant"], []).append({"char_id": rec["char_id"], "pred": pred, "gt": gt, "correct": pred == gt, "raw": raw})

    summary = {}
    for v, rows in sorted(by_var.items()):
        summary[v] = {
            "n": len(rows),
            "gt": EXPECTED[v],
            "acc": round(float(np.mean([r["correct"] for r in rows])), 4),
            "pred_A_rate": round(float(np.mean([r["pred"] == "A" for r in rows])), 4),
            "preds": [r["pred"] for r in rows],
            "sample_rationale": rows[0]["raw"],
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train-chars", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_recs = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    chars = sorted({r["char_id"] for r in all_recs})
    train_chars = set(chars[: args.n_train_chars])
    test_chars = set(chars[args.n_train_chars :])

    train_recs = [r for r in all_recs if r["char_id"] in train_chars and r["variant"] in TRAIN_RATIONALES]
    test_recs = [r for r in all_recs if r["char_id"] in test_chars]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")

    print("=== BEFORE AUDIT-THEN-BIND LORA (ZERO-SHOT AUDIT PROMPT ON HELD-OUT CHARS) ===", flush=True)
    before = eval_audit(model, processor, test_recs)
    print(json.dumps(before, indent=2, ensure_ascii=False), flush=True)

    trainable = inject_lora(model, r=args.rank, alpha=args.rank * 2.0)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    model.train()

    for ep in range(args.epochs):
        random.shuffle(train_recs)
        losses = []
        for rec in train_recs:
            img = Image.open(rec["image_path"]).convert("RGB")
            target_txt = TRAIN_RATIONALES[rec["variant"]]
            msgs_p = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": AUDIT_PROMPT}]}]
            msgs_f = msgs_p + [{"role": "assistant", "content": [{"type": "text", "text": target_txt}]}]
            txt_p = processor.apply_chat_template(msgs_p, tokenize=False, add_generation_prompt=True)
            txt_f = processor.apply_chat_template(msgs_f, tokenize=False, add_generation_prompt=False)
            in_p = processor(text=[txt_p], images=[img], return_tensors="pt")
            in_f = processor(text=[txt_f], images=[img], return_tensors="pt").to(model.device)
            labels = in_f["input_ids"].clone()
            labels[:, : in_p["input_ids"].shape[1]] = -100
            optimizer.zero_grad()
            loss = model(**in_f, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"Epoch {ep + 1}/{args.epochs} mean_loss={np.mean(losses):.4f}", flush=True)

    print("=== AFTER AUDIT-THEN-BIND LORA (HELD-OUT CHARS + UNSEEN SYMBOLS) ===", flush=True)
    after = eval_audit(model, processor, test_recs)
    print(json.dumps(after, indent=2, ensure_ascii=False), flush=True)

    out_path = args.out_dir / "p3_audit_rationale_lora_results.json"
    out_path.write_text(json.dumps({"before_audit_lora": before, "after_audit_lora": after}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
