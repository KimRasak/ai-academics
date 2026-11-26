#!/usr/bin/env python3
"""
Exp 4 (P3 Optimization & Mechanistic Transfer Test):
  Pure-PyTorch Rank-8 LoRA Fine-Tuning on Qwen2.5-VL-7B-Instruct to test whether
  Biological-Override Binding Failure can be fixed AND whether the fix generalizes
  ZERO-SHOT to COMPLETELY HELD-OUT non-biological anime affect symbols!

Experimental Design (Zero Leakage, Strict OOD Generalization):
  - Characters split:
      * TRAIN chars: first 10 characters (e.g. fc_00..fc_09)
      * TEST  chars: last  6 characters  (e.g. fc_10..fc_15, unseen faces/hair/colors)
  - Symbols split (Leave-Symbols-Out Transfer!):
      * TRAIN symbols: `orig` (A), `placebo_hairpin` (A), `forehead_shadow_on` (B), `sweat_drop_on` (D)
      * HELD-OUT UNSEEN symbols: `eye_highlight_off` (B), `blush_lines_on` (C), `dead_eyes_and_shadow` (B)
  - Scientific Question:
      If we train a tiny LoRA ONLY on {forehead_shadow, sweat_drop, orig, placebo_hairpin}
      so the VLM learns "non-biological anime graphic cues override a smiling mouth",
      does accuracy on COMPLETELY UNSEEN `eye_highlight_off` and `blush_lines_on` jump
      on held-out characters — while keeping `orig` and `placebo_hairpin` at ~100% A?

Zero external deps beyond torch + transformers (pure-PyTorch LoRA injection!).
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


DIRECT_PROMPT = (
    "Look at this 2D anime character portrait and judge their TRUE emotional / psychological state.\n"
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


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        dtype = base.weight.dtype
        device = base.weight.device
        self.scaling = alpha / float(r)
        self.lora_a = nn.Parameter(torch.empty((r, in_f), dtype=dtype, device=device))
        self.lora_b = nn.Parameter(torch.zeros((out_f, r), dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (x @ self.lora_a.t()) @ self.lora_b.t()
        return out + delta * self.scaling


def inject_lora(model: nn.Module, target_suffixes: tuple[str, ...] = ("q_proj", "v_proj", "o_proj"), r: int = 8, alpha: float = 16.0) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    # Only inject into language_model layers (keeps vision backbone frozen for clean binding test!)
    for name, module in list(model.named_modules()):
        if "visual" in name:
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(child_name == s for s in target_suffixes):
                wrapped = LoRALinear(child, r=r, alpha=alpha)
                setattr(module, child_name, wrapped)
                trainable.extend([wrapped.lora_a, wrapped.lora_b])
    return trainable


def eval_split(model, processor, records: list[dict[str, Any]]) -> dict[str, Any]:
    model.eval()
    by_var: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DIRECT_PROMPT}]}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[txt], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        raw = processor.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        m = re.search(r"\b([ABCD])\b", raw.upper())
        pred = m.group(1) if m else "UNK"
        gt = EXPECTED[rec["variant"]]
        by_var.setdefault(rec["variant"], []).append({"pred": pred, "gt": gt, "correct": pred == gt, "raw": raw})

    summary = {}
    for v, rows in sorted(by_var.items()):
        summary[v] = {
            "n": len(rows),
            "gt": EXPECTED[v],
            "acc": round(float(np.mean([r["correct"] for r in rows])), 4),
            "pred_A_rate": round(float(np.mean([r["pred"] == "A" for r in rows])), 4),
            "preds": [r["pred"] for r in rows],
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train-chars", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1.5e-4)
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

    # Leave-Symbols-Out split: train ONLY on orig, placebo_hairpin, forehead_shadow_on, sweat_drop_on
    # Hold out COMPLETELY UNSEEN: eye_highlight_off, blush_lines_on, dead_eyes_and_shadow
    train_variants = {"orig", "placebo_hairpin", "forehead_shadow_on", "sweat_drop_on"}
    train_recs = [r for r in all_recs if r["char_id"] in train_chars and r["variant"] in train_variants]
    test_recs = [r for r in all_recs if r["char_id"] in test_chars]
    print(f"Split: {len(train_chars)} train chars ({len(train_recs)} ex) | {len(test_chars)} held-out test chars ({len(test_recs)} ex)", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")

    print("=== BEFORE LORA (ZERO-SHOT ON HELD-OUT CHARS) ===", flush=True)
    before = eval_split(model, processor, test_recs)
    print(json.dumps(before, indent=2), flush=True)

    # Inject LoRA into LLM attention projections only
    trainable = inject_lora(model, r=args.rank, alpha=args.rank * 2.0)
    n_params = sum(p.numel() for p in trainable)
    print(f"Injected pure-PyTorch LoRA (r={args.rank}): {n_params:,} trainable params", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    model.train()

    for ep in range(args.epochs):
        random.shuffle(train_recs)
        losses = []
        for rec in train_recs:
            img = Image.open(rec["image_path"]).convert("RGB")
            gt = EXPECTED[rec["variant"]]
            msgs_prompt = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DIRECT_PROMPT}]}]
            msgs_full = msgs_prompt + [{"role": "assistant", "content": [{"type": "text", "text": gt}]}]

            txt_prompt = processor.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)
            txt_full = processor.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)

            in_p = processor(text=[txt_prompt], images=[img], return_tensors="pt")
            in_f = processor(text=[txt_full], images=[img], return_tensors="pt").to(model.device)
            prompt_len = in_p["input_ids"].shape[1]

            labels = in_f["input_ids"].clone()
            labels[:, :prompt_len] = -100

            optimizer.zero_grad()
            out = model(**in_f, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"Epoch {ep + 1}/{args.epochs} mean_loss={np.mean(losses):.4f}", flush=True)

    print("=== AFTER LEAVE-SYMBOLS-OUT LORA (HELD-OUT CHARS + UNSEEN SYMBOLS) ===", flush=True)
    after = eval_split(model, processor, test_recs)
    print(json.dumps(after, indent=2), flush=True)

    res = {
        "model_dir": args.model_dir,
        "train_chars": sorted(train_chars),
        "test_chars": sorted(test_chars),
        "train_variants": sorted(train_variants),
        "held_out_unseen_variants": ["eye_highlight_off", "blush_lines_on", "dead_eyes_and_shadow"],
        "before_lora": before,
        "after_lora": after,
    }
    out_path = args.out_dir / "p3_leave_symbols_out_lora_results.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved LoRA transfer results -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
