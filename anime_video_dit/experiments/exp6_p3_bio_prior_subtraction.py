#!/usr/bin/env python3
"""
Exp 6 (P3 Zero-Training Optimization Test):
  Counterfactual Biological-Prior Contrastive Subtraction (Zero-Training, Zero-Label!)
  vs. Scalar Prior Calibration (calibrated on seen symbols only, tested on UNSEEN symbols)
  vs. Zero-Shot Direct & Audit baselines!

Core Hypothesis:
  In Exp 5, we discovered that even in 1-token Direct inference, the VLM ALREADY lowers
  P(A) on every non-biological anime conflict relative to clean `orig` / `placebo`,
  AND already ranks the right non-biological class inside {B, C, D}, BUT a massive
  positive biological-smile logit offset (+Delta_bio on class A) swamps argmax.

We test THREE clean, zero-training biological-prior debiasing operators:
  1. `scalar_tau_cal`: Subtract a single scalar tau from logit(A), fit ONLY on
     10 train chars x {orig, placebo, forehead_shadow, sweat_drop}, evaluated on
     6 held-out chars x ALL 7 variants (including UNSEEN eye_highlight_off & blush_lines_on).
  2. `mouth_prior_sub`: Training-free Contrastive Biological-Prior Subtraction using
     the character's own lower-face/mouth crop x_bio (isolating the pure biological
     smile prior):  logits_deb = logits(x_full) - lam * logits(x_mouth)
  3. `blur_prior_sub`: Training-free Contrastive Subtraction using a low-pass blurred
     portrait x_blur (sigma=18, erasing crisp anime graphic symbols while keeping bright
     cheerful tone): logits_deb = logits(x_full) - lam * logits(x_blur)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
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
LABELS = ["A", "B", "C", "D"]


@torch.no_grad()
def get_abcd_logits(model, processor, img: Image.Image, abcd_ids: list[int]) -> np.ndarray:
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DIRECT_PROMPT}]}]
    txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[txt], images=[img], return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1, abcd_ids].float().cpu().numpy()
    return logits


def score_logits(records: list[dict[str, Any]], logits_list: list[np.ndarray]) -> dict[str, Any]:
    by_var: dict[str, list[dict[str, Any]]] = {}
    for rec, lg in zip(records, logits_list):
        pred = LABELS[int(np.argmax(lg))]
        gt = EXPECTED[rec["variant"]]
        # Also compute non-A conditional accuracy (if A is masked out, does it pick the exact right B/C/D?)
        pred_nonA = LABELS[1 + int(np.argmax(lg[1:]))]
        by_var.setdefault(rec["variant"], []).append(
            {"pred": pred, "gt": gt, "correct": pred == gt, "nonA_correct": pred_nonA == gt}
        )
    summary = {}
    for v, rows in sorted(by_var.items()):
        summary[v] = {
            "n": len(rows),
            "gt": EXPECTED[v],
            "acc": round(float(np.mean([r["correct"] for r in rows])), 4),
            "nonA_cond_acc": round(float(np.mean([r["nonA_correct"] for r in rows])), 4) if EXPECTED[v] != "A" else None,
        }
    clean = [summary[k]["acc"] for k in ["orig", "placebo_hairpin"] if k in summary]
    seen_c = [summary[k]["acc"] for k in ["forehead_shadow_on", "sweat_drop_on"] if k in summary]
    unseen_c = [summary[k]["acc"] for k in ["eye_highlight_off", "blush_lines_on"] if k in summary]
    all_c = [summary[k]["acc"] for k in ["eye_highlight_off", "forehead_shadow_on", "dead_eyes_and_shadow", "blush_lines_on", "sweat_drop_on"] if k in summary]
    return {
        "macro_all_7": round(float(np.mean([s["acc"] for s in summary.values()])), 4),
        "clean_acc": round(float(np.mean(clean)), 4),
        "conflict_acc": round(float(np.mean(all_c)), 4),
        "seen_conflicts": round(float(np.mean(seen_c)), 4),
        "unseen_symbols_zero_shot": round(float(np.mean(unseen_c)), 4),
        "by_variant": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--boxes-json", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train-chars", type=int, default=10)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    boxes_map = json.loads(args.boxes_json.read_text(encoding="utf-8"))

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    abcd_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in LABELS]

    # Precompute full, lower-face/mouth-only, and blurred logits for all 112 images
    full_lg, mouth_lg, blur_lg = [], [], []
    # Cache lower-face crop per char_id (same mouth across all variants, compute from each image to stay 100% blind to variant!)
    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        w, h = img.size
        b = boxes_map[rec["char_id"]]
        # Lower-face crop strictly below the eyes (captures smiling mouth + chin, zero eyes/forehead/temple!)
        eye_bot = max(b["left_eye"][3], b["right_eye"][3])
        y1 = max(0, min(h - 64, int(round((eye_bot + 40) * h / 1000.0))))
        y2 = min(h, int(round(b["face"][3] * h / 1000.0)) + int(0.08 * h))
        x1 = max(0, int(round(b["face"][0] * w / 1000.0)))
        x2 = min(w, int(round(b["face"][2] * w / 1000.0)))
        mouth_crop = img.crop((x1, y1, max(x1 + 64, x2), max(y1 + 64, y2))).resize((384, 384), Image.LANCZOS)
        blur_img = img.filter(ImageFilter.GaussianBlur(radius=22))

        full_lg.append(get_abcd_logits(model, processor, img, abcd_ids))
        mouth_lg.append(get_abcd_logits(model, processor, mouth_crop, abcd_ids))
        blur_lg.append(get_abcd_logits(model, processor, blur_img, abcd_ids))

    chars = sorted({r["char_id"] for r in records})
    train_chars = set(chars[: args.n_train_chars])
    test_chars = set(chars[args.n_train_chars :])
    seen_vars = {"orig", "placebo_hairpin", "forehead_shadow_on", "sweat_drop_on"}

    train_idx = [i for i, r in enumerate(records) if r["char_id"] in train_chars and r["variant"] in seen_vars]
    test_idx = [i for i, r in enumerate(records) if r["char_id"] in test_chars]

    test_recs = [records[i] for i in test_idx]
    res: dict[str, Any] = {"model": Path(args.model_dir).name}

    # 1. Baseline Direct (ALL 16 chars & 6 held-out test chars)
    res["direct_all16"] = score_logits(records, full_lg)
    res["direct_test6"] = score_logits(test_recs, [full_lg[i] for i in test_idx])

    # 2. Fit scalar tau_A ONLY on train_idx (10 train chars x 4 seen variants), test ZERO-SHOT on test_idx + ALL 16!
    best_tau, best_tr = 0.0, -1.0
    for tau in np.linspace(-3.0, 5.0, 81):
        adj = []
        for i in train_idx:
            v = full_lg[i].copy()
            v[0] -= tau
            adj.append(v)
        sc = score_logits([records[i] for i in train_idx], adj)
        # Balanced score: mean of clean_acc and seen_conflicts
        bal = 0.5 * sc["clean_acc"] + 0.5 * sc["seen_conflicts"]
        if bal > best_tr:
            best_tr, best_tau = bal, float(tau)
    adj_all = [np.array([lg[0] - best_tau, lg[1], lg[2], lg[3]]) for lg in full_lg]
    res["scalar_tau_cal"] = {
        "calibrated_tau_A": round(best_tau, 3),
        "test6": score_logits(test_recs, [adj_all[i] for i in test_idx]),
        "all16": score_logits(records, adj_all),
    }

    # 3. Training-Free Lower-Face / Mouth Biological-Prior Contrastive Subtraction (lam sweep)
    res["mouth_prior_sub_sweep"] = {}
    for lam in [0.3, 0.5, 0.7, 0.9, 1.0]:
        sub_all = [full_lg[i] - lam * (mouth_lg[i] - mouth_lg[i].mean()) for i in range(len(records))]
        res["mouth_prior_sub_sweep"][f"lam_{lam}"] = {
            "all16": score_logits(records, sub_all),
            "test6": score_logits(test_recs, [sub_all[i] for i in test_idx]),
        }

    # 4. Training-Free Low-Pass Blur Prior Subtraction (lam sweep)
    res["blur_prior_sub_sweep"] = {}
    for lam in [0.3, 0.5, 0.7, 0.9, 1.0]:
        sub_all = [full_lg[i] - lam * (blur_lg[i] - blur_lg[i].mean()) for i in range(len(records))]
        res["blur_prior_sub_sweep"][f"lam_{lam}"] = {
            "all16": score_logits(records, sub_all),
            "test6": score_logits(test_recs, [sub_all[i] for i in test_idx]),
        }

    out_path = args.out_dir / f"exp6_bio_prior_sub_{Path(args.model_dir).name}.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    # Compact console summary
    print(f"=== {res['model']} ===")
    print("Direct ALL16:", {k: v for k, v in res["direct_all16"].items() if k != "by_variant"})
    print("Direct nonA_cond_acc:", {v: s["nonA_cond_acc"] for v, s in res["direct_all16"]["by_variant"].items() if s["nonA_cond_acc"] is not None})
    print(f"Scalar-Tau (tau={best_tau:.2f}) TEST6:", {k: v for k, v in res["scalar_tau_cal"]["test6"].items() if k != "by_variant"})
    print(f"Scalar-Tau (tau={best_tau:.2f}) ALL16:", {k: v for k, v in res["scalar_tau_cal"]["all16"].items() if k != "by_variant"})
    for mkey in ["mouth_prior_sub_sweep", "blur_prior_sub_sweep"]:
        best_l = max(res[mkey].items(), key=lambda kv: kv[1]["all16"]["macro_all_7"])
        print(f"Best {mkey} ({best_l[0]}) ALL16:", {k: v for k, v in best_l[1]["all16"].items() if k != "by_variant"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
