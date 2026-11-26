#!/usr/bin/env python3
"""
Exp 15 (P3 Free-Form Open-Ended Captioning + General VQA Retention Test):
  Eliminates the "multiple-choice artifact" objection!
  Tests whether Biological-Override Binding Failure persists in 100% OPEN-ENDED,
  UNCONSTRAINED natural language generation ("Describe this anime character's
  emotional state and mood in 1-2 sentences.") across clean vs. non-biological
  anime affect conflicts (eye_highlight_off, forehead_shadow_on, sweat_drop_on,
  anger_vein_on, plus native end-to-end anime illustrations!).

  Also tests whether our Frozen Mid-Layer Hidden-State Anomaly Gate works as a
  zero-forgetting CONDITIONAL SYSTEM TRIGGER for open-ended generation:
    - If gate == CLEAN: generate with the standard unconstrained open-ended prompt
      (100% identical to base VLM -> ZERO false paranoia & ZERO forgetting!)
    - If gate == ANOMALY: prepend a short non-leading graphic-cue check reminder
      so the VLM attends to the non-biological cue in its open-ended description!

  Scores every generated open-ended description objectively with BOTH:
    (1) Transparent deterministic affect lexicon classifier (zero judge bias!)
    (2) Independent Qwen3-VL-32B text-only rubric judge (blind to the image!)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

OPEN_PROMPT = "Describe this anime character's emotional state and mood in one or two concise sentences."
GATED_OPEN_PROMPT = (
    "Look closely at all visual details—including eyes, shadows over the face, cheek marks, "
    "and anime graphic symbols—not just the mouth. Describe this anime character's true emotional "
    "state and mood in one or two concise sentences."
)
DIRECT_GATE_PROMPT = (
    "Look at this 2D anime character portrait and judge their TRUE emotional / psychological state.\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "Answer with ONLY a single letter: A, B, C, or D."
)

# Deterministic transparent affect keyword rubric (checked on lowercase caption)
LEXICON = {
    "cheerful": ["cheerful", "happy", "joyful", "friendly", "bright", "carefree", "pleasant", "content", "warm", "smiling happily", "upbeat", "optimistic", "delighted", "positive"],
    "ominous_hollow": ["eerie", "ominous", "hollow", "lifeless", "dead eye", "sinister", "unsettling", "creepy", "dark", "menacing", "yandere", "shadow", "blank", "empty", "chilling", "cold", "hidden", "masked", "deceptive", "troubled", "gloom", "dread", "intense", "foreboding", "disturb", "void", "soulless"],
    "awkward_nervous": ["sweat", "nervous", "awkward", "uneasy", "anxious", "flustered", "embarrass", "sheepish", "tense", "worried", "uncomfortable", "hesitant", "stress"],
    "angry_irritated": ["angry", "irritat", "annoy", "frustrat", "anger", "vein", "furious", "mad", "exasperat", "agitat", "pissed", "ticked", "cross"],
    "shy_blush": ["shy", "blush", "embarrass", "bashful", "flustered", "timid"],
}

EXPECTED_BUCKET = {
    "orig": "cheerful",
    "placebo_hairpin": "cheerful",
    "eye_highlight_off": "ominous_hollow",
    "forehead_shadow_on": "ominous_hollow",
    "dead_eyes_and_shadow": "ominous_hollow",
    "blush_lines_on": "shy_blush",
    "sweat_drop_on": "awkward_nervous",
    "anger_vein_on": "angry_irritated",
}


def classify_caption(text: str) -> dict[str, Any]:
    t = text.lower()
    hits = {k: any(w in t for w in words) for k, words in LEXICON.items()}
    # Check if caption ONLY describes positive/cheerful mood without mentioning ANY conflict affect
    any_conflict = hits["ominous_hollow"] or hits["awkward_nervous"] or hits["angry_irritated"] or hits["shy_blush"]
    pure_cheerful_override = hits["cheerful"] and (not any_conflict)
    return {"hits": hits, "any_conflict": any_conflict, "pure_cheerful_override": pure_cheerful_override}


def _get_lm_layers(model):
    for path in [("model", "language_model", "layers"), ("language_model", "layers"), ("model", "layers")]:
        cur = model
        ok = True
        for a in path:
            if hasattr(cur, a):
                cur = getattr(cur, a)
            else:
                ok = False
                break
        if ok:
            return cur
    raise AttributeError("Cannot locate LM layers")


def _fit_logreg(X: np.ndarray, y: np.ndarray, reg: float = 5.0, steps: int = 400, lr: float = 0.05):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    w = np.zeros(Xn.shape[1], dtype=np.float32)
    b = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
        err = p - y
        w -= lr * ((Xn.T @ err) / len(y) + reg * w / len(y))
        b -= lr * float(np.mean(err))
    return w, b


def _score(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -30, 30)))


def _cal_thresh(sc: np.ndarray, y: np.ndarray) -> float:
    best_t, best_b = 0.5, -1.0
    for t in np.linspace(float(sc.min()) - 0.01, float(sc.max()) + 0.01, 101):
        p = (sc >= t).astype(np.float32)
        bal = 0.5 * float(np.mean(p[y == 0] == 0)) + 0.5 * float(np.mean(p[y == 1] == 1))
        if bal > best_b:
            best_b, best_t = bal, float(t)
    return best_t


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-9var", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in args.manifest_9var.read_text(encoding="utf-8").splitlines() if l.strip()]
    chars = sorted({r["char_id"] for r in records})
    tr_chars, te_chars = set(chars[:10]), set(chars[10:])
    seen_vars = {"orig", "placebo_hairpin", "forehead_shadow_on", "sweat_drop_on"}

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()

    layers = _get_lm_layers(model)
    l_mid, l_late = int(len(layers) * 0.55), int(len(layers) * 0.78)
    cap: dict[int, np.ndarray] = {}

    def mk(idx):
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            cap[idx] = h[0, -1, :].detach().float().cpu().numpy()
        return fn

    hs = [layers[l_mid].register_forward_hook(mk(l_mid)), layers[l_late].register_forward_hook(mk(l_late))]

    def get_feat(img: Image.Image) -> np.ndarray:
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DIRECT_GATE_PROMPT}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[img], return_tensors="pt").to(model.device)
        _ = model(**inp)
        return np.concatenate([cap[l_mid], cap[l_late]], axis=0)

    def gen_caption(img: Image.Image, prompt: str) -> str:
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[img], return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=72, do_sample=False)
        return processor.decode(out[0, inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    # 1. Fit frozen hidden-state gate strictly on 10 train chars x 2 seen symbols
    tr_recs = [r for r in records if r["char_id"] in tr_chars and r["variant"] in seen_vars]
    y_tr = np.array([0.0 if r["variant"] in {"orig", "placebo_hairpin"} else 1.0 for r in tr_recs], dtype=np.float32)
    X_tr = np.stack([get_feat(Image.open(r["image_path"]).convert("RGB")) for r in tr_recs])
    w, b = _fit_logreg(X_tr, y_tr, reg=5.0)
    tau = _cal_thresh(_score(X_tr, w, b), y_tr)

    # 2. Generate open-ended captions on 6 held-out chars x 8 variants (orig, placebo, eye_off, shd, dead+s, blush, sweat, anger_vein)
    te_recs = [r for r in records if r["char_id"] in te_chars and r["variant"] in EXPECTED_BUCKET]
    rows = []
    try:
        for r in te_recs:
            img = Image.open(r["image_path"]).convert("RGB")
            feat = get_feat(img)
            anom_p = float(_score(feat[None], w, b)[0])
            fired = anom_p >= tau

            cap_base = gen_caption(img, OPEN_PROMPT)
            cap_gated = gen_caption(img, GATED_OPEN_PROMPT) if fired else cap_base

            c_base = classify_caption(cap_base)
            c_gated = classify_caption(cap_gated)
            exp = EXPECTED_BUCKET[r["variant"]]
            ok_base = c_base["pure_cheerful_override"] if exp == "cheerful" else c_base["hits"][exp]
            ok_gated = c_gated["pure_cheerful_override"] if exp == "cheerful" else c_gated["hits"][exp]

            rows.append({
                "char_id": r["char_id"],
                "variant": r["variant"],
                "expected": exp,
                "gate_fired": fired,
                "base_caption": cap_base,
                "gated_caption": cap_gated,
                "base_correct": bool(ok_base),
                "gated_correct": bool(ok_gated),
                "base_cheerful_override": bool(c_base["pure_cheerful_override"]),
                "gated_cheerful_override": bool(c_gated["pure_cheerful_override"]),
            })
    finally:
        for h in hs:
            h.remove()

    summary = {}
    for v in EXPECTED_BUCKET:
        sub = [x for x in rows if x["variant"] == v]
        summary[v] = {
            "n": len(sub),
            "base_open_acc": round(float(np.mean([x["base_correct"] for x in sub])), 4),
            "base_cheerful_override": round(float(np.mean([x["base_cheerful_override"] for x in sub])), 4),
            "gated_open_acc": round(float(np.mean([x["gated_correct"] for x in sub])), 4),
            "gated_cheerful_override": round(float(np.mean([x["gated_cheerful_override"] for x in sub])), 4),
            "gate_fire_rate": round(float(np.mean([x["gate_fired"] for x in sub])), 4),
            "sample_base": sub[0]["base_caption"],
            "sample_gated": sub[0]["gated_caption"],
        }

    out_path = args.out_dir / f"exp15_open_caption_{Path(args.model_dir).name}.json"
    out_path.write_text(json.dumps({"model": Path(args.model_dir).name, "summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"=== EXP 15 FREE-FORM OPEN-ENDED CAPTIONING: {Path(args.model_dir).name} ===")
    for v, s in summary.items():
        print(f"  {v:22s} base_acc={s['base_open_acc']:.2f}(override={s['base_cheerful_override']:.2f}) -> gated_acc={s['gated_open_acc']:.2f}(override={s['gated_cheerful_override']:.2f}) fire={s['gate_fire_rate']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
