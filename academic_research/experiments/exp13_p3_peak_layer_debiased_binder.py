#!/usr/bin/env python3
"""
Exp 13 (Mechanistically-Guided Peak-Layer Gate + Clean-Prior-Debiased Stage-2 Binder):

Directly combines TWO clean, zero-VLM-weight-update insights from Exp 9 & Exp 11:
  1. Stage 1 (Peak-Layer Pre-Erasure Anomaly Gate):
     Taps frozen VLM hidden states at the mechanistically optimal mid-to-late
     layer BEFORE late-layer biological-smile erasure (Exp 11):
       - InternVL3-8B / LLaVA-OV-7B: L15 (0.56 depth, where AUROC peaked at 0.903–1.000!)
       - Qwen2.5-VL-7B: L21 (0.78 depth)
       - Qwen3-VL-32B: L56 (0.89 depth, right before the L56->L63 -22% crash!)
     Trained ONLY on 10 synthetic train chars x {orig, placebo} vs {forehead_shadow, sweat_drop}
     (ZERO unseen symbols ever seen!).

  2. Stage 2 (Clean-Baseline Prior-Debiased Non-A Binder — 1-line option-bias cancellation!):
     Computes mean_clean_logits = E_{x in clean_train}[logits(x)] using ONLY the 10 clean
     training faces (ZERO conflict labels!). At test time, whenever Stage 1 fires:
       delta_k(x) = logits_k(x) - mean_clean_logits_k   for k in {B, C, D, E}
     This 1-line subtraction cancels option-order / recency bias (e.g. Qwen2.5-VL's
     Option-E bias and InternVL3's label prior) at ZERO extra inference cost!

Evaluates across ALL 4 SOTA VLM families on BOTH:
  (A) 9-Variant Counterfactual Test Split (6 held-out chars x 9 variants = 54 imgs,
      including 4 COMPLETELY UNSEEN cues: eye_off, blush, anger_vein, bio_brows!)
  (B) 100% Native End-to-End Animagine-XL-4.0 Anime Art (32 imgs, ZERO overlays!)
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

LABELS5 = ["A", "B", "C", "D", "E"]
EXPECTED9 = {
    "orig": "A",
    "placebo_hairpin": "A",
    "eye_highlight_off": "B",
    "forehead_shadow_on": "B",
    "dead_eyes_and_shadow": "B",
    "blush_lines_on": "C",
    "sweat_drop_on": "D",
    "bio_angry_brows": "E",
    "anger_vein_on": "E",
}
EXPECTED_NAT = {
    "native_cheerful": "A",
    "native_empty_eyes_smile": "B",
    "native_shaded_face_smile": "B",
    "native_sweatdrop_smile": "D",
}

DIRECT5_PROMPT = (
    "Look at this 2D anime character portrait and judge their TRUE emotional / psychological state.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "E) Irritated, annoyed, or suppressing anger behind a smile\n"
    "Answer with ONLY a single letter: A, B, C, D, or E."
)

AUDIT5_PROMPT = (
    "Inspect this 2D anime portrait carefully for ANY facial or non-biological anime graphic affect cues "
    "(such as furrowed angry brows, a red anime cross-vein anger mark, unusual dark forehead shadow, "
    "missing/hollow eye highlights, cheek blush lines, or a floating sweat drop), ignoring ordinary hairpins.\n"
    "In 2D anime, graphic affect symbols and angry brows OVERRIDE a smiling mouth.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "E) Irritated, annoyed, or suppressing anger behind a smile\n"
    "Answer with ONLY a single letter: A, B, C, D, or E."
)


def _get_lm_layers(model):
    for path in [("model", "language_model", "layers"), ("language_model", "layers"), ("model", "layers")]:
        cur = model
        ok = True
        for attr in path:
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
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
    ap.add_argument("--manifest-native", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--peak-layer-frac", type=float, required=True, help="e.g. 0.56 for InternVL3/LLaVA, 0.78 for Qwen2.5, 0.89 for Qwen3-32B")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs9 = [json.loads(l) for l in args.manifest_9var.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs_nat = [json.loads(l) for l in args.manifest_native.read_text(encoding="utf-8").splitlines() if l.strip()]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    abcde_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in LABELS5]

    layers = _get_lm_layers(model)
    l_star = min(len(layers) - 1, max(0, int(round(args.peak_layer_frac * (len(layers) - 1)))))
    cap: dict[str, np.ndarray] = {}

    def hook_fn(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h[0, -1, :].detach().float().cpu().numpy()

    handle = layers[l_star].register_forward_hook(hook_fn)

    def run_recs(recs: list[dict[str, Any]]):
        dlg, alg, fd, fa = [], [], [], []
        for r in recs:
            img = Image.open(r["image_path"]).convert("RGB")
            for prompt, ll, fl in [(DIRECT5_PROMPT, dlg, fd), (AUDIT5_PROMPT, alg, fa)]:
                m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
                t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                inp = processor(text=[t], images=[img], return_tensors="pt").to(model.device)
                ll.append(model(**inp).logits[0, -1, abcde_ids].float().cpu().numpy())
                fl.append(cap["h"].copy())
        return np.stack(dlg), np.stack(alg), np.stack(fd), np.stack(fa)

    try:
        d9, a9, fd9, fa9 = run_recs(recs9)
        dn, an, fdn, fan = run_recs(recs_nat)
    finally:
        handle.remove()

    chars = sorted({r["char_id"] for r in recs9})
    tr_chars, te_chars = set(chars[:10]), set(chars[10:])
    seen_vars = {"orig", "placebo_hairpin", "forehead_shadow_on", "sweat_drop_on"}

    tr_idx = [i for i, r in enumerate(recs9) if r["char_id"] in tr_chars and r["variant"] in seen_vars]
    tr_clean_idx = [i for i, r in enumerate(recs9) if r["char_id"] in tr_chars and r["variant"] in {"orig", "placebo_hairpin"}]
    te_idx = [i for i, r in enumerate(recs9) if r["char_id"] in te_chars]
    te_recs = [recs9[i] for i in te_idx]
    y_tr = np.array([0.0 if EXPECTED9[recs9[i]["variant"]] == "A" else 1.0 for i in tr_idx], dtype=np.float32)

    # Fit peak-layer anomaly gate strictly on 10 train chars x 2 seen symbols
    wd, bd = _fit_logreg(fd9[tr_idx], y_tr, reg=5.0)
    tau_d = _cal_thresh(_score(fd9[tr_idx], wd, bd), y_tr)
    wa, ba = _fit_logreg(fa9[tr_idx], y_tr, reg=5.0)
    tau_a = _cal_thresh(_score(fa9[tr_idx], wa, ba), y_tr)

    # Compute CLEAN-BASELINE LOGIT PRIOR strictly from 10 clean training faces (ZERO conflict labels!)
    mu_clean_d = d9[tr_clean_idx].mean(axis=0)
    mu_clean_a = a9[tr_clean_idx].mean(axis=0)

    def bind_debiased(flags: np.ndarray, dlg: np.ndarray, alg: np.ndarray, valid_labels: list[str], mode: str) -> list[str]:
        valid_nonA = [LABELS5.index(ch) for ch in valid_labels if ch != "A"]
        out = []
        for i, flg in enumerate(flags):
            if not flg:
                out.append("A")
            else:
                if mode == "raw_ens":
                    p1 = np.exp(dlg[i, valid_nonA] - np.max(dlg[i, valid_nonA])); p1 /= p1.sum()
                    p2 = np.exp(alg[i, valid_nonA] - np.max(alg[i, valid_nonA])); p2 /= p2.sum()
                    sc = p1 + p2
                elif mode == "clean_debiased_direct":
                    sc = (dlg[i] - mu_clean_d)[valid_nonA]
                elif mode == "clean_debiased_ens":
                    # Z-normalized clean-delta ensemble (cancels option-order & recency bias!)
                    sd = (dlg[i] - mu_clean_d)[valid_nonA]
                    sa = (alg[i] - mu_clean_a)[valid_nonA]
                    sc = sd + sa
                else:
                    raise ValueError(mode)
                out.append(LABELS5[valid_nonA[int(np.argmax(sc))]])
        return out

    def eval9(preds: list[str]) -> dict[str, Any]:
        by_v: dict[str, list[bool]] = {}
        for r, p in zip(te_recs, preds):
            by_v.setdefault(r["variant"], []).append(p == EXPECTED9[r["variant"]])
        sv = {v: round(float(np.mean(xs)), 4) for v, xs in sorted(by_v.items())}
        return {
            "macro_9": round(float(np.mean(list(sv.values()))), 4),
            "clean": round(float(np.mean([sv["orig"], sv["placebo_hairpin"]])), 4),
            "seen2": round(float(np.mean([sv["forehead_shadow_on"], sv["sweat_drop_on"]])), 4),
            "unseen4": round(float(np.mean([sv["eye_highlight_off"], sv["blush_lines_on"], sv["anger_vein_on"], sv["bio_angry_brows"]])), 4),
            "by_variant": sv,
        }

    def eval_nat(preds: list[str]) -> dict[str, Any]:
        by_v: dict[str, list[bool]] = {}
        for r, p in zip(recs_nat, preds):
            by_v.setdefault(r["variant"], []).append(p == EXPECTED_NAT[r["variant"]])
        sv = {v: round(float(np.mean(xs)), 4) for v, xs in sorted(by_v.items())}
        return {
            "macro_4": round(float(np.mean(list(sv.values()))), 4),
            "clean": sv["native_cheerful"],
            "conf3": round(float(np.mean([sv["native_empty_eyes_smile"], sv["native_shaded_face_smile"], sv["native_sweatdrop_smile"]])), 4),
            "by_variant": sv,
        }

    flg9_d = _score(fd9[te_idx], wd, bd) >= tau_d
    flg9_a = _score(fa9[te_idx], wa, ba) >= tau_a
    flgn_d = _score(fdn, wd, bd) >= tau_d

    # For native 4-way eval, mask to valid 4 labels {A, B, C, D}
    nat4_idx = [0, 1, 2, 3]
    res = {
        "model": Path(args.model_dir).name,
        "peak_layer": l_star,
        "n_layers": len(layers),
        "test6_9var": {
            "direct_5opt": eval9([LABELS5[int(np.argmax(d9[i]))] for i in te_idx]),
            "audit_5opt": eval9([LABELS5[int(np.argmax(a9[i]))] for i in te_idx]),
            "peak_gate_d + raw_ens": eval9(bind_debiased(flg9_d, d9[te_idx], a9[te_idx], LABELS5, "raw_ens")),
            "peak_gate_d + clean_debiased_direct": eval9(bind_debiased(flg9_d, d9[te_idx], a9[te_idx], LABELS5, "clean_debiased_direct")),
            "peak_gate_d + clean_debiased_ens": eval9(bind_debiased(flg9_d, d9[te_idx], a9[te_idx], LABELS5, "clean_debiased_ens")),
            "peak_gate_a + clean_debiased_ens": eval9(bind_debiased(flg9_a, d9[te_idx], a9[te_idx], LABELS5, "clean_debiased_ens")),
        },
        "native_32imgs": {
            "direct_4opt": eval_nat([LABELS5[nat4_idx[int(np.argmax(dn[i, nat4_idx]))]] for i in range(len(recs_nat))]),
            "audit_4opt": eval_nat([LABELS5[nat4_idx[int(np.argmax(an[i, nat4_idx]))]] for i in range(len(recs_nat))]),
            "peak_gate_d + raw_ens": eval_nat(bind_debiased(flgn_d, dn, an, ["A", "B", "C", "D"], "raw_ens")),
            "peak_gate_d + clean_debiased_ens": eval_nat(bind_debiased(flgn_d, dn, an, ["A", "B", "C", "D"], "clean_debiased_ens")),
        },
    }
    out_path = args.out_dir / f"exp13_peak_debiased_{Path(args.model_dir).name}.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== EXP 13 PEAK-LAYER (L{l_star}/{len(layers)}) + CLEAN-PRIOR-DEBIASED BINDER: {res['model']} ===")
    print("  [9-Variant Held-Out Test6 (4 UNSEEN Symbols!)]")
    for k, o in res["test6_9var"].items():
        bv = o["by_variant"]
        print(
            f"    {k:36s} macro9={o['macro_9']:.3f} clean={o['clean']:.2f} seen2={o['seen2']:.2f} UNSEEN4={o['unseen4']:.3f} "
            f"| eye={bv['eye_highlight_off']:.2f} shd={bv['forehead_shadow_on']:.2f} bls={bv['blush_lines_on']:.2f} swt={bv['sweat_drop_on']:.2f} vein={bv['anger_vein_on']:.2f} brw={bv['bio_angry_brows']:.2f}"
        )
    print("  [100% Native End-to-End Anime (32 imgs, ZERO overlays!)]")
    for k, o in res["native_32imgs"].items():
        bv = o["by_variant"]
        print(
            f"    {k:36s} macro4={o['macro_4']:.3f} clean={o['clean']:.2f} conf3={o['conf3']:.3f} "
            f"| empty={bv['native_empty_eyes_smile']:.2f} shd={bv['native_shaded_face_smile']:.2f} swt={bv['native_sweatdrop_smile']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
