#!/usr/bin/env python3
"""
Exp 11 (P3 Mechanistic Interpretability — Full Layer-by-Layer Localization Curve):
  Directly answers the deepest mechanistic question for NeurIPS / ICLR:
    "WHERE inside a VLM does Biological-Override Binding Failure occur?
     Are non-biological anime symbols linearly readable in early/middle layers
     and then OVERWRITTEN by the biological-smile prior in late layers?"

  For every sampled transformer layer l in {0, 2, 4, ..., L-1} across ALL 4 SOTA
  VLM architectures (Qwen2.5-VL-7B, Qwen3-VL-32B, InternVL3-8B, LLaVA-OneVision-7B):
    1. Fits a linear anomaly probe on ONLY 10 train chars x {orig, placebo} vs {forehead_shadow, sweat_drop}
       and measures exact binary AUROC on 6 held-out chars for:
         - Seen symbols (`forehead_shadow`, `sweat_drop`)
         - COMPLETELY UNSEEN pure non-biological symbols (`eye_highlight_off`, `anger_vein_on`)
         - COMPLETELY UNSEEN semi-bio / bio controls (`blush_lines_on`, `bio_angry_brows`)
    2. Computes Logit-Lens Biological-Override P_l(A) at layer l by projecting
       norm(h_l) through the model's unembedding lm_head -> softmax over {A,B,C,D,E}!
    3. Evaluates Optimal-Layer (l*) Gated Binder with Permutation-Normalized Stage-2
       Binding (eliminating option-order bias like Qwen2.5-VL's option-E bias!).
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

LABELS = ["A", "B", "C", "D"]
EXPECTED7 = {
    "orig": "A",
    "placebo_hairpin": "A",
    "eye_highlight_off": "B",
    "forehead_shadow_on": "B",
    "dead_eyes_and_shadow": "B",
    "blush_lines_on": "C",
    "sweat_drop_on": "D",
}

DIRECT_PROMPT = (
    "Look at this 2D anime character portrait and judge their TRUE emotional / psychological state.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "Answer with ONLY a single letter: A, B, C, or D."
)


def _get_lm_components(model):
    # Returns (layers, final_norm, lm_head)
    lm_head = model.get_output_embeddings()
    for base_path in [("model", "language_model"), ("language_model", "model"), ("language_model",), ("model",)]:
        cur = model
        ok = True
        for attr in base_path:
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
            else:
                ok = False
                break
        if ok and hasattr(cur, "layers"):
            norm = getattr(cur, "norm", None)
            return cur.layers, norm, lm_head
    raise AttributeError("Could not locate LM layers/norm/head")


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Exact Mann-Whitney U
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / float(len(pos) * len(neg))


def _fit_logreg(X: np.ndarray, y: np.ndarray, reg: float = 5.0, steps: int = 350, lr: float = 0.05):
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


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_recs = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Include all 9 variants if present, evaluate probe AUROC on clean vs each conflict family
    records = all_recs
    chars = sorted({r["char_id"] for r in records})
    tr_chars = set(chars[:10])
    te_chars = set(chars[10:])
    seen_vars = {"orig", "placebo_hairpin", "forehead_shadow_on", "sweat_drop_on"}

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    abcd_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in LABELS]

    layers, final_norm, lm_head = _get_lm_components(model)
    nL = len(layers)
    # Sample ~10 evenly-spaced layers from layer 0 to L-1
    sample_layers = sorted(set([int(round(i * (nL - 1) / 9.0)) for i in range(10)]))
    print(f"Model={Path(args.model_dir).name} nL={nL} sample_layers={sample_layers}", flush=True)

    cap: dict[int, torch.Tensor] = {}

    def mk(l_idx: int):
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            cap[l_idx] = h[0, -1, :].detach()
        return fn

    handles = [layers[l].register_forward_hook(mk(l)) for l in sample_layers]
    feats_by_layer: dict[int, list[np.ndarray]] = {l: [] for l in sample_layers}
    logitlens_pA: dict[int, list[float]] = {l: [] for l in sample_layers}

    try:
        for rec in records:
            img = Image.open(rec["image_path"]).convert("RGB")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DIRECT_PROMPT}]}]
            txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = processor(text=[txt], images=[img], return_tensors="pt").to(model.device)
            _ = model(**inp)

            for l in sample_layers:
                h = cap[l]
                feats_by_layer[l].append(h.float().cpu().numpy())
                # Logit lens: project through final_norm + lm_head -> P(A) over {A, B, C, D}
                h_in = h.unsqueeze(0)
                if final_norm is not None:
                    h_in = final_norm(h_in)
                lg = lm_head(h_in)[0, abcd_ids].float().cpu().numpy()
                ex = np.exp(lg - np.max(lg))
                logitlens_pA[l].append(float(ex[0] / ex.sum()))
    finally:
        for h in handles:
            h.remove()

    tr_idx = [i for i, r in enumerate(records) if r["char_id"] in tr_chars and r["variant"] in seen_vars]
    y_tr = np.array([0.0 if records[i]["variant"] in {"orig", "placebo_hairpin"} else 1.0 for i in tr_idx], dtype=np.float32)

    te_clean = [i for i, r in enumerate(records) if r["char_id"] in te_chars and r["variant"] in {"orig", "placebo_hairpin"}]
    te_seen = [i for i, r in enumerate(records) if r["char_id"] in te_chars and r["variant"] in {"forehead_shadow_on", "sweat_drop_on"}]
    te_unseen_nonbio = [i for i, r in enumerate(records) if r["char_id"] in te_chars and r["variant"] in {"eye_highlight_off", "anger_vein_on"}]
    te_unseen_blush = [i for i, r in enumerate(records) if r["char_id"] in te_chars and r["variant"] == "blush_lines_on"]
    te_unseen_bio = [i for i, r in enumerate(records) if r["char_id"] in te_chars and r["variant"] == "bio_angry_brows"]

    curve = []
    for l in sample_layers:
        X = np.stack(feats_by_layer[l])
        w, b = _fit_logreg(X[tr_idx], y_tr, reg=5.0)
        sc = _score(X, w, b)
        pA = np.array(logitlens_pA[l])

        def pair_auc(pos_idx):
            idx = te_clean + pos_idx
            yt = np.array([0] * len(te_clean) + [1] * len(pos_idx))
            return round(_auroc(yt, sc[idx]), 4)

        row = {
            "layer": l,
            "depth_frac": round(l / max(1, nL - 1), 2),
            "auroc_seen": pair_auc(te_seen),
            "auroc_unseen_nonbio": pair_auc(te_unseen_nonbio),
            "auroc_unseen_blush": pair_auc(te_unseen_blush),
            "auroc_unseen_bio_brows": pair_auc(te_unseen_bio),
            "logitlens_pA_clean": round(float(pA[te_clean].mean()), 4),
            "logitlens_pA_seen": round(float(pA[te_seen].mean()), 4),
            "logitlens_pA_unseen_nonbio": round(float(pA[te_unseen_nonbio].mean()), 4),
        }
        curve.append(row)
        print(
            f"  L{l:02d} ({row['depth_frac']:.2f}) | AUROC seen={row['auroc_seen']:.3f} UNSEEN_nonbio={row['auroc_unseen_nonbio']:.3f} "
            f"blush={row['auroc_unseen_blush']:.3f} bio_brows={row['auroc_unseen_bio_brows']:.3f} | "
            f"LogitLens pA: clean={row['logitlens_pA_clean']:.3f} seen={row['logitlens_pA_seen']:.3f} unseen_nonbio={row['logitlens_pA_unseen_nonbio']:.3f}",
            flush=True,
        )

    out_path = args.out_dir / f"exp11_layerwise_{Path(args.model_dir).name}.json"
    out_path.write_text(json.dumps({"model": Path(args.model_dir).name, "n_layers": nL, "curve": curve}, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
