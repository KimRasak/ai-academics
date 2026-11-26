#!/usr/bin/env python3
"""
Exp 9 (P3 Complete 4-Model × 9-Variant Leave-4-Symbols-Out Generalization Test):

Tests our winning Frozen-Hidden-State Anomaly-Gated Binder across ALL FOUR SOTA VLM
architectures (Qwen2.5-VL-7B, Qwen3-VL-32B, InternVL3-8B, LLaVA-OneVision-7B) on the
full 9-variant suite (16 full-color TV-anime chars x 9 variants = 144 images):
  - Train chars (10) x ONLY 4 seen variants: {orig, placebo_hairpin, forehead_shadow_on, sweat_drop_on}
  - Held-out test chars (6) x ALL 9 variants, including FOUR COMPLETELY UNSEEN CUES:
      1. `eye_highlight_off` (B — pure non-bio)
      2. `blush_lines_on`    (C — semi-bio)
      3. `anger_vein_on`     (E — pure non-bio `💢`)
      4. `bio_angry_brows`   (E — matched equal-pixel biological FACS AU4)

Also runs a sample-efficiency sweep (n_train_chars in {2, 5, 10}) and compares
single-layer vs multi-layer frozen hidden-state anomaly gating (ZERO VLM weight updates!).
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


@torch.no_grad()
def extract_all(model, processor, records: list[dict[str, Any]]) -> dict[str, Any]:
    tok = processor.tokenizer
    abcde_ids = [tok.encode(ch, add_special_tokens=False)[-1] for ch in LABELS5]
    layers = _get_lm_layers(model)
    nL = len(layers)
    l_mid = int(nL * 0.55)
    l_late = int(nL * 0.78)
    cap: dict[int, np.ndarray] = {}

    def mk(idx: int):
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            cap[idx] = h[0, -1, :].detach().float().cpu().numpy()
        return fn

    handles = [layers[l_mid].register_forward_hook(mk(l_mid)), layers[l_late].register_forward_hook(mk(l_late))]
    direct_lg, audit_lg, f_d_late, f_d_multi, f_a_multi = [], [], [], [], []
    try:
        for rec in records:
            img = Image.open(rec["image_path"]).convert("RGB")
            # Direct
            m_d = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DIRECT5_PROMPT}]}]
            t_d = processor.apply_chat_template(m_d, tokenize=False, add_generation_prompt=True)
            in_d = processor(text=[t_d], images=[img], return_tensors="pt").to(model.device)
            direct_lg.append(model(**in_d).logits[0, -1, abcde_ids].float().cpu().numpy())
            f_d_late.append(cap[l_late].copy())
            f_d_multi.append(np.concatenate([cap[l_mid], cap[l_late]], axis=0))

            # Audit
            m_a = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": AUDIT5_PROMPT}]}]
            t_a = processor.apply_chat_template(m_a, tokenize=False, add_generation_prompt=True)
            in_a = processor(text=[t_a], images=[img], return_tensors="pt").to(model.device)
            audit_lg.append(model(**in_a).logits[0, -1, abcde_ids].float().cpu().numpy())
            f_a_multi.append(np.concatenate([cap[l_mid], cap[l_late]], axis=0))
    finally:
        for h in handles:
            h.remove()

    return {
        "layers": [l_mid, l_late],
        "direct_lg": np.stack(direct_lg),
        "audit_lg": np.stack(audit_lg),
        "f_d_late": np.stack(f_d_late),
        "f_d_multi": np.stack(f_d_multi),
        "f_a_multi": np.stack(f_a_multi),
    }


def _fit_logreg(X: np.ndarray, y: np.ndarray, reg: float = 5.0, steps: int = 400, lr: float = 0.05) -> tuple[np.ndarray, float]:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    w = np.zeros(Xn.shape[1], dtype=np.float32)
    b = 0.0
    for _ in range(steps):
        z = np.clip(Xn @ w + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        w -= lr * ((Xn.T @ err) / len(y) + reg * w / len(y))
        b -= lr * float(np.mean(err))
    return w, b


def _score_logreg(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -30, 30)))


def _cal_thresh(sc_tr: np.ndarray, y_tr: np.ndarray) -> float:
    best_t, best_b = 0.5, -1.0
    for t in np.linspace(float(sc_tr.min()) - 0.01, float(sc_tr.max()) + 0.01, 101):
        p = (sc_tr >= t).astype(np.float32)
        bal = 0.5 * float(np.mean(p[y_tr == 0] == 0)) + 0.5 * float(np.mean(p[y_tr == 1] == 1))
        if bal > best_b:
            best_b, best_t = bal, float(t)
    return best_t


def _eval(records: list[dict[str, Any]], preds: list[str]) -> dict[str, Any]:
    by_v: dict[str, list[bool]] = {}
    for r, p in zip(records, preds):
        by_v.setdefault(r["variant"], []).append(p == EXPECTED9[r["variant"]])
    sv = {v: round(float(np.mean(xs)), 4) for v, xs in sorted(by_v.items())}
    clean = float(np.mean([sv["orig"], sv["placebo_hairpin"]]))
    seen_c = float(np.mean([sv["forehead_shadow_on"], sv["sweat_drop_on"]]))
    unseen_4 = float(np.mean([sv["eye_highlight_off"], sv["blush_lines_on"], sv["anger_vein_on"], sv["bio_angry_brows"]]))
    unseen_nonbio_2 = float(np.mean([sv["eye_highlight_off"], sv["anger_vein_on"]]))
    return {
        "macro_9": round(float(np.mean(list(sv.values()))), 4),
        "clean_acc": round(clean, 4),
        "seen_2_acc": round(seen_c, 4),
        "unseen_4_acc": round(unseen_4, 4),
        "unseen_pure_nonbio_2": round(unseen_nonbio_2, 4),
        "by_variant": sv,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    chars = sorted({r["char_id"] for r in records})
    test_chars = set(chars[10:])  # 6 held-out unseen chars
    te_idx = [i for i, r in enumerate(records) if r["char_id"] in test_chars]
    te_recs = [records[i] for i in te_idx]
    seen_vars = {"orig", "placebo_hairpin", "forehead_shadow_on", "sweat_drop_on"}

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()

    data = extract_all(model, processor, records)

    def bind(anom_flags: np.ndarray) -> list[str]:
        out = []
        for idx, flg in zip(te_idx, anom_flags):
            if not flg:
                out.append("A")
            else:
                p1 = np.exp(data["direct_lg"][idx, 1:] - np.max(data["direct_lg"][idx, 1:])); p1 /= p1.sum()
                p2 = np.exp(data["audit_lg"][idx, 1:] - np.max(data["audit_lg"][idx, 1:])); p2 /= p2.sum()
                out.append(LABELS5[1 + int(np.argmax(p1 + p2))])
        return out

    res: dict[str, Any] = {"model": Path(args.model_dir).name, "layers": data["layers"]}
    res["direct_5opt"] = _eval(te_recs, [LABELS5[int(np.argmax(data["direct_lg"][i]))] for i in te_idx])
    res["audit_5opt"] = _eval(te_recs, [LABELS5[int(np.argmax(data["audit_lg"][i]))] for i in te_idx])

    # Sample-efficiency sweep: n_train_chars in {2, 5, 10}
    res["probe_sweep"] = {}
    for n_tr in [2, 5, 10]:
        tr_chars = set(chars[:n_tr])
        tr_idx = [i for i, r in enumerate(records) if r["char_id"] in tr_chars and r["variant"] in seen_vars]
        y_tr = np.array([0.0 if EXPECTED9[records[i]["variant"]] == "A" else 1.0 for i in tr_idx], dtype=np.float32)

        for fname in ["f_d_late", "f_d_multi", "f_a_multi"]:
            F_mat = data[fname]
            w, b = _fit_logreg(F_mat[tr_idx], y_tr, reg=5.0)
            sc_tr = _score_logreg(F_mat[tr_idx], w, b)
            sc_te = _score_logreg(F_mat[te_idx], w, b)
            tau = _cal_thresh(sc_tr, y_tr)
            fire = {
                v: round(float(np.mean(sc_te[[k for k, r in enumerate(te_recs) if r["variant"] == v]] >= tau)), 4)
                for v in sorted(EXPECTED9.keys())
            }
            ev = _eval(te_recs, bind(sc_te >= tau))
            ev["gate_fire_rate"] = fire
            res["probe_sweep"][f"n{n_tr}_{fname}"] = ev

    out_path = args.out_dir / f"exp9_9var_probe_{Path(args.model_dir).name}.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== EXP 9 (4-UNSEEN-SYMBOL PROBE GATE): {res['model']} ===")
    for k, o in [
        ("Direct 5-opt", res["direct_5opt"]),
        ("Audit  5-opt", res["audit_5opt"]),
        ("Probe n=2  (d_multi)", res["probe_sweep"]["n2_f_d_multi"]),
        ("Probe n=5  (d_multi)", res["probe_sweep"]["n5_f_d_multi"]),
        ("Probe n=10 (d_late)", res["probe_sweep"]["n10_f_d_late"]),
        ("Probe n=10 (d_multi)", res["probe_sweep"]["n10_f_d_multi"]),
        ("Probe n=10 (a_multi)", res["probe_sweep"]["n10_f_a_multi"]),
    ]:
        print(
            f"  {k:22s} macro9={o['macro_9']:.3f} clean={o['clean_acc']:.3f} seen2={o['seen_2_acc']:.3f} "
            f"UNSEEN4={o['unseen_4_acc']:.3f} UNSEEN_nonbio2={o['unseen_pure_nonbio_2']:.3f}"
        )
    print("  n10_f_d_multi gate fire:", res["probe_sweep"]["n10_f_d_multi"]["gate_fire_rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
