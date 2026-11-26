#!/usr/bin/env python3
"""
Exp 16 (P3 Causal Double-Dissociation & 1-Forward-Pass Mid-Layer-Gated Circuit Surgery!):

Part 1 — Causal Double Dissociation (Attention vs. MLP Override Circuit):
  Directly tests the architectural divergence uncovered by Exp 14A DLA:
    - InternVL3-8B: Biological-Smile Override is Attention-driven (L20..L27 self_attn)
    - Qwen2.5-VL-7B: Biological-Smile Override is MLP-driven (L20..L27 mlp)
  We project out ONLY the 1D biological-override direction w_bio_hat from:
    (a) Late self_attn outputs ONLY
    (b) Late mlp outputs ONLY
    (c) Both late self_attn + mlp outputs
  and measure the exact causal drop in P(A) and accuracy gain on held-out conflicts!

Part 2 — 1-Forward-Pass Mid-Layer-Gated Causal Surgery (1.0x Inference Cost!):
  During a SINGLE standard Direct forward pass (ZERO second forward pass, ZERO CoT,
  ZERO weight updates, ZERO forgetting on clean/general queries!):
    1. At mid-layer l_gate (e.g. L15/L21), a lightweight frozen-state linear anomaly
       gate (trained ONLY on 10 train chars x {orig, placebo} vs {forehead_shadow, sweat_drop})
       computes p_anom = sigmoid(w^T h_mid + b) IN-PLACE during the forward pass!
    2. ONLY if p_anom >= tau, late layers [l_start..L-1] attenuate the 1D biological-override
       projection (self_attn + mlp) along w_bio_hat by factor alpha!
    3. If p_anom < tau (clean faces / placebo), the late layers are 100% untouched!

  Evaluates on:
    - 6 held-out unseen chars x ALL 7 variants (including UNSEEN `eye_highlight_off` & `blush_lines_on`)
    - 32 100% Native End-to-End Animagine-XL-4.0 anime images (cross-domain Synth -> Native!)
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
EXPECTED_NAT = {
    "native_cheerful": "A",
    "native_empty_eyes_smile": "B",
    "native_shaded_face_smile": "B",
    "native_sweatdrop_smile": "D",
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


def _get_lm(model):
    lm_head = model.get_output_embeddings()
    for path in [("model", "language_model"), ("language_model", "model"), ("language_model",), ("model",)]:
        cur = model
        ok = True
        for a in path:
            if hasattr(cur, a):
                cur = getattr(cur, a)
            else:
                ok = False
                break
        if ok and hasattr(cur, "layers"):
            return cur.layers, getattr(cur, "norm", None), lm_head
    raise AttributeError("Cannot locate LM")


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
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--manifest-native", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs_nat = [json.loads(l) for l in args.manifest_native.read_text(encoding="utf-8").splitlines() if l.strip()]

    chars = sorted({r["char_id"] for r in recs})
    tr_chars, te_chars = set(chars[:10]), set(chars[10:])
    seen_vars = {"orig", "placebo_hairpin", "forehead_shadow_on", "sweat_drop_on"}
    tr_recs = [r for r in recs if r["char_id"] in tr_chars and r["variant"] in seen_vars]
    te_recs = [r for r in recs if r["char_id"] in te_chars and r["variant"] in EXPECTED7]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    abcd_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in LABELS]

    layers, final_norm, lm_head = _get_lm(model)
    nL = len(layers)
    l_mid = int(nL * 0.55)
    l_late_gate = int(nL * 0.72)  # strictly BEFORE late surgery window [l_surg..nL-1]
    l_surg = l_late_gate + 1      # e.g. L21..L27 on 28-layer model!

    # Unit biological-override direction w_bio_hat in residual space
    W_abcd = lm_head.weight[abcd_ids].detach().float()
    w_bio = W_abcd[0] - W_abcd[1:].mean(dim=0)
    if final_norm is not None and hasattr(final_norm, "weight"):
        w_bio = w_bio * final_norm.weight.detach().float()
    w_bio_hat = (w_bio / (w_bio.norm() + 1e-8)).to(model.device)

    def encode(img: Image.Image):
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DIRECT_PROMPT}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        return processor(text=[t], images=[img], return_tensors="pt").to(model.device)

    # 1. Fit 1-forward-pass mid-layer anomaly gate strictly on 10 train chars x 2 seen symbols
    cap_g: dict[int, np.ndarray] = {}
    def mk_g(idx):
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            cap_g[idx] = h[0, -1, :].detach().float().cpu().numpy()
        return fn
    hg = [layers[l_mid].register_forward_hook(mk_g(l_mid)), layers[l_late_gate].register_forward_hook(mk_g(l_late_gate))]
    X_tr_list = []
    try:
        for r in tr_recs:
            inp = encode(Image.open(r["image_path"]).convert("RGB"))
            _ = model(**inp)
            X_tr_list.append(np.concatenate([cap_g[l_mid], cap_g[l_late_gate]], axis=0))
    finally:
        for h in hg:
            h.remove()
    y_tr = np.array([0.0 if EXPECTED7[r["variant"]] == "A" else 1.0 for r in tr_recs], dtype=np.float32)
    X_tr = np.stack(X_tr_list)
    w_gate, b_gate = _fit_logreg(X_tr, y_tr, reg=5.0)
    tau_gate = _cal_thresh(_score(X_tr, w_gate, b_gate), y_tr)
    w_gate_t = torch.from_numpy(w_gate).to(model.device)

    def run_intervention(eval_recs: list[dict[str, Any]], exp_map: dict[str, str], target: str, alpha: float, gated: bool) -> dict[str, Any]:
        """
        target in {'none', 'attn', 'mlp', 'both'}
        If gated=True, runs TRUE 1-FORWARD-PASS surgery: computes p_anom in-place at
        (l_mid, l_late_gate), then conditionally projects out positive w_bio_hat component
        in layers [l_surg..nL-1] ONLY if p_anom >= tau_gate!
        """
        state = {"f_mid": None, "fire": not gated}
        handles = []

        if gated:
            def h_mid(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                state["f_mid"] = h[0, -1, :].detach().float()
            def h_lg(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                f_cat = torch.cat([state["f_mid"], h[0, -1, :].detach().float()], dim=0)
                f_cat = f_cat / (f_cat.norm() + 1e-8)
                p = float(torch.sigmoid(torch.dot(f_cat, w_gate_t) + b_gate).item())
                state["fire"] = p >= tau_gate
            handles.append(layers[l_mid].register_forward_hook(h_mid))
            handles.append(layers[l_late_gate].register_forward_hook(h_lg))

        if target != "none" and alpha > 0.0:
            def ablate_hook(_m, _i, out):
                if not state["fire"]:
                    return out
                if isinstance(out, tuple):
                    h = out[0].clone()
                    v = h[0, -1, :].float()
                    proj = torch.dot(v, w_bio_hat)
                    if proj > 0:
                        h[0, -1, :] = (v - alpha * proj * w_bio_hat).to(h.dtype)
                    return (h,) + out[1:]
                h = out.clone()
                v = h[0, -1, :].float()
                proj = torch.dot(v, w_bio_hat)
                if proj > 0:
                    h[0, -1, :] = (v - alpha * proj * w_bio_hat).to(h.dtype)
                return h

            for l in range(l_surg, nL):
                if target in {"attn", "both"}:
                    handles.append(layers[l].self_attn.register_forward_hook(ablate_hook))
                if target in {"mlp", "both"}:
                    handles.append(layers[l].mlp.register_forward_hook(ablate_hook))

        by_v: dict[str, list[bool]] = {}
        by_pA: dict[str, list[float]] = {}
        try:
            for r in eval_recs:
                inp = encode(Image.open(r["image_path"]).convert("RGB"))
                lg = model(**inp).logits[0, -1, abcd_ids].float().cpu().numpy()
                ex = np.exp(lg - np.max(lg)); p = ex / ex.sum()
                pred = LABELS[int(np.argmax(p))]
                by_v.setdefault(r["variant"], []).append(pred == exp_map[r["variant"]])
                by_pA.setdefault(r["variant"], []).append(float(p[0]))
        finally:
            for h in handles:
                h.remove()

        sv = {v: round(float(np.mean(xs)), 4) for v, xs in sorted(by_v.items())}
        spA = {v: round(float(np.mean(xs)), 4) for v, xs in sorted(by_pA.items())}
        clean_keys = [k for k in sv if exp_map[k] == "A"]
        conf_keys = [k for k in sv if exp_map[k] != "A"]
        return {
            "macro": round(float(np.mean(list(sv.values()))), 4),
            "clean": round(float(np.mean([sv[k] for k in clean_keys])), 4),
            "conflicts": round(float(np.mean([sv[k] for k in conf_keys])), 4),
            "mean_pA_conflicts": round(float(np.mean([spA[k] for k in conf_keys])), 4),
            "by_variant": sv,
            "mean_pA": spA,
        }

    print(f"=== EXP 16 CAUSAL DOUBLE-DISSOCIATION & 1-PASS GATED SURGERY: {Path(args.model_dir).name} (gate=[L{l_mid},L{l_late_gate}], surg=L{l_surg}..L{nL-1}) ===", flush=True)
    res: dict[str, Any] = {"model": Path(args.model_dir).name, "gate_layers": [l_mid, l_late_gate], "surg_layers": [l_surg, nL - 1]}

    # Part 1: Unconditional Double Dissociation (Attn-only vs MLP-only vs Both)
    configs = [
        ("baseline_none", "none", 0.0, False),
        ("uncond_ATTN_only_a1.0", "attn", 1.0, False),
        ("uncond_MLP_only_a1.0", "mlp", 1.0, False),
        ("uncond_BOTH_a1.0", "both", 1.0, False),
        ("gated_1PASS_BOTH_a1.0", "both", 1.0, True),
        ("gated_1PASS_BOTH_a1.5", "both", 1.5, True),
        ("gated_1PASS_BOTH_a2.0", "both", 2.0, True),
    ]
    res["test6_7var"] = {}
    for name, tgt, alp, gtd in configs:
        o = run_intervention(te_recs, EXPECTED7, tgt, alp, gtd)
        res["test6_7var"][name] = o
        bv = o["by_variant"]
        print(
            f"  [Test6] {name:24s} macro={o['macro']:.3f} clean={o['clean']:.3f} conf={o['conflicts']:.3f} pA_conf={o['mean_pA_conflicts']:.3f} "
            f"| eye={bv['eye_highlight_off']:.2f} shd={bv['forehead_shadow_on']:.2f} d+s={bv['dead_eyes_and_shadow']:.2f} bls={bv['blush_lines_on']:.2f} swt={bv['sweat_drop_on']:.2f}",
            flush=True,
        )

    res["native_32imgs"] = {}
    for name, tgt, alp, gtd in [("baseline_none", "none", 0.0, False), ("gated_1PASS_BOTH_a1.5", "both", 1.5, True), ("gated_1PASS_BOTH_a2.0", "both", 2.0, True)]:
        o = run_intervention(recs_nat, EXPECTED_NAT, tgt, alp, gtd)
        res["native_32imgs"][name] = o
        bv = o["by_variant"]
        print(
            f"  [Native] {name:23s} macro={o['macro']:.3f} clean={o['clean']:.3f} conf={o['conflicts']:.3f} pA_conf={o['mean_pA_conflicts']:.3f} "
            f"| empty={bv['native_empty_eyes_smile']:.2f} shd={bv['native_shaded_face_smile']:.2f} swt={bv['native_sweatdrop_smile']:.2f}",
            flush=True,
        )

    out_path = args.out_dir / f"exp16_1pass_surgery_{Path(args.model_dir).name}.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
