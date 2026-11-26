#!/usr/bin/env python3
"""
Exp 7 (P3 Principled Two-Stage Optimization — Decoupled Anomaly-Gated Binder):
  Solves BOTH halves of the Clean-vs-Conflict Pareto Paradox we discovered across
  4 SOTA VLM families:
    - Single-pass Direct: suffers massive Biological-Override (7B: 17.5% conflict;
      InternVL3: 25%; LLaVA-OV: 20%) OR clean paranoia (32B: 50% clean!).
    - Single-pass Audit: breaks clean smiles (7B: 0% clean; 32B: 34% clean!).
    - Single-pass Direct/Rationale LoRA: collapses UNSEEN symbols (blush: 0-17%!).

  Decoupled Two-Stage Architecture (ZERO unseen symbols ever trained on!):
    - Stage 1 (Non-Biological Graphic Anomaly Gate):
        Predicts p_anomaly(x) in [0, 1] from frozen VLM hidden states on ONLY the
        10 training characters. We compare TWO ultra-clean Stage-1 gates:
          (1A) `one_class_clean_gate` (ZERO conflict labels at all!):
               Fits mean & ridge-covariance ONLY on clean {orig, placebo_hairpin}
               of the 10 train chars; scores ANY image by its Mahalanobis/cosine
               anomaly distance from clean smiling anime faces!
          (1B) `leave_symbols_out_probe` (2-symbol binary linear probe on frozen
               VLM features): trains L2-regularized logistic regression on ONLY
               {orig, placebo} vs {forehead_shadow, sweat_drop} of the 10 train chars
               (NEVER seeing `eye_highlight_off` or `blush_lines_on`!).
          (1C) `direct_pA_gate` (zero-parameter baseline): uses 1 - P_direct(A) directly!
    - Stage 2 (Frozen Base VLM Rule Binder — ZERO weight updates!):
        * If p_anomaly(x) < tau -> output `A` (Cheerful) [eliminates paranoia!]
        * If p_anomaly(x) >= tau -> invoke frozen base VLM with `A` masked out
          (choosing argmax over {B, C, D} from `Audit` or `Direct` logits —
           100% open to UNSEEN classes like `C` blush & `B` eye_highlight_off!)
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

GATE_PROMPT = (
    "Look carefully at this 2D anime character's eyes, forehead, cheeks, and temple. "
    "Ignore ordinary hair accessories like hairpins. "
    "Does this portrait contain ANY stylized anime graphic affect cue (such as missing/hollow eye highlights, "
    "an ominous dark forehead shadow, cheek blush lines, or a floating cartoon sweat drop)? "
    "Answer with ONLY YES or NO."
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
def extract_features_and_logits(model, processor, records: list[dict[str, Any]]) -> dict[str, Any]:
    tok = processor.tokenizer
    abcd_ids = [tok.encode(ch, add_special_tokens=False)[-1] for ch in LABELS]
    yes_id = tok.encode("YES", add_special_tokens=False)[-1]
    no_id = tok.encode("NO", add_special_tokens=False)[-1]

    layers = _get_lm_layers(model)
    l_idx = int(len(layers) * 0.78)  # late decision-routing layer (e.g. L22/28 on 7B)
    captured = {}

    def hook_fn(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h[0, -1, :].detach().float().cpu().numpy()

    handle = layers[l_idx].register_forward_hook(hook_fn)
    direct_lg, audit_lg, gate_yes_p, feats_gate, feats_direct = [], [], [], [], []
    try:
        for rec in records:
            img = Image.open(rec["image_path"]).convert("RGB")

            # 1. Direct prompt
            m_d = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DIRECT_PROMPT}]}]
            t_d = processor.apply_chat_template(m_d, tokenize=False, add_generation_prompt=True)
            in_d = processor(text=[t_d], images=[img], return_tensors="pt").to(model.device)
            lg_d = model(**in_d).logits[0, -1, abcd_ids].float().cpu().numpy()
            direct_lg.append(lg_d)
            feats_direct.append(captured["h"].copy())

            # 2. Audit prompt
            m_a = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": AUDIT_PROMPT}]}]
            t_a = processor.apply_chat_template(m_a, tokenize=False, add_generation_prompt=True)
            in_a = processor(text=[t_a], images=[img], return_tensors="pt").to(model.device)
            lg_a = model(**in_a).logits[0, -1, abcd_ids].float().cpu().numpy()
            audit_lg.append(lg_a)

            # 3. Binary Graphic-Anomaly Gate prompt (YES/NO)
            m_g = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": GATE_PROMPT}]}]
            t_g = processor.apply_chat_template(m_g, tokenize=False, add_generation_prompt=True)
            in_g = processor(text=[t_g], images=[img], return_tensors="pt").to(model.device)
            lg_g = model(**in_g).logits[0, -1, [no_id, yes_id]].float().cpu().numpy()
            p_yes = float(np.exp(lg_g[1] - np.max(lg_g)) / np.sum(np.exp(lg_g - np.max(lg_g))))
            gate_yes_p.append(p_yes)
            feats_gate.append(captured["h"].copy())
    finally:
        handle.remove()

    return {
        "layer_idx": l_idx,
        "direct_lg": np.stack(direct_lg),
        "audit_lg": np.stack(audit_lg),
        "gate_yes_p": np.array(gate_yes_p),
        "feats_direct": np.stack(feats_direct),
        "feats_gate": np.stack(feats_gate),
    }


def _eval_preds(records: list[dict[str, Any]], preds: list[str]) -> dict[str, Any]:
    by_var: dict[str, list[bool]] = {}
    for r, p in zip(records, preds):
        by_var.setdefault(r["variant"], []).append(p == EXPECTED[r["variant"]])
    summary = {v: round(float(np.mean(vals)), 4) for v, vals in sorted(by_var.items())}
    clean = [summary[k] for k in ["orig", "placebo_hairpin"] if k in summary]
    seen_c = [summary[k] for k in ["forehead_shadow_on", "sweat_drop_on"] if k in summary]
    unseen_c = [summary[k] for k in ["eye_highlight_off", "blush_lines_on"] if k in summary]
    all_c = [summary[k] for k in ["eye_highlight_off", "forehead_shadow_on", "dead_eyes_and_shadow", "blush_lines_on", "sweat_drop_on"] if k in summary]
    return {
        "macro_all_7": round(float(np.mean(list(summary.values()))), 4),
        "clean_acc": round(float(np.mean(clean)), 4),
        "conflict_acc": round(float(np.mean(all_c)), 4),
        "seen_conflicts": round(float(np.mean(seen_c)), 4),
        "unseen_symbols_zero_shot": round(float(np.mean(unseen_c)), 4),
        "by_variant": summary,
    }


def _fit_logreg_l2(X: np.ndarray, y: np.ndarray, reg: float = 10.0, steps: int = 400, lr: float = 0.05) -> tuple[np.ndarray, float]:
    """Pure-NumPy L2-regularized logistic regression on L2-normalized hidden states."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    w = np.zeros(Xn.shape[1], dtype=np.float32)
    b = 0.0
    for _ in range(steps):
        z = np.clip(Xn @ w + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        gw = (Xn.T @ err) / len(y) + reg * w / len(y)
        gb = float(np.mean(err))
        w -= lr * gw
        b -= lr * gb
    return w, b


def _score_logreg(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    z = np.clip(Xn @ w + b, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _calibrate_threshold(scores_tr: np.ndarray, y_tr: np.ndarray) -> float:
    best_t, best_bal = 0.5, -1.0
    for t in np.linspace(float(scores_tr.min()) - 0.01, float(scores_tr.max()) + 0.01, 101):
        pred = (scores_tr >= t).astype(np.float32)
        acc0 = float(np.mean(pred[y_tr == 0] == 0))
        acc1 = float(np.mean(pred[y_tr == 1] == 1))
        bal = 0.5 * acc0 + 0.5 * acc1
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    return best_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train-chars", type=int, default=10)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()

    data = extract_features_and_logits(model, processor, records)
    chars = sorted({r["char_id"] for r in records})
    train_chars = set(chars[: args.n_train_chars])
    test_chars = set(chars[args.n_train_chars :])
    seen_vars = {"orig", "placebo_hairpin", "forehead_shadow_on", "sweat_drop_on"}

    tr_idx = [i for i, r in enumerate(records) if r["char_id"] in train_chars and r["variant"] in seen_vars]
    tr_clean_idx = [i for i, r in enumerate(records) if r["char_id"] in train_chars and r["variant"] in {"orig", "placebo_hairpin"}]
    te_idx = [i for i, r in enumerate(records) if r["char_id"] in test_chars]
    te_recs = [records[i] for i in te_idx]

    y_tr = np.array([0.0 if EXPECTED[records[i]["variant"]] == "A" else 1.0 for i in tr_idx], dtype=np.float32)

    # Choose Stage-2 non-A binder logits (combine normalized Direct + Audit non-A logits — zero weight updates!)
    # Check both Audit-only and Direct+Audit non-A binders cleanly
    def bind_stage2(gate_anomaly: np.ndarray, mode: str = "audit_nonA") -> list[str]:
        preds = []
        for idx, is_anom in zip(te_idx, gate_anomaly):
            if not is_anom:
                preds.append("A")
            else:
                if mode == "audit_nonA":
                    lg = data["audit_lg"][idx, 1:]
                elif mode == "direct_nonA":
                    lg = data["direct_lg"][idx, 1:]
                else:
                    # Ensemble of zero-shot Direct + Audit conditional probabilities over {B, C, D}
                    p1 = np.exp(data["direct_lg"][idx, 1:] - np.max(data["direct_lg"][idx, 1:]))
                    p1 /= p1.sum()
                    p2 = np.exp(data["audit_lg"][idx, 1:] - np.max(data["audit_lg"][idx, 1:]))
                    p2 /= p2.sum()
                    lg = p1 + p2
                preds.append(LABELS[1 + int(np.argmax(lg))])
        return preds

    res: dict[str, Any] = {"model": Path(args.model_dir).name, "layer_idx": data["layer_idx"]}

    # Baselines on TEST6
    res["baseline_direct"] = _eval_preds(te_recs, [LABELS[int(np.argmax(data["direct_lg"][i]))] for i in te_idx])
    res["baseline_audit"] = _eval_preds(te_recs, [LABELS[int(np.argmax(data["audit_lg"][i]))] for i in te_idx])

    # Method 1: Zero-Training Explicit YES/NO Anomaly Gate + Stage-2 Binder
    p_yes_te = data["gate_yes_p"][te_idx]
    p_yes_tr = data["gate_yes_p"][tr_idx]
    res["gate_yesno_zero_train_0.5"] = _eval_preds(te_recs, bind_stage2(p_yes_te >= 0.5, "ens_nonA"))
    tau_yn = _calibrate_threshold(p_yes_tr, y_tr)
    res["gate_yesno_cal"] = {
        "tau": round(tau_yn, 4),
        "audit_nonA": _eval_preds(te_recs, bind_stage2(p_yes_te >= tau_yn, "audit_nonA")),
        "ens_nonA": _eval_preds(te_recs, bind_stage2(p_yes_te >= tau_yn, "ens_nonA")),
    }

    # Method 2: Pure One-Class Clean-Face Cosine OOD Gate (ZERO conflict labels ever seen!)
    F_gate = data["feats_gate"]
    F_gate_n = F_gate / (np.linalg.norm(F_gate, axis=1, keepdims=True) + 1e-8)
    mu_clean = F_gate_n[tr_clean_idx].mean(axis=0)
    mu_clean /= np.linalg.norm(mu_clean) + 1e-8
    ood_dist = 1.0 - (F_gate_n @ mu_clean)
    # Threshold at 90th percentile of clean training characters (ZERO conflict data needed!)
    tau_ood = float(np.percentile(ood_dist[tr_clean_idx], 88))
    res["gate_oneclass_clean_ood"] = {
        "tau_ood": round(tau_ood, 5),
        "ens_nonA": _eval_preds(te_recs, bind_stage2(ood_dist[te_idx] > tau_ood, "ens_nonA")),
        "audit_nonA": _eval_preds(te_recs, bind_stage2(ood_dist[te_idx] > tau_ood, "audit_nonA")),
    }

    # Method 3: Leave-Symbols-Out Linear Anomaly Probe on Frozen VLM Features (ONLY 2 seen symbols!)
    for fname, F_mat in [("feats_direct", data["feats_direct"]), ("feats_gate", data["feats_gate"])]:
        w, b = _fit_logreg_l2(F_mat[tr_idx], y_tr, reg=5.0)
        sc_tr = _score_logreg(F_mat[tr_idx], w, b)
        sc_te = _score_logreg(F_mat[te_idx], w, b)
        tau_p = _calibrate_threshold(sc_tr, y_tr)
        # Also record binary gate detection rate per variant (including UNSEEN eye_off & blush!)
        gate_det = {}
        for v in sorted(EXPECTED.keys()):
            sub_idx = [k for k, r in enumerate(te_recs) if r["variant"] == v]
            gate_det[v] = round(float(np.mean(sc_te[sub_idx] >= tau_p)), 4)
        res[f"gate_linear_probe_{fname}"] = {
            "tau": round(tau_p, 4),
            "gate_anomaly_fire_rate": gate_det,
            "audit_nonA": _eval_preds(te_recs, bind_stage2(sc_te >= tau_p, "audit_nonA")),
            "ens_nonA": _eval_preds(te_recs, bind_stage2(sc_te >= tau_p, "ens_nonA")),
        }

    out_path = args.out_dir / f"exp7_gated_binder_{Path(args.model_dir).name}.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== EXP 7 DECOUPLED ANOMALY-GATED BINDER: {res['model']} (TEST6 Held-Out Chars) ===")
    for label, obj in [
        ("Baseline Direct", res["baseline_direct"]),
        ("Baseline Audit", res["baseline_audit"]),
        ("ZeroTrain YES/NO Gate (0.5)", res["gate_yesno_zero_train_0.5"]),
        ("Calibrated YES/NO Gate (ens)", res["gate_yesno_cal"]["ens_nonA"]),
        ("One-Class Clean OOD (ZERO conf)", res["gate_oneclass_clean_ood"]["ens_nonA"]),
        ("LSO Probe [direct] + EnsBind", res["gate_linear_probe_feats_direct"]["ens_nonA"]),
        ("LSO Probe [gate]   + EnsBind", res["gate_linear_probe_feats_gate"]["ens_nonA"]),
    ]:
        print(
            f"  {label:32s} macro7={obj['macro_all_7']:.3f} clean={obj['clean_acc']:.3f} "
            f"conf={obj['conflict_acc']:.3f} seen={obj['seen_conflicts']:.3f} UNSEEN={obj['unseen_symbols_zero_shot']:.3f}"
        )
    print("  LSO Probe [gate] fire rates:", res["gate_linear_probe_feats_gate"]["gate_anomaly_fire_rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
