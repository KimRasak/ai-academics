#!/usr/bin/env python3
"""
Exp 18A (P3 Mechanistic Psychometric Dose-Response Curve —
         Monotonic Mid-Layer Perception vs. Flat Late-Layer Biological-Override Clamp!):

Sweeps pixel-linear opacity/prominence lambda in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} for:
  - Seen cue: `forehead_shadow_on`
  - UNSEEN non-bio cue #1: `eye_highlight_off`
  - UNSEEN non-bio cue #2: `anger_vein_on` (`💢`)
  - Placebo control: `placebo_hairpin` (MUST stay flat in BOTH mid-layer and final-layer!)
across 12 held-out unseen characters (fc_36..fc_47; probe trained ONLY on binary {0, 1}
for {orig, placebo} vs {forehead_shadow, sweat_drop} on fc_00..fc_35!).

Proves whether mid-layer linear representation tracks symbol prominence smoothly/monotonically
even while the final output stays clamped at high P(A: Cheerful) by the late biological override!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp17c_scale48_5fold_and_unified_1pass as e17c

LAMBDAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TEST_CUES = ["forehead_shadow_on", "eye_highlight_off", "anger_vein_on", "placebo_hairpin"]


def _spearman(x: list[float], y: list[float]) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 1e-12 else 0.0


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest48", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    chars = sorted({r["char_id"] for r in recs})
    tr_chars = set(chars[:36])
    te_chars = sorted(chars[36:])  # 12 held-out unseen chars

    by_cv = {(r["char_id"], r["variant"]): r["image_path"] for r in recs}

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ids5 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS5]

    layers, final_norm, lm_head = e17c._get_lm(model)
    nL = len(layers)
    l_mid = int(nL * 0.55)
    l_late_gate = int(nL * 0.72)

    cap: dict[int, np.ndarray] = {}
    hooks = []
    for li in (l_mid, l_late_gate):
        def _mk(idx):
            def _fn(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                cap[idx] = h[0, -1, :].detach().float().cpu().numpy()
            return _fn
        hooks.append(layers[li].register_forward_hook(_mk(li)))

    def forward_one(img: Image.Image):
        cap.clear()
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": e17c.DIRECT5_PROMPT}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[img], return_tensors="pt").to(model.device)
        out = model(**inp)
        lg = out.logits[0, -1, ids5].float().cpu().numpy()
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        h = np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0)
        return float(pr[0]), h

    # 1. Fit frozen mid-layer anomaly probe strictly on fc_00..fc_35 x {orig, placebo} vs {forehead_shadow, sweat_drop}
    X_tr, y_tr = [], []
    for r in recs:
        if r["char_id"] in tr_chars and r["variant"] in e17c.SEEN_TRAIN_VARS:
            img = Image.open(r["image_path"]).convert("RGB")
            _, h = forward_one(img)
            X_tr.append(h)
            y_tr.append(0.0 if r["variant"] in {"orig", "placebo_hairpin"} else 1.0)
    X_tr = np.stack(X_tr, axis=0)
    y_tr = np.array(y_tr, dtype=np.float32)
    w, b = e17c._fit_logreg(X_tr, y_tr)

    # 2. Sweep lambda in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} across 12 unseen chars x 4 cues
    curves = {}
    for cue in TEST_CUES:
        pts = []
        for lam in LAMBDAS:
            pA_list, mid_list = [], []
            for cid in te_chars:
                im0 = Image.open(by_cv[(cid, "orig")]).convert("RGB")
                im1 = Image.open(by_cv[(cid, cue)]).convert("RGB")
                if lam <= 0.0:
                    im = im0
                elif lam >= 1.0:
                    im = im1
                else:
                    im = Image.blend(im0, im1, alpha=lam)
                pA, h = forward_one(im)
                s_mid = float(e17c._score(h[None, :], w, b)[0])
                pA_list.append(pA)
                mid_list.append(s_mid)
            pts.append({
                "lambda": lam,
                "mean_mid_probe_score": round(float(np.mean(mid_list)), 4),
                "std_mid_probe_score": round(float(np.std(mid_list)), 4),
                "mean_final_pA_override": round(float(np.mean(pA_list)), 4),
                "std_final_pA_override": round(float(np.std(pA_list)), 4),
            })
        rho_mid = _spearman(LAMBDAS, [p["mean_mid_probe_score"] for p in pts])
        rho_pA = _spearman(LAMBDAS, [p["mean_final_pA_override"] for p in pts])
        curves[cue] = {
            "spearman_mid_vs_lambda": round(rho_mid, 4),
            "spearman_final_pA_vs_lambda": round(rho_pA, 4),
            "delta_mid_0_to_1": round(pts[-1]["mean_mid_probe_score"] - pts[0]["mean_mid_probe_score"], 4),
            "delta_pA_0_to_1": round(pts[-1]["mean_final_pA_override"] - pts[0]["mean_final_pA_override"], 4),
            "points": pts,
        }
        print(
            f"[{Path(args.model_dir).name}] {cue:20s} | Mid-Probe: {pts[0]['mean_mid_probe_score']:.3f} -> {pts[-1]['mean_mid_probe_score']:.3f} (rho={rho_mid:+.3f}) | "
            f"Final P(A): {pts[0]['mean_final_pA_override']:.3f} -> {pts[-1]['mean_final_pA_override']:.3f} (rho={rho_pA:+.3f})",
            flush=True,
        )

    for hk in hooks:
        hk.remove()

    out_p = args.out_dir / f"exp18a_dose_response_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "n_test_chars": len(te_chars), "curves": curves}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
