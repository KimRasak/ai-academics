#!/usr/bin/env python3
"""
Exp 30A (Track `P7` — Confound-Free Cross-Domain Probing + Direct Causal Layer Patching!):

Closes EVERY possible mechanistic confound on P7:
  1. **Confound-Free Cross-Domain Linear Identity Probe** (`Train strictly on SameColor_RGB` ->
     `Eval ZERO-SHOT on BWGray_RGB & HueShift_RGB` at every layer l = 0..L):
     Because `SameColor_RGB` has ZERO grayscale or hue-shift cue, a probe trained on `SameColor_RGB`
     can ONLY read the model's genuine "Candidate A matches Ref vs Candidate B matches Ref" direction!
     Applying it zero-shot to `BWGray_RGB` and `HueShift_RGB` tests whether the internal identity
     representation itself is already hijacked by color OR remains intact before the final readout!
  2. **Direct Causal Block Patching (`BWGray_Luma -> BWGray_RGB`)**:
     On `BWGray_RGB` (where `Qwen3-VL-32B` collapses to ~46–52% and `InternVL3-14B` collapses to ~28–33%),
     we causally patch ONLY the last-token residual stream (`h_l[:, -1]`) OR ONLY the late-layer
     attention outputs (`self_attn[:, -1]`) from `BWGray_Luma` into `BWGray_RGB` across 4 quarter-depth
     windows (`Q1: 0..25%`, `Q2: 25..50%`, `Q3: 50..75%`, `Q4: 75..100%`) plus `All-Late-Attn (50..100%)`
     and `All-Late-MLP (50..100%)`!
     This directly separates:
       - Whether the Color-Palette Shortcut is causally localized in the **late decision readout (`Q4`)**
         specifically inside **Late Attention** vs **Late MLP**, OR requires early visual token intervention!
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp14a_causal_head_patching as e14a
import exp25a_p7_hair_color_shortcut_pilot as e25a
import exp26b_p7_deconfounded_xpose_benchmark as e26b


def _fit_ridge_dir(X: np.ndarray, y: np.ndarray, l2: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    K = Xc @ Xc.T + l2 * np.eye(Xc.shape[0], dtype=np.float32)
    alpha = np.linalg.solve(K, (2.0 * y - 1.0).astype(np.float32))
    w = Xc.T @ alpha  # [D]
    return mu.squeeze(0), w


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest48", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--xpose-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp26a_p7_crosspose"))
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int, default=24)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xdir = args.xpose_dir / "crosspose_imgs"
    pairs = e26b._build_deconfounded_pairs(args.manifest48, xdir)[: args.max_pairs]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    layers, _norm, _lm_head = e14a._get_lm(model)
    n_layers = len(layers)
    ab_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in ["A", "B"]]

    def prep(ref: Image.Image, cA: Image.Image, cB: Image.Image):
        m = [{"role": "user", "content": [
            {"type": "image", "image": ref},
            {"type": "image", "image": cA},
            {"type": "image", "image": cB},
            {"type": "text", "text": e25a.PROMPT_REID},
        ]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        return processor(text=[t], images=[ref, cA, cB], return_tensors="pt").to(model.device)

    # =========================================================================
    # PART 1: CONFOUND-FREE CROSS-DOMAIN LINEAR PROBE (Train SameColor -> Test BWGray/HueShift)
    # =========================================================================
    probe_layers = sorted(set(list(range(0, n_layers + 1, max(1, n_layers // 16))) + [n_layers]))
    H_same = {l: [] for l in probe_layers}
    H_bw = {l: [] for l in probe_layers}
    H_hue = {l: [] for l in probe_layers}
    H_luma = {l: [] for l in probe_layers}
    y_all, pid_all = [], []

    print(f"\n=== EXP 30A CONFOUND-FREE PROBE + CAUSAL PATCHING: {Path(args.model_dir).name} (L={n_layers}, n={len(pairs)}) ===", flush=True)
    for pi, (r_pos, _r_neg, xp_pos, xp_neg) in enumerate(pairs):
        ir = Image.open(r_pos["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
        ip = Image.open(xp_pos).convert("RGB").resize((448, 448), Image.LANCZOS)
        in_ = Image.open(xp_neg).convert("RGB").resize((448, 448), Image.LANCZOS)
        gr, gp, gn = e26b._clean_luma_gray(ir), e26b._clean_luma_gray(ip), e26b._clean_luma_gray(in_)
        ip_hue = e25a._to_hue_shifted(ip, 95)

        for (cA_s, cB_s, cA_b, cB_b, cA_h, cB_h, cA_l, cB_l, tgt) in [
            (ip, in_, gp, in_, ip_hue, in_, gp, gn, 0),
            (in_, ip, in_, gp, in_, ip_hue, gn, gp, 1),
        ]:
            for buf, (a, b, c) in [
                (H_same, (ir, cA_s, cB_s)),
                (H_bw, (ir, cA_b, cB_b)),
                (H_hue, (ir, cA_h, cB_h)),
                (H_luma, (gr, cA_l, cB_l)),
            ]:
                out = model(**prep(a, b, c), output_hidden_states=True)
                for l in probe_layers:
                    buf[l].append(out.hidden_states[l][0, -1].detach().float().cpu().numpy())
            y_all.append(tgt)
            pid_all.append(pi)

    y_arr = np.array(y_all, dtype=np.int64)
    pid_arr = np.array(pid_all, dtype=np.int64)
    uniq_p = np.unique(pid_arr)

    xdomain_probe = {}
    for l in probe_layers:
        Xs = np.stack(H_same[l], axis=0)
        Xb = np.stack(H_bw[l], axis=0)
        Xh = np.stack(H_hue[l], axis=0)
        Xl = np.stack(H_luma[l], axis=0)
        c_same = c_bw = c_hue = c_luma = tot = 0
        for f in range(4):
            te_p = set(uniq_p[f::4].tolist())
            te = np.array([p in te_p for p in pid_arr], dtype=bool)
            tr = ~te
            mu, w = _fit_ridge_dir(Xs[tr], y_arr[tr], l2=20.0)
            c_same += int(np.sum(((Xs[te] - mu) @ w > 0) == y_arr[te]))
            c_bw += int(np.sum(((Xb[te] - mu) @ w > 0) == y_arr[te]))
            c_hue += int(np.sum(((Xh[te] - mu) @ w > 0) == y_arr[te]))
            c_luma += int(np.sum(((Xl[te] - mu) @ w > 0) == y_arr[te]))
            tot += int(te.sum())
        xdomain_probe[str(l)] = {
            "SameColor_CV": round(c_same / tot, 4),
            "Same2BWGray_ZS": round(c_bw / tot, 4),
            "Same2HueShift_ZS": round(c_hue / tot, 4),
            "Same2Luma_ZS": round(c_luma / tot, 4),
        }
    print("  [Part 1: Confound-Free Zero-Shot Cross-Domain Identity Probe (Train SameColor -> Eval OOD)]", flush=True)
    for l in probe_layers[::2] + ([probe_layers[-1]] if probe_layers[-1] not in probe_layers[::2] else []):
        p = xdomain_probe[str(l)]
        print(f"    L{l:02d}: SameCV={p['SameColor_CV']:.3f} | ->BWGray={p['Same2BWGray_ZS']:.3f} | ->HueShift={p['Same2HueShift_ZS']:.3f} | ->Luma={p['Same2Luma_ZS']:.3f}", flush=True)

    # =========================================================================
    # PART 2: DIRECT CAUSAL ACTIVATION PATCHING (BWGray_Luma -> BWGray_RGB)
    # =========================================================================
    q1, q2, q3 = n_layers // 4, n_layers // 2, (3 * n_layers) // 4
    conditions = [
        ("Raw_BWGray_RGB", None, None),
        ("Clean_BWGray_Luma", "clean", None),
        ("Patch_Q1_Resid_0_25", "resid", range(0, q1)),
        ("Patch_Q2_Resid_25_50", "resid", range(q1, q2)),
        ("Patch_Q3_Resid_50_75", "resid", range(q2, q3)),
        ("Patch_Q4_Resid_75_100", "resid", range(q3, n_layers)),
        ("Patch_Late_Attn_50_100", "attn", range(q2, n_layers)),
        ("Patch_Late_MLP_50_100", "mlp", range(q2, n_layers)),
        ("Patch_Late_Both_50_100", "both", range(q2, n_layers)),
    ]
    patch_acc = {name: [] for name, _, _ in conditions}
    patch_mrg = {name: [] for name, _, _ in conditions}

    for r_pos, _r_neg, xp_pos, xp_neg in pairs:
        ir = Image.open(r_pos["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
        ip = Image.open(xp_pos).convert("RGB").resize((448, 448), Image.LANCZOS)
        in_ = Image.open(xp_neg).convert("RGB").resize((448, 448), Image.LANCZOS)
        gr, gp, gn = e26b._clean_luma_gray(ir), e26b._clean_luma_gray(ip), e26b._clean_luma_gray(in_)

        for (cA_b, cB_b, cA_l, cB_l, tgt) in [(gp, in_, gp, gn, 0), (in_, gp, gn, gp, 1)]:
            wrong = 1 - tgt
            # Capture clean Luma cache (last-token resid, attn, mlp at every layer)
            cache_resid: list[torch.Tensor | None] = [None] * n_layers
            cache_attn: list[torch.Tensor | None] = [None] * n_layers
            cache_mlp: list[torch.Tensor | None] = [None] * n_layers
            cap_hooks = []
            for li, lyr in enumerate(layers):
                def _mk_r(i):
                    def _h(_m, _in, out):
                        t = out[0] if isinstance(out, tuple) else out
                        cache_resid[i] = t[:, -1, :].detach().clone()
                    return _h
                def _mk_a(i):
                    def _h(_m, _in, out):
                        t = out[0] if isinstance(out, tuple) else out
                        cache_attn[i] = t[:, -1, :].detach().clone()
                    return _h
                def _mk_m(i):
                    def _h(_m, _in, out):
                        t = out[0] if isinstance(out, tuple) else out
                        cache_mlp[i] = t[:, -1, :].detach().clone()
                    return _h
                cap_hooks.append(lyr.register_forward_hook(_mk_r(li)))
                cap_hooks.append(lyr.self_attn.register_forward_hook(_mk_a(li)))
                cap_hooks.append(lyr.mlp.register_forward_hook(_mk_m(li)))

            out_luma = model(**prep(gr, cA_l, cB_l))
            for h in cap_hooks:
                h.remove()
            lg_luma = out_luma.logits[0, -1, ab_ids].float().cpu().numpy()
            patch_acc["Clean_BWGray_Luma"].append(int(int(np.argmax(lg_luma)) == tgt))
            patch_mrg["Clean_BWGray_Luma"].append(float(lg_luma[tgt] - lg_luma[wrong]))

            inp_rgb = prep(ir, cA_b, cB_b)
            for cname, mode, lrange in conditions:
                if mode == "clean":
                    continue
                phooks = []
                if mode is not None and lrange is not None:
                    for li in lrange:
                        if mode == "resid":
                            def _mk_pr(i):
                                def _h(_m, _in, out):
                                    if isinstance(out, tuple):
                                        t = out[0].clone()
                                        t[:, -1, :] = cache_resid[i]
                                        return (t,) + out[1:]
                                    t = out.clone(); t[:, -1, :] = cache_resid[i]; return t
                                return _h
                            phooks.append(layers[li].register_forward_hook(_mk_pr(li)))
                        if mode in ("attn", "both"):
                            def _mk_pa(i):
                                def _h(_m, _in, out):
                                    if isinstance(out, tuple):
                                        t = out[0].clone()
                                        t[:, -1, :] = cache_attn[i]
                                        return (t,) + out[1:]
                                    t = out.clone(); t[:, -1, :] = cache_attn[i]; return t
                                return _h
                            phooks.append(layers[li].self_attn.register_forward_hook(_mk_pa(li)))
                        if mode in ("mlp", "both"):
                            def _mk_pm(i):
                                def _h(_m, _in, out):
                                    if isinstance(out, tuple):
                                        t = out[0].clone()
                                        t[:, -1, :] = cache_mlp[i]
                                        return (t,) + out[1:]
                                    t = out.clone(); t[:, -1, :] = cache_mlp[i]; return t
                                return _h
                            phooks.append(layers[li].mlp.register_forward_hook(_mk_pm(li)))
                out_p = model(**inp_rgb)
                for h in phooks:
                    h.remove()
                lg = out_p.logits[0, -1, ab_ids].float().cpu().numpy()
                patch_acc[cname].append(int(int(np.argmax(lg)) == tgt))
                patch_mrg[cname].append(float(lg[tgt] - lg[wrong]))

    patch_summary = {}
    print("  [Part 2: Direct Causal Patching (BWGray_Luma -> BWGray_RGB, Last-Token Intervention)]", flush=True)
    for cname, _, _ in conditions:
        acc = float(np.mean(patch_acc[cname]))
        mrg = float(np.mean(patch_mrg[cname]))
        patch_summary[cname] = {"acc": round(acc, 4), "margin": round(mrg, 3)}
        print(f"    {cname:24s} | Acc={acc:.3f} | Margin={mrg:+.2f}", flush=True)

    out_p = args.out_dir / f"exp30a_causal_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({
        "model": Path(args.model_dir).name,
        "n_layers": n_layers,
        "n_pairs": len(pairs),
        "xdomain_probe": xdomain_probe,
        "causal_patching": patch_summary,
    }, indent=2), encoding="utf-8")
    print(f"Saved causal patching results to {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
