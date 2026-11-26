#!/usr/bin/env python3
"""
Exp 29A (Track `P7` Mechanistic Interpretability — Layerwise Logit-Lens, Attention vs. MLP Direct Logit
Attribution, and Linear Identity Probing of the 2D Anime Color-Palette Shortcut!):

Answers the central mechanistic question linking our two pillars (`P3` & `P7`):
  When `Qwen3-VL-32B` / `Qwen2.5-VL-32B` / `Qwen2.5-VL-7B` / `InternVL3-14B` fail on Color->B&W Manga
  (`XPose_BWGray`) or `XPose_HueShift135`, WHERE does the Color-Palette Shortcut happen?
    - Hypothesis H1 (Late-Layer Readout Override, like P3): Mid layers compute the true structural
      character match (high linear-probe accuracy / positive mid-layer margin), and late Attn/MLP
      overwrites it with the color-matching distractor.
    - Hypothesis H2 (Early Visual-Comparison Hijack): Color-palette similarity dominates cross-image
      matching early, preventing Raw-RGB mid layers from forming the structural match — which is WHY
      input-level Luminance Canonicalization / Adaptive Chroma Gating unlocks +35% to +51%!

Measures across ALL layers l = 0..L on n = 24 balanced deconfounded pairs (48 queries/regime):
  1. Logit-Lens `Acc(l)` & `Signed Margin = logit(correct) - logit(wrong)` across 4 regimes:
     `SameColor_RGB`, `BWGray_RGB`, `HueShift_RGB`, and `BWGray_Luma` (control).
  2. Exact Attention (`self_attn`) vs. MLP (`mlp`) Direct Logit Attribution (DLA) per layer l.
  3. 4-Fold Pair-Disjoint Linear Probe accuracy on `h_l` (predicting ground-truth `A vs B` directly
     from the final-token residual stream under `Raw RGB` vs `Luma`).
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


def _ridge_probe_cv(H: np.ndarray, y: np.ndarray, pair_ids: np.ndarray, n_folds: int = 4, l2: float = 10.0) -> float:
    """Pair-disjoint Ridge/logistic linear probe accuracy predicting y in {0, 1} from residual stream H [N, D]."""
    uniq_pairs = np.unique(pair_ids)
    correct = 0
    total = 0
    for f in range(n_folds):
        te_pairs = set(uniq_pairs[f::n_folds].tolist())
        te_mask = np.array([pid in te_pairs for pid in pair_ids], dtype=bool)
        tr_mask = ~te_mask
        Xtr, ytr = H[tr_mask], y[tr_mask]
        Xte, yte = H[te_mask], y[te_mask]
        mu = Xtr.mean(axis=0, keepdims=True)
        Xtr_c = Xtr - mu
        Xte_c = Xte - mu
        # Dual ridge regression: w = X^T (X X^T + l2 I)^-1 (2y - 1)
        K = Xtr_c @ Xtr_c.T + l2 * np.eye(Xtr_c.shape[0], dtype=np.float32)
        alpha = np.linalg.solve(K, (2.0 * ytr - 1.0).astype(np.float32))
        scores = (Xte_c @ Xtr_c.T) @ alpha
        preds = (scores > 0.0).astype(np.int64)
        correct += int(np.sum(preds == yte))
        total += int(len(yte))
    return float(correct / max(1, total))


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
    layers, norm, lm_head = e14a._get_lm(model)
    n_layers = len(layers)
    ab_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in ["A", "B"]]
    W_ab = lm_head.weight[ab_ids].detach().float().cpu()  # [2, D]

    # Hook collectors for last-token attn & mlp outputs
    attn_outs: list[torch.Tensor | None] = [None] * n_layers
    mlp_outs: list[torch.Tensor | None] = [None] * n_layers
    hooks = []
    for li, layer in enumerate(layers):
        def _mk_attn(idx: int):
            def _hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                attn_outs[idx] = t[0, -1].detach().float().cpu()
            return _hook
        def _mk_mlp(idx: int):
            def _hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                mlp_outs[idx] = t[0, -1].detach().float().cpu()
            return _hook
        hooks.append(layer.self_attn.register_forward_hook(_mk_attn(li)))
        hooks.append(layer.mlp.register_forward_hook(_mk_mlp(li)))

    def forward_trace(ref: Image.Image, cA: Image.Image, cB: Image.Image):
        m = [{"role": "user", "content": [
            {"type": "image", "image": ref},
            {"type": "image", "image": cA},
            {"type": "image", "image": cB},
            {"type": "text", "text": e25a.PROMPT_REID},
        ]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[ref, cA, cB], return_tensors="pt").to(model.device)
        out = model(**inp, output_hidden_states=True)
        # Hidden states: tuple of length n_layers + 1 (emb + each layer output)
        hs = [h[0, -1].detach().float().cpu() for h in out.hidden_states]
        # Exact final RMSNorm scale factor for DLA
        h_final = out.hidden_states[-1][0, -1].detach().float()
        if norm is not None:
            h_normed = norm(h_final.to(model.dtype)).float()
            # Effective unembedding direction after norm weight:
            w_norm = norm.weight.detach().float().cpu()
            rms = float(torch.sqrt(torch.mean(h_final ** 2) + getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))).cpu())
            # Exact logit-lens per layer using norm + lm_head
            ll_logits = []
            for h_cpu in hs:
                hn = norm(h_cpu.to(model.device, dtype=model.dtype)).float()
                ll_logits.append((hn @ lm_head.weight[ab_ids].float().T).cpu().numpy())
            eff_W_ab = W_ab * (w_norm.unsqueeze(0) / max(rms, 1e-6))
        else:
            ll_logits = [(h_cpu @ W_ab.T).numpy() for h_cpu in hs]
            eff_W_ab = W_ab
        final_lg = out.logits[0, -1, ab_ids].detach().float().cpu().numpy()
        return {
            "hs": hs,
            "ll_logits": np.stack(ll_logits, axis=0),  # [L+1, 2]
            "attn": [a.clone() for a in attn_outs],
            "mlp": [m_.clone() for m_ in mlp_outs],
            "eff_W_ab": eff_W_ab,
            "final_lg": final_lg,
        }

    regimes = [
        ("SameColor_RGB", lambda r, p, n: (r, p, n)),
        ("BWGray_RGB", lambda r, p, n: (r, e26b._clean_luma_gray(p), n)),
        ("HueShift_RGB", lambda r, p, n: (r, e25a._to_hue_shifted(p, 95), n)),
        ("BWGray_Luma", lambda r, p, n: (e26b._clean_luma_gray(r), e26b._clean_luma_gray(p), e26b._clean_luma_gray(n))),
    ]

    summary = {"model": Path(args.model_dir).name, "n_layers": n_layers, "n_pairs": len(pairs), "regimes": {}}
    print(f"\n=== EXP 29A MECHANISTIC LAYER LOCALIZATION: {Path(args.model_dir).name} (L={n_layers}, n={len(pairs)} pairs) ===", flush=True)

    for rname, rfn in regimes:
        ll_acc = np.zeros(n_layers + 1, dtype=np.float64)
        ll_margin = np.zeros(n_layers + 1, dtype=np.float64)
        dla_attn = np.zeros(n_layers, dtype=np.float64)
        dla_mlp = np.zeros(n_layers, dtype=np.float64)
        final_acc = []
        # Store residual states for linear probe at every 2nd layer (or every layer if L<=32)
        probe_layers = list(range(0, n_layers + 1, 2 if n_layers > 36 else 1))
        if n_layers not in probe_layers:
            probe_layers.append(n_layers)
        H_buf = {li: [] for li in probe_layers}
        y_buf = []
        pid_buf = []

        n_q = 0
        for pi, (r_pos, _r_neg, xp_pos, xp_neg) in enumerate(pairs):
            im_ref0 = Image.open(r_pos["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
            im_pos0 = Image.open(xp_pos).convert("RGB").resize((448, 448), Image.LANCZOS)
            im_neg0 = Image.open(xp_neg).convert("RGB").resize((448, 448), Image.LANCZOS)
            im_ref, im_pos, im_neg = rfn(im_ref0, im_pos0, im_neg0)

            for (cA, cB, target_idx) in [(im_pos, im_neg, 0), (im_neg, im_pos, 1)]:
                tr = forward_trace(im_ref, cA, cB)
                wrong_idx = 1 - target_idx
                # Logit lens
                lgs = tr["ll_logits"]  # [L+1, 2]
                ll_acc += (np.argmax(lgs, axis=1) == target_idx).astype(np.float64)
                ll_margin += (lgs[:, target_idx] - lgs[:, wrong_idx])
                final_acc.append(int(int(np.argmax(tr["final_lg"])) == target_idx))
                # DLA onto (W[correct] - W[wrong])
                dir_cw = tr["eff_W_ab"][target_idx] - tr["eff_W_ab"][wrong_idx]  # [D]
                for li in range(n_layers):
                    dla_attn[li] += float(torch.dot(tr["attn"][li], dir_cw))
                    dla_mlp[li] += float(torch.dot(tr["mlp"][li], dir_cw))
                for li in probe_layers:
                    H_buf[li].append(tr["hs"][li].numpy())
                y_buf.append(target_idx)
                pid_buf.append(pi)
                n_q += 1

        ll_acc /= n_q
        ll_margin /= n_q
        dla_attn /= n_q
        dla_mlp /= n_q

        y_arr = np.array(y_buf, dtype=np.int64)
        pid_arr = np.array(pid_buf, dtype=np.int64)
        probe_curve = {}
        for li in probe_layers:
            H_mat = np.stack(H_buf[li], axis=0).astype(np.float32)
            probe_curve[str(li)] = round(_ridge_probe_cv(H_mat, y_arr, pid_arr, n_folds=4, l2=10.0), 4)

        best_probe_l = max(probe_curve.items(), key=lambda kv: kv[1])
        best_ll_l = int(np.argmax(ll_acc))
        # Split DLA into early (0..L/2) vs late (L/2..L)
        mid = n_layers // 2
        sum_attn_late = float(np.sum(dla_attn[mid:]))
        sum_mlp_late = float(np.sum(dla_mlp[mid:]))
        top_neg_attn = sorted([(int(i), round(float(dla_attn[i]), 3)) for i in range(n_layers)], key=lambda x: x[1])[:5]
        top_neg_mlp = sorted([(int(i), round(float(dla_mlp[i]), 3)) for i in range(n_layers)], key=lambda x: x[1])[:5]

        summary["regimes"][rname] = {
            "final_acc": round(float(np.mean(final_acc)), 4),
            "peak_logitlens_layer": best_ll_l,
            "peak_logitlens_acc": round(float(ll_acc[best_ll_l]), 4),
            "peak_probe_layer": int(best_probe_l[0]),
            "peak_probe_acc": float(best_probe_l[1]),
            "late_dla_attn_sum": round(sum_attn_late, 3),
            "late_dla_mlp_sum": round(sum_mlp_late, 3),
            "top5_neg_attn_layers": top_neg_attn,
            "top5_neg_mlp_layers": top_neg_mlp,
            "ll_acc": [round(float(x), 4) for x in ll_acc],
            "ll_margin": [round(float(x), 4) for x in ll_margin],
            "dla_attn": [round(float(x), 4) for x in dla_attn],
            "dla_mlp": [round(float(x), 4) for x in dla_mlp],
            "probe_acc_by_layer": probe_curve,
        }
        print(
            f"  {rname:14s} | FinalAcc={np.mean(final_acc):.3f} | PeakLL=L{best_ll_l}({ll_acc[best_ll_l]:.3f}) | "
            f"PeakProbe=L{best_probe_l[0]}({best_probe_l[1]:.3f}) | LateDLA(Attn={sum_attn_late:+.2f}, MLP={sum_mlp_late:+.2f})",
            flush=True,
        )

    for h in hooks:
        h.remove()

    out_p = args.out_dir / f"exp29a_mech_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved mechanistic trace to {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
