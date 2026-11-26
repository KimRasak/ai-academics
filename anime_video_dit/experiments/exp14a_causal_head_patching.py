#!/usr/bin/env python3
"""
Exp 14A (P3 Causal Mechanistic Verification — Direct Logit Attribution + Causal Patching):
  Moves from correlational probing (Exp 11) to STRICT CAUSAL INTERVENTION:

  Part 1 — Exact Attention vs. MLP Direct Logit Attribution (DLA) across ALL layers 0..L-1:
    Decomposes the final logit margin `logit(A) - mean(logit(B,C,D))` into exact additive
    contributions from every layer's Attention block (`self_attn`) and MLP block (`mlp`)
    (normalized by the final RMSNorm scale factor).
    Directly answers: "Is the late-layer Biological-Smile Override driven by Attention
    (re-attending to smiling-mouth vision tokens) or by MLPs (stored biological facial prior)?"

  Part 2 — Exact Causal Activation Patching (Clean `orig` <-> Conflict Minimal Pairs):
    Because our counterfactuals are PIXEL-IDENTICAL on the smiling mouth and differ ONLY
    on the non-biological anime symbol, patching the last-token residual stream from
    `conflict` into `orig` (and `orig` into `conflict`) at layer l isolates the exact
    causal mediation window [l_start, l_end] where non-biological symbol information
    flows into the decision token!

  Part 3 — Exact Per-Head Attention Decomposition & Causal Knock-Out:
    At the top biological-override attention layers, decomposes `o_proj` into individual
    attention heads h = 0..H-1 to find the exact "Biological-Smile Override Heads",
    then causally ablates ONLY the top-K heads (selected strictly on train chars!)
    on held-out unseen chars & UNSEEN symbols!
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
EXPECTED = {
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


def _encode(processor, img: Image.Image, device):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DIRECT_PROMPT}]}]
    txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(text=[txt], images=[img], return_tensors="pt").to(device)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Index by (char_id, variant)
    by_cv = {(r["char_id"], r["variant"]): r for r in records if r["variant"] in EXPECTED}
    chars = sorted({r["char_id"] for r in records})
    tr_chars, te_chars = chars[:10], chars[10:]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    abcd_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in LABELS]

    layers, final_norm, lm_head = _get_lm(model)
    nL = len(layers)
    # Biological-override unembedding direction: w_A - mean(w_B, w_C, w_D)
    W_abcd = lm_head.weight[abcd_ids].detach().float()  # [4, D]
    w_bio = W_abcd[0] - W_abcd[1:].mean(dim=0)          # [D]
    norm_w = final_norm.weight.detach().float() if (final_norm is not None and hasattr(final_norm, "weight")) else torch.ones_like(w_bio)
    w_bio_eff = (w_bio * norm_w).to(model.device)       # effective direction pre-RMSNorm (divide by final rms per sample)

    # =========================================================================
    # PART 1: Exact Attention vs. MLP Direct Logit Attribution across ALL layers
    # =========================================================================
    print(f"=== PART 1: Attention vs. MLP Direct Logit Attribution ({Path(args.model_dir).name}, nL={nL}) ===", flush=True)
    attn_dla = {v: np.zeros(nL, dtype=np.float64) for v in ["orig", "forehead_shadow_on", "sweat_drop_on", "eye_highlight_off", "blush_lines_on"]}
    mlp_dla = {v: np.zeros(nL, dtype=np.float64) for v in attn_dla}
    counts = {v: 0 for v in attn_dla}

    cap_attn: dict[int, torch.Tensor] = {}
    cap_mlp: dict[int, torch.Tensor] = {}
    cap_final: dict[str, torch.Tensor] = {}

    handles = []
    for l in range(nL):
        def mk_a(idx):
            def fn(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                cap_attn[idx] = h[0, -1, :].detach().float()
            return fn
        def mk_m(idx):
            def fn(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                cap_mlp[idx] = h[0, -1, :].detach().float()
            return fn
        handles.append(layers[l].self_attn.register_forward_hook(mk_a(l)))
        handles.append(layers[l].mlp.register_forward_hook(mk_m(l)))

    def cap_last(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap_final["h"] = h[0, -1, :].detach().float()
    handles.append(layers[-1].register_forward_hook(cap_last))

    try:
        for cid in chars:
            for v in attn_dla:
                if (cid, v) not in by_cv:
                    continue
                img = Image.open(by_cv[(cid, v)]["image_path"]).convert("RGB")
                inp = _encode(processor, img, model.device)
                _ = model(**inp)
                rms = torch.sqrt(torch.mean(cap_final["h"] ** 2) + 1e-6)
                for l in range(nL):
                    attn_dla[v][l] += float(torch.dot(cap_attn[l], w_bio_eff) / rms)
                    mlp_dla[v][l] += float(torch.dot(cap_mlp[l], w_bio_eff) / rms)
                counts[v] += 1
    finally:
        for h in handles:
            h.remove()

    for v in attn_dla:
        attn_dla[v] /= max(1, counts[v])
        mlp_dla[v] /= max(1, counts[v])

    # Summarize total Attn vs MLP late-layer (last 25% layers) biological-override push on conflicts!
    late_start = int(nL * 0.72)
    dla_summary = {}
    for v in attn_dla:
        dla_summary[v] = {
            "total_attn_all_layers": round(float(attn_dla[v].sum()), 3),
            "total_mlp_all_layers": round(float(mlp_dla[v].sum()), 3),
            "late_attn_sum": round(float(attn_dla[v][late_start:].sum()), 3),
            "late_mlp_sum": round(float(mlp_dla[v][late_start:].sum()), 3),
            "top3_attn_override_layers": [int(x) for x in np.argsort(-attn_dla[v])[:3]],
            "top3_mlp_override_layers": [int(x) for x in np.argsort(-mlp_dla[v])[:3]],
            "per_layer_attn": [round(float(x), 3) for x in attn_dla[v]],
            "per_layer_mlp": [round(float(x), 3) for x in mlp_dla[v]],
        }
        print(
            f"  {v:20s} | LATE[{late_start}..{nL-1}] Attn={dla_summary[v]['late_attn_sum']:+6.2f}  MLP={dla_summary[v]['late_mlp_sum']:+6.2f} "
            f"| TopAttn={dla_summary[v]['top3_attn_override_layers']} TopMLP={dla_summary[v]['top3_mlp_override_layers']}",
            flush=True,
        )

    # =========================================================================
    # PART 2: Exact Causal Activation Patching (Patching Conflict <-> Orig!)
    # =========================================================================
    print("=== PART 2: Causal Layer-by-Layer Residual Patching (Conflict -> Orig) ===", flush=True)
    patch_layers = sorted(set([int(round(i * (nL - 1) / 9.0)) for i in range(10)]))
    patch_results: dict[str, dict[int, float]] = {}

    # Run on 6 held-out characters for clean causal mediation measurement
    for v in ["forehead_shadow_on", "sweat_drop_on", "eye_highlight_off", "blush_lines_on"]:
        deltas_by_l = {l: [] for l in patch_layers}
        for cid in te_chars:
            img_o = Image.open(by_cv[(cid, "orig")]["image_path"]).convert("RGB")
            img_c = Image.open(by_cv[(cid, v)]["image_path"]).convert("RGB")
            in_o = _encode(processor, img_o, model.device)
            in_c = _encode(processor, img_c, model.device)

            # Capture conflict residual at all patch_layers
            c_res: dict[int, torch.Tensor] = {}
            hs = []
            for l in patch_layers:
                def mk_c(idx):
                    def fn(_m, _i, out):
                        h = out[0] if isinstance(out, tuple) else out
                        c_res[idx] = h[:, -1:, :].detach().clone()
                    return fn
                hs.append(layers[l].register_forward_hook(mk_c(l)))
            lg_c = model(**in_c).logits[0, -1, abcd_ids].float().cpu().numpy()
            for h in hs:
                h.remove()

            # Baseline orig margin = logit(A) - logit(gt)
            lg_o = model(**in_o).logits[0, -1, abcd_ids].float().cpu().numpy()
            gt_idx = LABELS.index(EXPECTED[v])
            pA_orig = float(np.exp(lg_o[0] - np.max(lg_o)) / np.sum(np.exp(lg_o - np.max(lg_o))))
            margin_orig = float(lg_o[gt_idx] - lg_o[0])

            # Patch conflict's last-token residual at layer l into orig forward pass!
            for l in patch_layers:
                def mk_patch(saved):
                    def fn(_m, _i, out):
                        if isinstance(out, tuple):
                            h = out[0].clone()
                            h[:, -1:, :] = saved.to(h.dtype)
                            return (h,) + out[1:]
                        h = out.clone()
                        h[:, -1:, :] = saved.to(h.dtype)
                        return h
                    return fn
                ph = layers[l].register_forward_hook(mk_patch(c_res[l]))
                lg_p = model(**in_o).logits[0, -1, abcd_ids].float().cpu().numpy()
                ph.remove()
                margin_patch = float(lg_p[gt_idx] - lg_p[0])
                # Positive delta = patching layer l causally shifts prediction toward TRUE non-bio affect gt!
                deltas_by_l[l].append(margin_patch - margin_orig)

        patch_results[v] = {l: round(float(np.mean(deltas_by_l[l])), 4) for l in patch_layers}
        print(f"  Causal Patch (conflict->orig) {v:20s}: " + " ".join(f"L{l}:{patch_results[v][l]:+.2f}" for l in patch_layers), flush=True)

    # =========================================================================
    # PART 3: Per-Head Attention Decomposition & Causal Knock-Out at Top Override Layer
    # =========================================================================
    # Find the single strongest late Attention layer pushing logit(A) on TRAIN conflicts
    seen_avg_attn = 0.5 * (attn_dla["forehead_shadow_on"] + attn_dla["sweat_drop_on"])
    top_attn_l = int(late_start + np.argmax(seen_avg_attn[late_start:]))
    attn_mod = layers[top_attn_l].self_attn
    o_proj = attn_mod.o_proj
    n_heads = getattr(attn_mod, "num_heads", None) or getattr(model.config, "num_attention_heads", None)
    if n_heads is None and hasattr(model.config, "text_config"):
        n_heads = model.config.text_config.num_attention_heads
    head_dim = o_proj.in_features // int(n_heads)
    print(f"=== PART 3: Per-Head Decomposition at Top Override Attn Layer L{top_attn_l} (n_heads={n_heads}, head_dim={head_dim}) ===", flush=True)

    # Measure each head's exact DLA contribution on TRAIN chars ({forehead_shadow, sweat_drop})
    head_push = np.zeros(int(n_heads), dtype=np.float64)
    head_clean = np.zeros(int(n_heads), dtype=np.float64)
    cap_oin = {}

    def pre_o(_m, inp):
        cap_oin["x"] = inp[0][0, -1, :].detach().float()  # [H * head_dim]

    ho = o_proj.register_forward_pre_hook(pre_o)
    hf = layers[-1].register_forward_hook(cap_last)
    W_o = o_proj.weight.detach().float()  # [D_out, H * head_dim]
    try:
        for cid in tr_chars:
            for v, acc_arr in [("orig", head_clean), ("forehead_shadow_on", head_push), ("sweat_drop_on", head_push)]:
                img = Image.open(by_cv[(cid, v)]["image_path"]).convert("RGB")
                inp = _encode(processor, img, model.device)
                _ = model(**inp)
                rms = torch.sqrt(torch.mean(cap_final["h"] ** 2) + 1e-6)
                x_h = cap_oin["x"].view(int(n_heads), head_dim)
                for hi in range(int(n_heads)):
                    W_hi = W_o[:, hi * head_dim : (hi + 1) * head_dim]
                    out_hi = W_hi @ x_h[hi]
                    acc_arr[hi] += float(torch.dot(out_hi, w_bio_eff) / rms)
    finally:
        ho.remove()
        hf.remove()

    head_push /= (2.0 * len(tr_chars))
    head_clean /= float(len(tr_chars))
    top_heads = [int(x) for x in np.argsort(-head_push)[:8]]
    print("  Top-8 Bio-Override Heads (L%d): %s" % (top_attn_l, [(h, round(float(head_push[h]), 3), round(float(head_clean[h]), 3)) for h in top_heads]), flush=True)

    # Test causal zero-ablation of Top-K Bio-Override Heads at top_attn_l on HELD-OUT TEST chars!
    knockout_res = {}
    for K in [0, 2, 4, 8]:
        ablate_set = set(top_heads[:K])
        def mk_ko(aset):
            def fn(_m, inp):
                x = inp[0].clone()
                if aset:
                    xv = x.view(x.shape[0], x.shape[1], int(n_heads), head_dim)
                    for hi in aset:
                        xv[:, -1, hi, :] = 0.0
                    x = xv.view_as(x)
                return (x,)
            return fn
        hk = o_proj.register_forward_pre_hook(mk_ko(ablate_set)) if K > 0 else None
        by_v: dict[str, list[bool]] = {}
        by_pA: dict[str, list[float]] = {}
        try:
            for cid in te_chars:
                for v in EXPECTED:
                    img = Image.open(by_cv[(cid, v)]["image_path"]).convert("RGB")
                    inp = _encode(processor, img, model.device)
                    lg = model(**inp).logits[0, -1, abcd_ids].float().cpu().numpy()
                    ex = np.exp(lg - np.max(lg)); p = ex / ex.sum()
                    pred = LABELS[int(np.argmax(p))]
                    by_v.setdefault(v, []).append(pred == EXPECTED[v])
                    by_pA.setdefault(v, []).append(float(p[0]))
        finally:
            if hk is not None:
                hk.remove()
        sv = {v: round(float(np.mean(xs)), 4) for v, xs in sorted(by_v.items())}
        spA = {v: round(float(np.mean(xs)), 4) for v, xs in sorted(by_pA.items())}
        knockout_res[f"KO_top{K}_heads"] = {
            "macro7": round(float(np.mean(list(sv.values()))), 4),
            "clean": round(float(np.mean([sv["orig"], sv["placebo_hairpin"]])), 4),
            "conflicts": round(float(np.mean([sv[k] for k in ["eye_highlight_off", "forehead_shadow_on", "dead_eyes_and_shadow", "blush_lines_on", "sweat_drop_on"]])), 4),
            "by_variant": sv,
            "mean_pA": spA,
        }
        print(f"  Causal Knock-Out K={K} (L{top_attn_l}): macro7={knockout_res[f'KO_top{K}_heads']['macro7']:.3f} clean={knockout_res[f'KO_top{K}_heads']['clean']:.3f} conf={knockout_res[f'KO_top{K}_heads']['conflicts']:.3f} mean_pA_eye={spA['eye_highlight_off']:.3f} mean_pA_shd={spA['forehead_shadow_on']:.3f}", flush=True)

    out_path = args.out_dir / f"exp14a_causal_{Path(args.model_dir).name}.json"
    out_path.write_text(
        json.dumps(
            {
                "model": Path(args.model_dir).name,
                "n_layers": nL,
                "late_start": late_start,
                "top_attn_layer": top_attn_l,
                "top8_bio_override_heads": [(h, round(float(head_push[h]), 3), round(float(head_clean[h]), 3)) for h in top_heads],
                "part1_dla_attn_vs_mlp": dla_summary,
                "part2_causal_patching_conflict_to_orig": patch_results,
                "part3_causal_head_knockout": knockout_res,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved causal verification -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
