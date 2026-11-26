#!/usr/bin/env python3
"""
Exp 21B (P3 Exact Spatial-Causal Copy-Layer Curve & L0 Equal-Area Control!):

Completes Exp 21A's spatial-causal proof with TWO decisive controls:
  1. Equal-Token-Count L0 Controls:
     - `baseline`
     - `ablate_MOUTH_L0`  (mean-ablate K mouth visual tokens at L0)
     - `ablate_EYES_L0`   (mean-ablate eye visual tokens at L0 -> should INCREASE P(A) on dead eyes!)
     - `ablate_BG_K_L0`   (mean-ablate EXACT SAME K background visual tokens at L0 -> MUST BE ~0!)
  2. Layer-by-Layer Mouth-Copy Localization Curve:
     Sweeps `ablate_MOUTH` at 8 evenly spaced layers across depth [0 .. 0.85*L] to pinpoint
     the EXACT layer window where smiling-mouth visual tokens are copied into the decision stream!
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
import exp17c_scale48_5fold_and_unified_1pass as e17c
import exp21a_spatial_mouth_token_ablation as e21a


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--boxes-json", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if r["variant"] in e21a.VARIANTS7]
    boxes_map = json.loads(args.boxes_json.read_text(encoding="utf-8"))

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ids4 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS4]
    layers, _, _ = e17c._get_lm(model)
    nL = len(layers)
    sweep_layers = sorted(set([int(round(f * (nL - 1))) for f in [0.0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.85]]))

    state = {"target_pos": [], "bg_pool": [], "layer": -1}

    def mk_pre(li: int):
        def _fn(_m, inp):
            if li != state["layer"] or not state["target_pos"]:
                return inp
            h = inp[0].clone()
            bg = h[0, state["bg_pool"], :].mean(dim=0, keepdim=True)
            h[0, state["target_pos"], :] = bg
            return (h,) + inp[1:]
        return _fn

    hooks = [layers[li].register_forward_pre_hook(mk_pre(li)) for li in sweep_layers]

    # Conditions: baseline + 3 L0 spatial controls + mouth copy curve across sweep_layers
    conds = ["baseline", "ablate_BG_K_L0", "ablate_EYES_L0"] + [f"ablate_MOUTH_L{li}" for li in sweep_layers]
    res_pA = {c: {v: [] for v in e21a.VARIANTS7} for c in conds}
    res_acc = {c: {v: [] for v in e21a.VARIANTS7} for c in conds}

    for r in recs:
        cid, v = r["char_id"], r["variant"]
        boxes = boxes_map[cid]
        img = Image.open(r["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": e17c.DIRECT4_PROMPT}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[img], return_tensors="pt").to(model.device)

        vis_pos, gh, gw = e21a._find_visual_token_indices(inp["input_ids"], processor, args.model_dir, inp)
        mouth_pos = e21a._box_to_grid_positions(vis_pos, gh, gw, boxes["mouth"], pad_cells=1)
        eyes_pos = sorted(set(
            e21a._box_to_grid_positions(vis_pos, gh, gw, boxes["left_eye"], pad_cells=0)
            + e21a._box_to_grid_positions(vis_pos, gh, gw, boxes["right_eye"], pad_cells=0)
        ))
        face_pos = set(e21a._box_to_grid_positions(vis_pos, gh, gw, boxes["face"], pad_cells=0))
        bg_pool = [p for p in vis_pos if p not in face_pos]
        if len(bg_pool) < len(mouth_pos):
            bg_pool = vis_pos[: max(len(mouth_pos), 8)]
        bg_k = bg_pool[: len(mouth_pos)]
        state["bg_pool"] = bg_pool
        exp_l = e21a.EXPECTED7[v]

        for c in conds:
            if c == "baseline":
                state["layer"], state["target_pos"] = -1, []
            elif c == "ablate_BG_K_L0":
                state["layer"], state["target_pos"] = 0, bg_k
            elif c == "ablate_EYES_L0":
                state["layer"], state["target_pos"] = 0, eyes_pos
            else:
                li = int(c.split("_L")[-1])
                state["layer"], state["target_pos"] = li, mouth_pos

            lg = model(**inp).logits[0, -1, ids4].float().cpu().numpy()
            pr = np.exp(lg - lg.max()); pr /= pr.sum()
            pred = e17c.LABELS4[int(np.argmax(lg))]
            res_pA[c][v].append(float(pr[0]))
            res_acc[c][v].append(int(pred == exp_l))

    for hk in hooks:
        hk.remove()

    summary = {}
    print(f"\n=== EXP 21B MOUTH-COPY LAYER CURVE & L0 CONTROLS: {Path(args.model_dir).name} (nL={nL}) ===", flush=True)
    for c in conds:
        by_v = {v: {"acc": round(float(np.mean(res_acc[c][v])), 4), "pA": round(float(np.mean(res_pA[c][v])), 4)} for v in e21a.VARIANTS7}
        clean_pA = float(np.mean([by_v["orig"]["pA"], by_v["placebo_hairpin"]["pA"]]))
        conf_vars = [v for v in e21a.VARIANTS7 if v not in {"orig", "placebo_hairpin"}]
        conf_pA = float(np.mean([by_v[v]["pA"] for v in conf_vars]))
        conf_acc = float(np.mean([by_v[v]["acc"] for v in conf_vars]))
        macro7 = float(np.mean([by_v[v]["acc"] for v in e21a.VARIANTS7]))
        summary[c] = {"macro7": round(macro7, 4), "clean_pA": round(clean_pA, 4), "conf5_acc": round(conf_acc, 4), "conf5_pA": round(conf_pA, 4), "eye_pA": by_v["eye_highlight_off"]["pA"], "shd_pA": by_v["forehead_shadow_on"]["pA"], "swt_pA": by_v["sweat_drop_on"]["pA"]}
        print(f"  {c:20s} | Macro7={macro7:.3f} Clean_pA={clean_pA:.3f} Conf5_acc={conf_acc:.3f} Conf5_pA={conf_pA:.3f} (eye={by_v['eye_highlight_off']['pA']:.2f} shd={by_v['forehead_shadow_on']['pA']:.2f} swt={by_v['sweat_drop_on']['pA']:.2f})", flush=True)

    out_p = args.out_dir / f"exp21b_mouth_copy_curve_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "n_layers": nL, "sweep_layers": sweep_layers, "summary": summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
