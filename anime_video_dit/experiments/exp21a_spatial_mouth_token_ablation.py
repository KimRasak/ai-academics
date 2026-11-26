#!/usr/bin/env python3
"""
Exp 21A (P3 Spatial-Causal Mechanistic Proof — Late-Layer Mouth Visual Token Mean-Ablation
         vs. Eye Visual Tokens vs. Equal-Area Background Control!):

Provides smoking-gun spatial-causal localization of the Biological-Smile Override circuit:
  Using our ground-truth facial bounding boxes (`mouth`, `left_eye`, `right_eye`), maps exact
  2D spatial coordinates to the VLM's visual token grid (exact raster grid for 448x448 input)
  and performs controlled mean-ablation of ONLY those K visual token residual states at:
    - `baseline`           (no ablation)
    - `ablate_MOUTH_L0`    (early pre-fusion ablation of mouth visual tokens at Layer 0)
    - `ablate_MOUTH_Llate` (LATE-ONLY ablation at l_surg = 0.72*L — layers 0..l_surg-1 UNTOUCHED!)
    - `ablate_EYES_Llate`  (LATE-ONLY ablation of eye visual tokens at l_surg)
    - `ablate_BG_K_Llate`  (EQUAL-TOKEN-COUNT K background/border visual tokens at l_surg)

Tests on 16 full-color TV-anime characters (fc_00..fc_15) x 7 variants (112 images):
  If ablating ONLY the K mouth visual tokens (either at L0 or strictly at L_late) causes
  false-cheerful P(A) to drop and non-biological conflict accuracy to surge — while ablating
  an equal number K of background tokens causes ~0 change — this spatially localizes the
  biological-override source directly to the smiling-mouth visual token stream!
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp17c_scale48_5fold_and_unified_1pass as e17c

VARIANTS7 = [
    "orig",
    "placebo_hairpin",
    "eye_highlight_off",
    "forehead_shadow_on",
    "dead_eyes_and_shadow",
    "blush_lines_on",
    "sweat_drop_on",
]
EXPECTED7 = {
    "orig": "A",
    "placebo_hairpin": "A",
    "eye_highlight_off": "B",
    "forehead_shadow_on": "B",
    "dead_eyes_and_shadow": "B",
    "blush_lines_on": "C",
    "sweat_drop_on": "D",
}


def _find_visual_token_indices(input_ids: torch.Tensor, processor, model_name: str, inputs: dict) -> tuple[list[int], int, int]:
    """
    Locate exact sequence positions of the primary visual patch grid tokens + its (gh, gw) 2D grid shape.
    """
    ids = input_ids[0].tolist()
    # Identify image placeholder token id(s)
    cand_toks = []
    for attr in ("image_token_id", "img_context_token_id"):
        if hasattr(processor, attr) and getattr(processor, attr) is not None:
            cand_toks.append(int(getattr(processor, attr)))
    tok = processor.tokenizer
    for special in ("<|image_pad|>", "<image>", "<IMG_CONTEXT>"):
        tid = tok.convert_tokens_to_ids(special)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            cand_toks.append(tid)

    best_pos = []
    for tid in dict.fromkeys(cand_toks):
        pos = [i for i, x in enumerate(ids) if x == tid]
        if len(pos) > len(best_pos):
            best_pos = pos

    assert len(best_pos) >= 16, f"Could not find visual tokens in {model_name} (found {len(best_pos)})"

    # Determine exact (gh, gw) grid shape
    if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
        thw = inputs["image_grid_thw"][0].tolist()
        gh, gw = int(thw[1]) // 2, int(thw[2]) // 2
        if gh * gw == len(best_pos):
            return best_pos, gh, gw

    # Square grid (e.g. InternVL3 448x448 -> 256 tokens = 16x16; if multi-tile, use LAST 256 thumbnail tile!)
    side = int(round(math.sqrt(len(best_pos))))
    if side * side == len(best_pos):
        return best_pos, side, side

    # InternVL3 dynamic tiling: each tile is 256 tokens (16x16), and the LAST 256 tokens are the global thumbnail!
    if len(best_pos) % 256 == 0:
        return best_pos[-256:], 16, 16

    # LLaVA-OneVision: first 729 tokens (27x27) are the global base image!
    if len(best_pos) >= 729:
        return best_pos[:729], 27, 27

    side = int(math.sqrt(len(best_pos)))
    return best_pos[: side * side], side, side


def _box_to_grid_positions(vis_pos: list[int], gh: int, gw: int, box_1000: list[int], pad_cells: int = 1) -> list[int]:
    x1 = max(0, int(math.floor(box_1000[0] / 1000.0 * gw)) - pad_cells)
    y1 = max(0, int(math.floor(box_1000[1] / 1000.0 * gh)) - pad_cells)
    x2 = min(gw - 1, int(math.ceil(box_1000[2] / 1000.0 * gw)) - 1 + pad_cells)
    y2 = min(gh - 1, int(math.ceil(box_1000[3] / 1000.0 * gh)) - 1 + pad_cells)
    out = []
    for r in range(y1, y2 + 1):
        for c in range(x1, x2 + 1):
            out.append(vis_pos[r * gw + c])
    return sorted(set(out))


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
    recs = [r for r in recs if r["variant"] in VARIANTS7]
    boxes_map = json.loads(args.boxes_json.read_text(encoding="utf-8"))

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ids4 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS4]
    layers, _, _ = e17c._get_lm(model)
    nL = len(layers)
    l_late = int(nL * 0.72)

    state = {"target_pos": [], "bg_pool": [], "layer": -1}

    def mk_pre_hook(li: int):
        def _fn(_m, inp):
            if li != state["layer"] or not state["target_pos"]:
                return inp
            h = inp[0].clone()
            bg = h[0, state["bg_pool"], :].mean(dim=0, keepdim=True)
            h[0, state["target_pos"], :] = bg
            return (h,) + inp[1:]
        return _fn

    hooks = [
        layers[0].register_forward_pre_hook(mk_pre_hook(0)),
        layers[l_late].register_forward_pre_hook(mk_pre_hook(l_late)),
    ]

    conditions = ["baseline", "ablate_MOUTH_L0", "ablate_MOUTH_Llate", "ablate_EYES_Llate", "ablate_BG_K_Llate"]
    res_pA = {c: {v: [] for v in VARIANTS7} for c in conditions}
    res_acc = {c: {v: [] for v in VARIANTS7} for c in conditions}
    k_counts = []

    for r in recs:
        cid = r["char_id"]
        v = r["variant"]
        boxes = boxes_map[cid]
        img = Image.open(r["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": e17c.DIRECT4_PROMPT}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[img], return_tensors="pt").to(model.device)

        vis_pos, gh, gw = _find_visual_token_indices(inp["input_ids"], processor, args.model_dir, inp)
        mouth_pos = _box_to_grid_positions(vis_pos, gh, gw, boxes["mouth"], pad_cells=1)
        eyes_pos = sorted(set(
            _box_to_grid_positions(vis_pos, gh, gw, boxes["left_eye"], pad_cells=0)
            + _box_to_grid_positions(vis_pos, gh, gw, boxes["right_eye"], pad_cells=0)
        ))
        face_pos = set(_box_to_grid_positions(vis_pos, gh, gw, boxes["face"], pad_cells=0))
        bg_pool = [p for p in vis_pos if p not in face_pos]
        if len(bg_pool) < len(mouth_pos):
            bg_pool = vis_pos[: max(len(mouth_pos), 8)]
        bg_k = bg_pool[: len(mouth_pos)]
        k_counts.append(len(mouth_pos))

        state["bg_pool"] = bg_pool
        exp_l = EXPECTED7[v]

        for cond in conditions:
            if cond == "baseline":
                state["layer"], state["target_pos"] = -1, []
            elif cond == "ablate_MOUTH_L0":
                state["layer"], state["target_pos"] = 0, mouth_pos
            elif cond == "ablate_MOUTH_Llate":
                state["layer"], state["target_pos"] = l_late, mouth_pos
            elif cond == "ablate_EYES_Llate":
                state["layer"], state["target_pos"] = l_late, eyes_pos
            elif cond == "ablate_BG_K_Llate":
                state["layer"], state["target_pos"] = l_late, bg_k

            out = model(**inp)
            lg = out.logits[0, -1, ids4].float().cpu().numpy()
            pr = np.exp(lg - lg.max()); pr /= pr.sum()
            pred = e17c.LABELS4[int(np.argmax(lg))]
            res_pA[cond][v].append(float(pr[0]))
            res_acc[cond][v].append(int(pred == exp_l))

    for hk in hooks:
        hk.remove()

    summary = {}
    print(f"\n=== EXP 21A SPATIAL-CAUSAL MOUTH VISUAL TOKEN ABLATION: {Path(args.model_dir).name} (grid={gh}x{gw}, mean K_mouth={np.mean(k_counts):.1f}, L_late={l_late}/{nL}) ===", flush=True)
    for cond in conditions:
        by_v = {
            v: {"acc": round(float(np.mean(res_acc[cond][v])), 4), "pA": round(float(np.mean(res_pA[cond][v])), 4)}
            for v in VARIANTS7
        }
        clean_acc = float(np.mean([by_v["orig"]["acc"], by_v["placebo_hairpin"]["acc"]]))
        conf_vars = [v for v in VARIANTS7 if v not in {"orig", "placebo_hairpin"}]
        conf_acc = float(np.mean([by_v[v]["acc"] for v in conf_vars]))
        conf_pA = float(np.mean([by_v[v]["pA"] for v in conf_vars]))
        clean_pA = float(np.mean([by_v["orig"]["pA"], by_v["placebo_hairpin"]["pA"]]))
        macro7 = float(np.mean([by_v[v]["acc"] for v in VARIANTS7]))
        summary[cond] = {
            "macro7": round(macro7, 4),
            "clean_acc": round(clean_acc, 4),
            "conf5_acc": round(conf_acc, 4),
            "clean_pA": round(clean_pA, 4),
            "conf5_pA": round(conf_pA, 4),
            "by_variant": by_v,
        }
        print(
            f"  {cond:20s} | Macro7={macro7:.3f} Clean={clean_acc:.3f}(pA={clean_pA:.3f}) Conf5={conf_acc:.3f}(pA={conf_pA:.3f}) | "
            f"eye={by_v['eye_highlight_off']['pA']:.2f} shd={by_v['forehead_shadow_on']['pA']:.2f} d+s={by_v['dead_eyes_and_shadow']['pA']:.2f} swt={by_v['sweat_drop_on']['pA']:.2f}",
            flush=True,
        )

    out_p = args.out_dir / f"exp21a_mouth_ablation_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "grid": [gh, gw], "l_late": l_late, "mean_k_mouth": round(float(np.mean(k_counts)), 2), "summary": summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
