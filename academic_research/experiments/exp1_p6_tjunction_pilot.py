#!/usr/bin/env python3
"""Exp 1 (P6 Pilot): Fixed-Style Minimal-Pair T-Junction Flip Test (Zero External Assets Required).

Generates paired A-over-B vs B-over-A stimuli with identical geometry in two render modes:
  1. L3_Flat2D: Pure flat anime color fills + crisp black ink outlines + T-junctions ONLY (zero shading).
  2. L1_Shaded: Same geometry + soft cast drop-shadow under the foreground ribbon/part (shading cue).

Includes two regimes:
  - identical_color: Both overlapping flat anime ribbons share the EXACT same flat RGB fill
    (semantic & color prior = strictly 50/50; ONLY the T-junction ink discontinuity tells who is in front!).
  - distinct_color: Distinct pastel anime colors (tests whether color segmentation helps or still fails flip).

Evaluates monocular depth models (Depth Anything V2 / DA3 / MoGe-2) and/or VLMs (Qwen2.5-VL / Qwen3-VL)
using strict Paired Flip Consistency at clean interior sample points >= 36px away from the crossing edge.

Usage:
  # Step 1: Generate stimuli only (runs anywhere with numpy + Pillow)
  python3 exp1_p6_tjunction_pilot.py --out-dir ./pilot_p6_out --generate-only

  # Step 2: Run Depth Anything V2 / MoGe-2 / Qwen-VL on GPU node
  python3 exp1_p6_tjunction_pilot.py --out-dir ./pilot_p6_out --eval-depth depth-anything/Depth-Anything-V2-Large-hf
  python3 exp1_p6_tjunction_pilot.py --out-dir ./pilot_p6_out --eval-vlm Qwen/Qwen2.5-VL-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _ribbon_polygon(
    cx: float,
    cy: float,
    angle_deg: float,
    length: float,
    width: float,
) -> list[tuple[float, float]]:
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    nx, ny = -dy, dx
    hl, hw = length / 2.0, width / 2.0
    return [
        (cx - dx * hl - nx * hw, cy - dy * hl - ny * hw),
        (cx + dx * hl - nx * hw, cy + dy * hl - ny * hw),
        (cx + dx * hl + nx * hw, cy + dy * hl + ny * hw),
        (cx - dx * hl + nx * hw, cy - dy * hl + ny * hw),
    ]


def _draw_ribbon(
    img: Image.Image,
    poly: list[tuple[float, float]],
    fill_rgb: tuple[int, int, int],
    outline_rgb: tuple[int, int, int] = (20, 20, 24),
    outline_px: int = 5,
    cast_shadow: bool = False,
) -> None:
    if cast_shadow:
        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        shifted = [(x + 12, y + 12) for (x, y) in poly]
        sd.polygon(shifted, fill=(0, 0, 0, 115))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=9))
        img.alpha_composite(shadow)

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.polygon(poly, fill=(*fill_rgb, 255))
    # Crisp closed anime ink contour around the ribbon (creates exact T-junction over the back ribbon)
    pts = poly + [poly[0]]
    draw.line(pts, fill=(*outline_rgb, 255), width=outline_px, joint="curve")
    img.alpha_composite(layer)


def generate_stimuli(out_dir: Path, img_size: int = 512) -> list[dict[str, Any]]:
    """Generate 20 minimal-pair T-junction flip pairs (40 images per render level = 80 images total)."""
    stim_dir = out_dir / "stimuli"
    stim_dir.mkdir(parents=True, exist_ok=True)

    palettes = [
        ("pink_same", (245, 175, 195), (245, 175, 195), (248, 248, 252)),
        ("blue_same", (150, 195, 240), (150, 195, 240), (250, 250, 252)),
        ("blonde_same", (248, 224, 150), (248, 224, 150), (248, 248, 250)),
        ("lavender_same", (195, 175, 238), (195, 175, 238), (250, 248, 252)),
        ("mint_same", (160, 225, 200), (160, 225, 200), (248, 250, 250)),
        ("pink_blue", (245, 175, 195), (150, 195, 240), (248, 248, 252)),
        ("gold_navy", (248, 220, 140), (95, 120, 185), (250, 250, 252)),
        ("mint_coral", (160, 225, 200), (245, 155, 145), (248, 250, 250)),
        ("white_red", (242, 242, 246), (225, 95, 110), (235, 238, 245)),
        ("peach_lilac", (250, 205, 175), (185, 165, 230), (248, 248, 252)),
    ]
    angle_pairs = [
        (25.0, -35.0),
        (10.0, 80.0),
    ]

    records: list[dict[str, Any]] = []
    pair_idx = 0
    cx, cy = img_size / 2.0, img_size / 2.0
    length = img_size * 0.78
    width = img_size * 0.17  # ~87px wide -> interior centers are well clear of stride-14 patches

    for pal_name, col_a, col_b, bg_rgb in palettes:
        regime = "identical_color" if col_a == col_b else "distinct_color"
        for ang_a, ang_b in angle_pairs:
            pair_id = f"pair_{pair_idx:02d}_{pal_name}_a{int(ang_a)}_b{int(ang_b)}"
            pair_idx += 1

            poly_a = _ribbon_polygon(cx, cy, ang_a, length, width)
            poly_b = _ribbon_polygon(cx, cy, ang_b, length, width)

            # Clean interior sampling points along A and B, offset 115px from crossing center
            # (guaranteed >= 36px from crossing boundary & ribbon edges, immune to ViT patch-14 blur)
            offset = img_size * 0.22
            pt_a = (
                int(round(cx + math.cos(math.radians(ang_a)) * offset)),
                int(round(cy + math.sin(math.radians(ang_a)) * offset)),
            )
            pt_b = (
                int(round(cx + math.cos(math.radians(ang_b)) * offset)),
                int(round(cy + math.sin(math.radians(ang_b)) * offset)),
            )

            for render_level, use_shadow in [("L3_Flat2D", False), ("L1_Shaded", True)]:
                for order in ["A_over_B", "B_over_A"]:
                    canvas = Image.new("RGBA", (img_size, img_size), (*bg_rgb, 255))
                    if order == "A_over_B":
                        _draw_ribbon(canvas, poly_b, col_b, cast_shadow=False)
                        _draw_ribbon(canvas, poly_a, col_a, cast_shadow=use_shadow)
                        front_label = "A"
                    else:
                        _draw_ribbon(canvas, poly_a, col_a, cast_shadow=False)
                        _draw_ribbon(canvas, poly_b, col_b, cast_shadow=use_shadow)
                        front_label = "B"

                    rgb = canvas.convert("RGB")
                    fname = f"{pair_id}__{render_level}__{order}.png"
                    fpath = stim_dir / fname
                    rgb.save(fpath)

                    # Also save dot-annotated copy for VLM spatial QA (red dot = A, blue dot = B)
                    ann = rgb.copy()
                    ad = ImageDraw.Draw(ann)
                    r = 10
                    ad.ellipse((pt_a[0] - r, pt_a[1] - r, pt_a[0] + r, pt_a[1] + r), fill=(230, 30, 30), outline=(255, 255, 255), width=2)
                    ad.ellipse((pt_b[0] - r, pt_b[1] - r, pt_b[0] + r, pt_b[1] + r), fill=(30, 90, 235), outline=(255, 255, 255), width=2)
                    ann_path = stim_dir / f"{pair_id}__{render_level}__{order}__dots.png"
                    ann.save(ann_path)

                    records.append(
                        {
                            "pair_id": pair_id,
                            "regime": regime,
                            "render_level": render_level,
                            "order": order,
                            "front_gt": front_label,
                            "pt_a": list(pt_a),
                            "pt_b": list(pt_b),
                            "image_path": str(fpath),
                            "dots_image_path": str(ann_path),
                        }
                    )

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Generated {len(records)} stimuli ({pair_idx} minimal pairs x 2 render levels x 2 orders) -> {manifest_path}")
    return records


def _sample_patch_median(depth_map: np.ndarray, pt: tuple[int, int], radius: int = 8) -> float:
    h, w = depth_map.shape[:2]
    x, y = pt
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    return float(np.median(depth_map[y0:y1, x0:x1]))


def eval_depth_hf(records: list[dict[str, Any]], model_id: str, out_dir: Path, higher_is_closer: bool = True) -> None:
    """Evaluate a HuggingFace transformers depth-estimation model (e.g., Depth Anything V2)."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=model_id, device=device)

    viz_dir = out_dir / f"depth_viz_{model_id.replace('/', '_')}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    preds: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        out = pipe(img)
        depth = np.squeeze(np.asarray(out["predicted_depth"], dtype=np.float32))
        if depth.ndim != 2:
            raise ValueError(f"Unexpected depth shape after squeeze: {depth.shape}")
        if depth.shape != (img.height, img.width):
            depth_img = Image.fromarray(depth, mode="F").resize((img.width, img.height), resample=Image.BILINEAR)
            depth = np.asarray(depth_img, dtype=np.float32)

        da = _sample_patch_median(depth, tuple(rec["pt_a"]))
        db = _sample_patch_median(depth, tuple(rec["pt_b"]))
        pred_front = "A" if ((da > db) if higher_is_closer else (da < db)) else "B"
        correct = pred_front == rec["front_gt"]
        preds[(rec["pair_id"], rec["render_level"], rec["order"])] = {
            "depth_a": da,
            "depth_b": db,
            "delta_ab": da - db,
            "pred_front": pred_front,
            "correct": correct,
            "regime": rec["regime"],
        }

        # Save visual diagnostic for pair_00 and pair_10
        if rec["pair_id"] in {"pair_00", "pair_10"}:
            dmin, dmax = float(depth.min()), float(depth.max())
            norm = (depth - dmin) / max(1e-6, dmax - dmin)
            if not higher_is_closer:
                norm = 1.0 - norm
            vis = Image.fromarray((np.clip(norm, 0, 1) * 255).astype(np.uint8)).convert("RGB")
            d = ImageDraw.Draw(vis)
            ax, ay = rec["pt_a"]
            bx, by = rec["pt_b"]
            d.ellipse([ax - 7, ay - 7, ax + 7, ay + 7], fill=(255, 40, 40), outline=(255, 255, 255), width=2)
            d.ellipse([bx - 7, by - 7, bx + 7, by + 7], fill=(40, 120, 255), outline=(255, 255, 255), width=2)
            strip = Image.new("RGB", (img.width * 2, img.height))
            strip.paste(img, (0, 0))
            strip.paste(vis, (img.width, 0))
            strip.save(viz_dir / f"{rec['pair_id']}_{rec['render_level']}_{rec['order']}.png")

    # Summarize single-image accuracy AND strict Paired Flip Consistency
    summary: dict[str, Any] = {"model_id": model_id, "higher_is_closer": higher_is_closer, "slices": {}}
    for render_level in ["L1_Shaded", "L3_Flat2D"]:
        for regime in ["identical_color", "distinct_color", "ALL"]:
            pair_ids = sorted({
                r["pair_id"]
                for r in records
                if r["render_level"] == render_level and (regime == "ALL" or r["regime"] == regime)
            })
            single_ok = 0
            single_tot = 0
            flip_ok = 0
            abs_deltas = []
            signed_flips = []
            for pid in pair_ids:
                p1 = preds[(pid, render_level, "A_over_B")]
                p2 = preds[(pid, render_level, "B_over_A")]
                single_ok += int(p1["correct"]) + int(p2["correct"])
                single_tot += 2
                if p1["correct"] and p2["correct"]:
                    flip_ok += 1
                sign = 1.0 if higher_is_closer else -1.0
                # Positive signed flip means depth difference reversed in the physically right direction
                signed_flips.append(sign * (p1["delta_ab"] - p2["delta_ab"]))
                abs_deltas.append(abs(p1["delta_ab"]) + abs(p2["delta_ab"]))
            key = f"{render_level}__{regime}"
            summary["slices"][key] = {
                "n_pairs": len(pair_ids),
                "single_acc": round(single_ok / max(1, single_tot), 4),
                "paired_flip_consistency": round(flip_ok / max(1, len(pair_ids)), 4),
                "mean_signed_flip_delta": round(float(np.mean(signed_flips)) if signed_flips else 0.0, 5),
            }

    res_path = out_dir / f"depth_eval_{model_id.replace('/', '_')}.json"
    res_path.write_text(
        json.dumps(
            {"summary": summary, "preds": {f"{k[0]}__{k[1]}__{k[2]}": v for k, v in preds.items()}},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved depth eval -> {res_path}")


def eval_vlm_hf(records: list[dict[str, Any]], model_id: str, out_dir: Path) -> None:
    """Evaluate Qwen2.5-VL / Qwen3-VL on the dot-annotated minimal-pair T-junction flips."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    prompt = (
        "Look carefully at the two overlapping flat ribbons in this image. "
        "One ribbon has a RED dot on it, and the other ribbon has a BLUE dot on it. "
        "Which ribbon is physically IN FRONT OF (on top of / occluding) the other at the crossing? "
        "Answer with ONLY a single word: RED or BLUE."
    )

    preds: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rec in records:
        img = Image.open(rec["dots_image_path"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        gen = out_ids[0, inputs["input_ids"].shape[1] :]
        raw = processor.decode(gen, skip_special_tokens=True).strip()
        up = raw.upper()
        pred = "A" if up.startswith("RED") else ("B" if up.startswith("BLUE") else ("A" if ("RED" in up and "BLUE" not in up) else ("B" if ("BLUE" in up and "RED" not in up) else "UNK")))
        preds[(rec["pair_id"], rec["render_level"], rec["order"])] = {
            "raw": raw,
            "pred_front": pred,
            "correct": pred == rec["front_gt"],
            "regime": rec["regime"],
        }

    summary: dict[str, Any] = {"model_id": model_id, "slices": {}}
    for render_level in ["L1_Shaded", "L3_Flat2D"]:
        for regime in ["identical_color", "distinct_color", "ALL"]:
            pair_ids = sorted({
                r["pair_id"]
                for r in records
                if r["render_level"] == render_level and (regime == "ALL" or r["regime"] == regime)
            })
            single_ok = 0
            flip_ok = 0
            for pid in pair_ids:
                p1 = preds[(pid, render_level, "A_over_B")]
                p2 = preds[(pid, render_level, "B_over_A")]
                single_ok += int(p1["correct"]) + int(p2["correct"])
                if p1["correct"] and p2["correct"]:
                    flip_ok += 1
            summary["slices"][f"{render_level}__{regime}"] = {
                "n_pairs": len(pair_ids),
                "single_acc": round(single_ok / max(1, 2 * len(pair_ids)), 4),
                "paired_flip_consistency": round(flip_ok / max(1, len(pair_ids)), 4),
            }

    safe_name = Path(model_id).name if "/" in model_id else model_id
    res_path = out_dir / f"vlm_eval_{safe_name}.json"
    res_path.write_text(json.dumps({"summary": summary, "preds": {f"{k[0]}__{k[1]}__{k[2]}": v for k, v in preds.items()}}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved VLM eval -> {res_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--skip-gen", action="store_true", help="Reuse existing manifest.jsonl in out-dir")
    ap.add_argument("--eval-depth", type=str, default=None, help="HF depth model id, e.g. depth-anything/Depth-Anything-V2-Large-hf")
    ap.add_argument("--eval-vlm", type=str, default=None, help="HF/local Qwen-VL model id/path")
    ap.add_argument("--lower-is-closer", action="store_true", help="Set for metric depth models where smaller value = closer")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.jsonl"
    if args.skip_gen and manifest.exists():
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = generate_stimuli(args.out_dir)
    if args.generate_only:
        return 0
    if args.eval_depth:
        eval_depth_hf(records, args.eval_depth, args.out_dir, higher_is_closer=not args.lower_is_closer)
    if args.eval_vlm:
        eval_vlm_hf(records, args.eval_vlm, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
