#!/usr/bin/env python3
"""
Exp 2C (P3 Editing Leakage) & Exp 3 (P2 2D Cel-Shading Relighting) on Qwen-Image-Edit-2511
plus 10-line Training-Free Cel Relighting Strong Baseline.

Runs on the clean copyright-free Animagine-XL-4.0 portraits built in Exp 2A,
using exact face_boxes.json from Exp 2B to compute 100% objective pixel-level metrics:
  - Exp 2C (P3):
      * Tier 1 Affective ('dead eyes' hollow shock while keeping smiling mouth)
      * Tier 2 Graphic-Explicit ('remove ONLY the tiny white circular highlight dots inside both irises')
      * Tier 2b Forehead Shadow ('add dark anime dread shadow strictly over forehead/eyes ONLY')
      Metrics: iris_highlight_drop, mouth_l1_leakage, outside_face_l1_leakage, global_delta_L
  - Exp 3 (P2):
      * Method A: 10-line training-free face-smoothed directional shading ratio strong baseline
      * Method B: Qwen-Image-Edit-2511 dramatic warm right-side cel-relighting prompt
      Metrics: plateau_gradient_pollution_ratio, ink_edge_retention_ratio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


def _to_px(box_1000: list[int], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, int(round(box_1000[0] * w / 1000.0))))
    y1 = max(0, min(h - 1, int(round(box_1000[1] * h / 1000.0))))
    x2 = max(x1 + 4, min(w, int(round(box_1000[2] * w / 1000.0))))
    y2 = max(y1 + 4, min(h, int(round(box_1000[3] * h / 1000.0))))
    return x1, y1, x2, y2


def _rgb_to_luma(arr: np.ndarray) -> np.ndarray:
    return 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    return np.hypot(gx, gy)


def compute_cel_metrics(orig_img: Image.Image, relit_img: Image.Image) -> dict[str, float]:
    o = np.asarray(orig_img.convert("RGB"), dtype=np.float32)
    r = np.asarray(relit_img.convert("RGB").resize(orig_img.size, Image.BILINEAR), dtype=np.float32)
    go = _grad_mag(_rgb_to_luma(o))
    gr = _grad_mag(_rgb_to_luma(r))
    # Flat cel plateaus in orig: gradient < 4.0
    plateau = go < 4.0
    # Crisp dark anime ink edges in orig: dark pixels with strong gradient
    ink = (_rgb_to_luma(o) < 85.0) & (go > 22.0)

    # Plateau gradient pollution: fraction of originally flat cel plateau pixels that now have smooth airbrush gradients (3..18)
    polluted = plateau & (gr > 3.5) & (gr < 24.0)
    pollution_ratio = float(polluted.sum() / max(1, plateau.sum()))
    # Ink edge retention: fraction of orig ink edge pixels that still have strong gradient (>16) and remain dark (<115)
    retained = ink & (gr > 16.0) & (_rgb_to_luma(r) < 115.0)
    ink_retention = float(retained.sum() / max(1, ink.sum()))
    # Color/chroma drift in non-ink regions
    mean_l1 = float(np.abs(r - o).mean())
    return {
        "plateau_gradient_pollution": round(pollution_ratio, 4),
        "ink_edge_retention": round(ink_retention, 4),
        "mean_l1": round(mean_l1, 2),
    }


def baseline_10line_cel_relight(img: Image.Image, light_dir: tuple[float, float] = (1.0, -0.25)) -> Image.Image:
    """Reviewer-#2's 10-line training-free face-smoothed directional shading ratio baseline."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / (w * 0.5)
    ny = (yy - h * 0.5) / (h * 0.5)
    # Smooth spherical face proxy normal
    nz = np.sqrt(np.clip(1.0 - 0.65 * nx * nx - 0.65 * ny * ny, 0.05, 1.0))
    lx, ly, lz = light_dir[0], light_dir[1], 0.65
    ln = (lx * lx + ly * ly + lz * lz) ** 0.5
    ndotl = (nx * lx + ny * ly + nz * lz) / ln
    # Quantize to 2-tone cel ramp (hard crisp cel terminator!) + cool shadow tint / warm key light
    lit = (ndotl > 0.28).astype(np.float32)
    lit_soft = np.asarray(Image.fromarray((lit * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32) / 255.0
    # Warm right-key vs cool indigo left-shadow multiply
    key_rgb = np.array([1.07, 1.01, 0.94], dtype=np.float32)
    shd_rgb = np.array([0.66, 0.68, 0.82], dtype=np.float32)
    # Protect background (very bright near-white) & dark ink lines
    luma = _rgb_to_luma(arr)
    fg = (luma < 238.0).astype(np.float32)
    out = arr.copy()
    for c in range(3):
        gain = shd_rgb[c] * (1.0 - lit_soft) + key_rgb[c] * lit_soft
        out[:, :, c] = np.clip(arr[:, :, c] * (1.0 - fg + fg * gain), 0, 255)
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def compute_p3_leakage(
    orig: Image.Image, edited: Image.Image, boxes: dict[str, list[int]], target_region: str
) -> dict[str, float]:
    o = np.asarray(orig.convert("RGB"), dtype=np.float32)
    e = np.asarray(edited.convert("RGB").resize(orig.size, Image.BILINEAR), dtype=np.float32)
    h, w = o.shape[:2]

    # Target mask
    tmask = np.zeros((h, w), dtype=bool)
    lx1, ly1, lx2, ly2 = _to_px(boxes["left_eye"], w, h)
    rx1, ry1, rx2, ry2 = _to_px(boxes["right_eye"], w, h)
    mx1, my1, mx2, my2 = _to_px(boxes["mouth"], w, h)
    fx1, fy1, fx2, fy2 = _to_px(boxes["face"], w, h)

    if target_region == "eyes":
        tmask[ly1:ly2, lx1:lx2] = True
        tmask[ry1:ry2, rx1:rx2] = True
    elif target_region == "forehead_eyes":
        tmask[max(0, fy1 - int(0.05 * h)) : max(ly2, ry2) + int(0.05 * h), fx1:fx2] = True

    outside = ~tmask
    mouth_mask = np.zeros((h, w), dtype=bool)
    mouth_mask[my1:my2, mx1:mx2] = True

    diff_rgb = np.abs(e - o).mean(axis=2)
    lo = _rgb_to_luma(o)
    le = _rgb_to_luma(e)

    # Measure bright white specular pixels inside eyes before vs after
    eye_hl_drops = []
    for (x1, y1, x2, y2) in [(lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2)]:
        patch_o = o[y1:y2, x1:x2]
        patch_e = e[y1:y2, x1:x2]
        cmax = patch_o.max(axis=2) / 255.0
        cmin = patch_o.min(axis=2) / 255.0
        sat = np.where(cmax > 1e-5, (cmax - cmin) / np.maximum(cmax, 1e-5), 0.0)
        # Inner 60% of box (iris region)
        bh, bw = y2 - y1, x2 - x1
        yy, xx = np.ogrid[0:bh, 0:bw]
        inner = ((xx - bw / 2.0) / max(1.0, bw * 0.35)) ** 2 + ((yy - bh / 2.0) / max(1.0, bh * 0.40)) ** 2 <= 1.0
        hl = inner & (cmax > 0.80) & (sat < 0.35)
        if hl.sum() > 0:
            eye_hl_drops.append(float(_rgb_to_luma(patch_o)[hl].mean() - _rgb_to_luma(patch_e)[hl].mean()))

    return {
        "global_delta_L_outside": round(float((le[outside] - lo[outside]).mean()), 2),
        "outside_mask_l1": round(float(diff_rgb[outside].mean()), 2),
        "mouth_l1_leakage": round(float(diff_rgb[mouth_mask].mean()), 2),
        "target_region_l1": round(float(diff_rgb[tmask].mean()), 2),
        "iris_highlight_luma_drop": round(float(np.mean(eye_hl_drops)) if eye_hl_drops else 0.0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portraits-dir", type=Path, required=True)
    ap.add_argument("--boxes-json", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/hg-Qwen-Image-Edit-2511")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-chars", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    boxes_map = json.loads(args.boxes_json.read_text(encoding="utf-8"))
    portraits = sorted(args.portraits_dir.glob("*.png"))[: args.max_chars]

    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    tasks = [
        (
            "p3_tier1_affective",
            "eyes",
            "Change this 2D anime character's eyes to classic anime 'dead eyes' (emotionally hollow, shocked, lifeless eyes without highlights) while keeping her exact same smiling mouth.",
        ),
        (
            "p3_tier2_graphic_explicit",
            "eyes",
            "Remove ONLY the tiny white circular specular highlight dots inside both colored irises of the character's eyes, filling those tiny dots with the surrounding iris color. Keep the smiling mouth, eyebrows, skin tone, hair, lighting, brightness, and background 100% unchanged.",
        ),
        (
            "p3_tier2_forehead_shadow",
            "forehead_eyes",
            "Add a classic anime ominous dark vertical shadow with fine dread lines strictly across the upper forehead and eyes only. Keep the smiling mouth, lower face, clothes, and background 100% unchanged.",
        ),
        (
            "p2_relight_cel_right",
            "relight",
            "Relight this 2D anime portrait with strong warm directional sunset key light from the right side and crisp flat cel-shaded cool shadows on the left side. Strictly preserve flat 2D anime cel-shading and sharp dark ink outlines; do NOT add 3D photorealistic airbrushing or soft gradients.",
        ),
    ]

    p3_rows: list[dict[str, Any]] = []
    p2_rows: list[dict[str, Any]] = []

    for idx, p in enumerate(portraits):
        cid = p.stem
        orig = Image.open(p).convert("RGB")
        # Resize to 768x768 for fast clean editing on Qwen-Image-Edit-2511
        orig_edit = orig.resize((768, 768), Image.LANCZOS)
        boxes = boxes_map[cid]

        # Run 10-line strong baseline for P2
        base_relit = baseline_10line_cel_relight(orig_edit)
        base_relit.save(args.out_dir / f"{cid}__p2_baseline10line.png")
        base_m = compute_cel_metrics(orig_edit, base_relit)
        p2_rows.append({"char_id": cid, "method": "baseline_10line_cel", **base_m})

        thumbs = [orig_edit.resize((256, 256), Image.LANCZOS)]
        labels = ["orig"]

        for tname, region, prompt in tasks:
            gen = torch.Generator(device="cuda").manual_seed(2026 + idx * 13)
            out_img = pipe(
                image=[orig_edit],
                prompt=prompt,
                negative_prompt=" ",
                num_inference_steps=args.steps,
                true_cfg_scale=4.0,
                generator=gen,
            ).images[0]
            out_img.save(args.out_dir / f"{cid}__{tname}.png")

            t = out_img.resize((256, 256), Image.LANCZOS)
            thumbs.append(t)
            labels.append(tname)

            if region == "relight":
                m = compute_cel_metrics(orig_edit, out_img)
                p2_rows.append({"char_id": cid, "method": "qwen_image_edit_2511", **m})
                print(f"[P2] {cid} qwen_relight={m} vs baseline10={base_m}", flush=True)
            else:
                m = compute_p3_leakage(orig_edit, out_img, boxes, region)
                p3_rows.append({"char_id": cid, "task": tname, **m})
                print(f"[P3] {cid} {tname}={m}", flush=True)

        thumbs.append(base_relit.resize((256, 256), Image.LANCZOS))
        labels.append("p2_baseline10line")

        sheet = Image.new("RGB", (256 * len(thumbs), 256))
        for i, (t, lab) in enumerate(zip(thumbs, labels)):
            ImageDraw.Draw(t).rectangle([0, 0, 256, 24], fill=(20, 20, 28))
            ImageDraw.Draw(t).text((6, 5), lab, fill=(255, 255, 255))
            sheet.paste(t, (i * 256, 0))
        sheet.save(args.out_dir / f"{cid}__edit_sheet.jpg", quality=92)

    # Aggregate summaries
    p3_summary: dict[str, Any] = {}
    for tname in ["p3_tier1_affective", "p3_tier2_graphic_explicit", "p3_tier2_forehead_shadow"]:
        sub = [r for r in p3_rows if r["task"] == tname]
        p3_summary[tname] = {
            k: round(float(np.mean([r[k] for r in sub])), 3)
            for k in ["global_delta_L_outside", "outside_mask_l1", "mouth_l1_leakage", "target_region_l1", "iris_highlight_luma_drop"]
        }

    p2_summary: dict[str, Any] = {}
    for mname in ["baseline_10line_cel", "qwen_image_edit_2511"]:
        sub = [r for r in p2_rows if r["method"] == mname]
        p2_summary[mname] = {
            k: round(float(np.mean([r[k] for r in sub])), 4)
            for k in ["plateau_gradient_pollution", "ink_edge_retention", "mean_l1"]
        }

    res = {"p3_edit_summary": p3_summary, "p2_relight_summary": p2_summary, "p3_rows": p3_rows, "p2_rows": p2_rows}
    out_json = args.out_dir / "edit_and_relight_results.json"
    out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== P3 EDITING LEAKAGE SUMMARY ===")
    print(json.dumps(p3_summary, indent=2))
    print("=== P2 RELIGHTING SUMMARY ===")
    print(json.dumps(p2_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
