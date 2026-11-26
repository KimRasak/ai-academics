#!/usr/bin/env python3
"""
Exp 3B (P2 Optimization Test): Depth-Guided Normal-Flattened 2D Cel-Shading Relighting
vs. 10-Line Spherical Baseline vs. Qwen-Image-Edit-2511.

Tests whether a clean, physics-grounded 2D anime relighting operator:
  1. Depth-Anything-V2-Large geometry estimation (captures real hair volume, head, neck, shoulders)
  2. Selective Face/Nose Normal Flattening (parameter alpha in [0, 1] prevents 3D plastic nose shadows)
  3. Depth Ray-Marched Hair-Bang Cast Shadow (casts crisp 2D graphic shadows from hair onto forehead)
  4. Hue-Shifted Chromatic 2-Tone Cel Ramp + Dark Ink Preservation (zero airbrush gradient pollution!)
quantitatively and visually outperforms BOTH:
  - Reviewer-#2's 10-line spherical baseline (ignores real geometry / misses hair & collar occlusion)
  - SOTA diffusion editor Qwen-Image-Edit-2511 (pollutes flat cel plateaus & blurs ink contours).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from transformers import pipeline


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
    plateau = go < 4.0
    ink = (_rgb_to_luma(o) < 85.0) & (go > 22.0)
    polluted = plateau & (gr > 3.5) & (gr < 24.0)
    retained = ink & (gr > 16.0) & (_rgb_to_luma(r) < 115.0)
    # Shadow step sharpness: ratio of sharp shadow edges (grad > 12) to soft mid-gradients (3.5..12) inside changed regions
    changed = np.abs(r - o).mean(axis=2) > 8.0
    sharp_edge = (changed & plateau & (gr >= 12.0)).sum()
    soft_grad = (changed & plateau & (gr > 3.5) & (gr < 12.0)).sum()
    return {
        "plateau_gradient_pollution": round(float(polluted.sum() / max(1, plateau.sum())), 4),
        "ink_edge_retention": round(float(retained.sum() / max(1, ink.sum())), 4),
        "cel_edge_to_airbrush_ratio": round(float(sharp_edge / max(1, soft_grad)), 3),
        "mean_l1": round(float(np.abs(r - o).mean()), 2),
    }


def _blur_f32(arr: np.ndarray, radius: float) -> np.ndarray:
    """Smooth a [0, 1] float32 2D array via separable PyTorch Gaussian conv (exact float32 precision)."""
    import torch.nn.functional as F

    sigma = max(0.5, float(radius))
    k = max(3, int(round(sigma * 3.0)) * 2 + 1)
    xs = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    kern = torch.exp(-0.5 * (xs / sigma) ** 2)
    kern = kern / kern.sum()
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    pad = k // 2
    t = F.pad(t, (pad, pad, pad, pad), mode="reflect")
    t = F.conv2d(t, kern.view(1, 1, 1, k))
    t = F.conv2d(t, kern.view(1, 1, k, 1))
    return t[0, 0].numpy()


def depth_guided_cel_relight(
    img: Image.Image,
    depth_norm: np.ndarray,
    light_dir: tuple[float, float, float] = (0.82, -0.25, 0.52),
    flatten_alpha: float = 0.55,
    terminator: float = 0.22,
    cast_shadow_steps: int = 18,
) -> Image.Image:
    """
    Depth-guided normal-flattened chromatic 2D cel-shading relighting:
      - `depth_norm`: [0, 1] closer = higher (from Depth-Anything-V2-Large)
      - `flatten_alpha`: blends fine high-freq 3D normals with smooth anime macro-plane normals
      - `cast_shadow_steps`: short screen-space depth ray-march along light_dir for crisp cast shadows
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    # 1. Foreground mask (exclude plain bright background using BOTH depth and near-white check)
    luma = _rgb_to_luma(arr)
    bg_luma = (luma > 240.0) & (depth_norm < np.percentile(depth_norm, 22))
    fg = _blur_f32((~bg_luma).astype(np.float32), radius=1.0)

    # 2. Multi-scale depth smoothing for normal flattening (removes bumpy 3D nose/cheek noise!)
    d_fine = _blur_f32(depth_norm, radius=3.5)
    d_macro = _blur_f32(depth_norm, radius=20.0)
    d_blend = (1.0 - flatten_alpha) * d_fine + flatten_alpha * d_macro

    gy, gx = np.gradient(d_blend)
    scale = 26.0
    nx, ny, nz = -gx * scale, -gy * scale, np.ones_like(d_blend)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    # Normalize light direction
    lx, ly, lz = light_dir
    ln = (lx * lx + ly * ly + lz * lz) ** 0.5
    lx, ly, lz = lx / ln, ly / ln, lz / ln
    ndotl = nx * lx + ny * ly + nz * lz

    # 3. Screen-space depth ray-march cast shadow (e.g., hair bangs onto forehead, head onto neck/collar)
    cast_occ = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for step in range(3, cast_shadow_steps + 1):
        sx = np.clip(np.round(xx + lx * step * 1.8).astype(np.int32), 0, w - 1)
        sy = np.clip(np.round(yy + ly * step * 1.8).astype(np.int32), 0, h - 1)
        dz = d_fine[sy, sx] - d_fine
        hit = (dz > (0.045 + 0.002 * step)).astype(np.float32)
        cast_occ = np.maximum(cast_occ, hit)
    cast_occ = np.asarray(Image.fromarray((cast_occ * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5)), dtype=np.float32) / 255.0

    # 4. Crisp 2-tone cel terminator (hard graphic shadow edge, NO soft airbrush!)
    lit_raw = ((ndotl > terminator) & (cast_occ < 0.5)).astype(np.float32)
    lit_aa = _blur_f32(lit_raw, radius=0.8)

    # Optional crisp rim highlight on the light-facing grazing silhouette
    rim = _blur_f32(((ndotl > 0.62) & (d_fine - d_macro > 0.015) & (cast_occ < 0.5)).astype(np.float32), radius=0.8)

    # 5. Artist-authentic Chromatic Anime Cel Shading (hue-shifted cool-violet shadow, warm sunset key, protect dark ink!)
    ink_protect = np.clip((luma - 55.0) / 45.0, 0.0, 1.0)
    effect = fg * ink_protect

    key_mul = np.array([1.07, 1.01, 0.93], dtype=np.float32)
    # Chromatic anime shadow: multiply + subtle cool-indigo shadow ambient lift (prevents muddy dark gray!)
    shd_mul = np.array([0.66, 0.65, 0.80], dtype=np.float32)
    shd_add = np.array([8.0, 4.0, 18.0], dtype=np.float32)
    rim_add = np.array([42.0, 28.0, 12.0], dtype=np.float32)

    out = arr.copy()
    for c in range(3):
        lit_val = arr[:, :, c] * key_mul[c] + rim * rim_add[c]
        shd_val = arr[:, :, c] * shd_mul[c] + shd_add[c]
        relit_c = shd_val * (1.0 - lit_aa) + lit_val * lit_aa
        out[:, :, c] = np.clip(arr[:, :, c] * (1.0 - effect) + relit_c * effect, 0, 255)

    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portraits-dir", type=Path, required=True)
    ap.add_argument("--edit-dir", type=Path, required=True, help="Dir containing Qwen-Image-Edit & 10-line outputs for comparison")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-chars", type=int, default=6)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=0 if torch.cuda.is_available() else -1)

    portraits = sorted(args.portraits_dir.glob("*.png"))[: args.max_chars]
    rows: list[dict[str, Any]] = []

    for p in portraits:
        cid = p.stem
        orig = Image.open(p).convert("RGB").resize((768, 768), Image.LANCZOS)
        depth = np.squeeze(np.asarray(pipe(orig)["predicted_depth"], dtype=np.float32))
        if depth.shape != (768, 768):
            depth = np.asarray(Image.fromarray(depth, mode="F").resize((768, 768), Image.BILINEAR), dtype=np.float32)
        dmin, dmax = float(depth.min()), float(depth.max())
        depth_norm = (depth - dmin) / max(1e-6, dmax - dmin)

        ours_right = depth_guided_cel_relight(orig, depth_norm, light_dir=(0.82, -0.25, 0.52), flatten_alpha=0.55)
        ours_left = depth_guided_cel_relight(orig, depth_norm, light_dir=(-0.82, -0.25, 0.52), flatten_alpha=0.55)
        ours_right.save(args.out_dir / f"{cid}__ours_cel_right.png")
        ours_left.save(args.out_dir / f"{cid}__ours_cel_left.png")

        m_ours = compute_cel_metrics(orig, ours_right)
        rows.append({"char_id": cid, "method": "ours_depth_cel_relight", **m_ours})

        # Build 4-way visual comparison sheet if Qwen-Image-Edit & 10-line baseline exist
        panels = [("orig", orig), ("ours_depth_cel_R", ours_right), ("ours_depth_cel_L", ours_left)]
        b10_path = args.edit_dir / f"{cid}__p2_baseline10line.png"
        qwen_path = args.edit_dir / f"{cid}__p2_relight_cel_right.png"
        if b10_path.exists():
            b10 = Image.open(b10_path).convert("RGB")
            panels.insert(1, ("baseline_10line", b10))
            rows.append({"char_id": cid, "method": "baseline_10line", **compute_cel_metrics(orig, b10)})
        if qwen_path.exists():
            qw = Image.open(qwen_path).convert("RGB")
            panels.insert(2, ("qwen_image_edit_2511", qw))
            rows.append({"char_id": cid, "method": "qwen_image_edit_2511", **compute_cel_metrics(orig, qw)})

        sheet = Image.new("RGB", (256 * len(panels), 256))
        for i, (lab, im) in enumerate(panels):
            t = im.resize((256, 256), Image.LANCZOS)
            ImageDraw.Draw(t).rectangle([0, 0, 256, 24], fill=(20, 20, 28))
            ImageDraw.Draw(t).text((6, 5), lab, fill=(255, 255, 255))
            sheet.paste(t, (i * 256, 0))
        sheet.save(args.out_dir / f"{cid}__p2_compare_sheet.jpg", quality=92)
        print(f"Processed {cid}: ours={m_ours}", flush=True)

    summary = {}
    for mname in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == mname]
        summary[mname] = {
            k: round(float(np.mean([r[k] for r in sub])), 4)
            for k in ["plateau_gradient_pollution", "ink_edge_retention", "cel_edge_to_airbrush_ratio", "mean_l1"]
        }
    (args.out_dir / "p2_comparison_summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("=== P2 3-WAY COMPARISON SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
