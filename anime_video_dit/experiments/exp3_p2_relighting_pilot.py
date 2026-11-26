#!/usr/bin/env python3
"""Exp 3 (P2 Pilot): 2D Anime Cel-Shading Relighting Failure & 10-Line Strong Baseline Audit.

Tests whether photorealistic relighting (IC-Light / UniRelight) AND the Reviewer-#2 10-line
training-free strong baseline (luminance shading ratio + bilateral face smoothing + hue-shifted
cel ramp composited back onto original lineart) fail on pre-shaded 2D anime portraits
(double shadows, lost bang cast shadows, nose-bridge artifacts, muddy hair colors).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def ten_line_strong_baseline(
    src_rgb: np.ndarray,
    relit_rgb: np.ndarray,
    smooth_radius: int = 18,
    threshold: float = 0.88,
    shadow_tint: tuple[float, float, float] = (0.72, 0.74, 0.90),
) -> tuple[np.ndarray, dict[str, float]]:
    """Training-free strong baseline: smooth shading ratio -> hard cel mask -> hue-shifted ramp onto src."""
    eps = 1e-3
    lum_src = 0.2126 * src_rgb[..., 0] + 0.7152 * src_rgb[..., 1] + 0.0722 * src_rgb[..., 2]
    lum_rel = 0.2126 * relit_rgb[..., 0] + 0.7152 * relit_rgb[..., 1] + 0.0722 * relit_rgb[..., 2]
    ratio = np.clip(lum_rel / (lum_src + eps), 0.0, 2.0)

    # Smooth macro shading field (approximates bilateral face smoothing to erase nose bumps)
    ratio_u8 = np.clip(ratio / 2.0 * 255.0, 0, 255).astype(np.uint8)
    smooth_u8 = np.asarray(Image.fromarray(ratio_u8).filter(ImageFilter.GaussianBlur(radius=smooth_radius)), dtype=np.float32)
    smooth_ratio = (smooth_u8 / 255.0) * 2.0

    in_shadow = (smooth_ratio < threshold)[..., None]
    tint = np.array(shadow_tint, dtype=np.float32)[None, None, :]
    out = np.where(in_shadow, np.clip(src_rgb * tint, 0.0, 1.0), src_rgb)

    # Quantify edge sharpness & hue-cleanliness proxy
    metrics = {
        "shadow_area_frac": round(float(in_shadow.mean()), 4),
        "raw_relit_vs_src_l1": round(float(np.abs(relit_rgb - src_rgb).mean()), 4),
        "baseline_vs_src_l1": round(float(np.abs(out - src_rgb).mean()), 4),
    }
    return out, metrics


def measure_cel_degradation(src_rgb: np.ndarray, relit_rgb: np.ndarray) -> dict[str, float]:
    """Objective indicators of 2D cel-shading collapse -> smooth 3D PBR gradients & chroma muddying."""
    # 1. Flat-plateau gradient pollution: in 2D cel art, most interior pixels have ~0 local gradient.
    #    3D PBR relighting fills flat skin/hair plateaus with smooth non-zero shading gradients.
    gx_src = np.abs(np.diff(src_rgb, axis=1)).mean(axis=-1)
    gy_src = np.abs(np.diff(src_rgb, axis=0)).mean(axis=-1)
    grad_src = 0.5 * (gx_src[:-1, :] + gy_src[:, :-1])

    gx_rel = np.abs(np.diff(relit_rgb, axis=1)).mean(axis=-1)
    gy_rel = np.abs(np.diff(relit_rgb, axis=0)).mean(axis=-1)
    grad_rel = 0.5 * (gx_rel[:-1, :] + gy_rel[:, :-1])

    # Pixels that were flat cel plateaus in source (grad < 0.01)
    flat_mask = grad_src < 0.01
    # Pixels that were crisp ink contours in source (grad > 0.15)
    ink_mask = grad_src > 0.15

    return {
        "src_flat_plateau_frac": round(float(flat_mask.mean()), 4),
        "relit_flat_plateau_frac": round(float((grad_rel < 0.01).mean()), 4),
        "plateau_gradient_pollution_ratio": round(
            float(grad_rel[flat_mask].mean() / max(1e-5, grad_src[flat_mask].mean())) if flat_mask.any() else 0.0,
            3,
        ),
        "ink_edge_retention_ratio": round(
            float(grad_rel[ink_mask].mean() / max(1e-5, grad_src[ink_mask].mean())) if ink_mask.any() else 0.0,
            4,
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--relit", type=Path, required=True)
    ap.add_argument("--out-baseline", type=Path, required=True)
    args = ap.parse_args()

    src = np.asarray(Image.open(args.src).convert("RGB"), dtype=np.float32) / 255.0
    rel = np.asarray(Image.open(args.relit).convert("RGB").resize((src.shape[1], src.shape[0])), dtype=np.float32) / 255.0

    deg = measure_cel_degradation(src, rel)
    base_img, base_m = ten_line_strong_baseline(src, rel)
    args.out_baseline.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(base_img * 255.0, 0, 255).astype(np.uint8)).save(args.out_baseline)
    print(json.dumps({"cel_degradation": deg, "ten_line_baseline": base_m}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
