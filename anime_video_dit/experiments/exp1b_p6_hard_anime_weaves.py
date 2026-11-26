#!/usr/bin/env python3
"""
Exp 1B (P6 OOD Test Suite): Curved 2D Anime Hair/Ribbon Weaves & Overlapping Cel-Shapes.

Purpose:
  Out-of-distribution (OOD) minimal-pair T-junction flip benchmark distinct from
  straight ribbons in exp1_p6_tjunction_pilot.py:
    1. `curved_weave`: Curved S-shaped / arc 2D anime hair-strand / ribbon crossings
       with tapered ends and cel-ink outlines (identical_color vs distinct_color).
    2. `cel_shapes`: Overlapping rounded cel-shaded plates / capsules / circles
       (completely different geometry from ribbons, testing whether T-junction
       understanding generalizes across shapes).

  Every pair has an exact 2x2 factorial control:
    - Render level: L1_Shaded (3D cylindrical/dome shading + soft contact cast shadow)
                    vs L3_Flat2D (pure uniform anime cel-fill + crisp dark ink outline)
    - Occlusion order: A_over_B vs B_over_A (pixel-identical outside the overlap region!)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PALETTE = [
    (226, 114, 134),  # anime coral pink
    (112, 168, 224),  # anime sky blue
    (142, 198, 138),  # anime mint green
    (234, 184, 102),  # anime golden hair
    (176, 138, 218),  # anime lavender hair
    (232, 146, 104),  # anime warm peach
]
INK = (34, 30, 42)
BG = (244, 242, 238)


def _bezier_points(p0, p1, p2, n=80) -> list[tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        x = u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0]
        y = u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]
        pts.append((x, y))
    return pts


def _stroke_ribbon_polygon(
    centerline: list[tuple[float, float]], half_width: float, taper: float = 0.25
) -> list[tuple[float, float]]:
    left_pts = []
    right_pts = []
    n = len(centerline)
    for i, (x, y) in enumerate(centerline):
        if i == 0:
            dx = centerline[1][0] - x
            dy = centerline[1][1] - y
        elif i == n - 1:
            dx = x - centerline[i - 1][0]
            dy = y - centerline[i - 1][1]
        else:
            dx = centerline[i + 1][0] - centerline[i - 1][0]
            dy = centerline[i + 1][1] - centerline[i - 1][1]
        norm = math.hypot(dx, dy) + 1e-8
        nx, ny = -dy / norm, dx / norm
        # Gentle taper toward ends (anime hair strand feel)
        s = i / max(1, n - 1)
        w = half_width * (1.0 - taper * ((2.0 * s - 1.0) ** 2))
        left_pts.append((x + nx * w, y + ny * w))
        right_pts.append((x - nx * w, y - ny * w))
    return left_pts + list(reversed(right_pts))


def _render_polygon_layer(
    size: int,
    poly: list[tuple[float, float]],
    color: tuple[int, int, int],
    shaded: bool,
    outline_w: int = 4,
) -> tuple[Image.Image, np.ndarray]:
    """Render a closed polygon onto an RGBA layer, with optional smooth 3D dome shading."""
    ss = 2
    S = size * ss
    poly_ss = [(x * ss, y * ss) for x, y in poly]

    mask_ss = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask_ss).polygon(poly_ss, fill=255)
    mask_img = mask_ss.resize((size, size), Image.BILINEAR)
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0

    # Crisp outline via eroded mask subtraction or stroked polygon
    edge_ss = Image.new("L", (S, S), 0)
    d_edge = ImageDraw.Draw(edge_ss)
    d_edge.polygon(poly_ss, outline=255, width=outline_w * ss)
    edge_img = edge_ss.resize((size, size), Image.BILINEAR)
    edge = (np.asarray(edge_img, dtype=np.float32) / 255.0) * (mask > 0.05)

    # Interior fill
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    rgb[:, :, 0] = color[0]
    rgb[:, :, 1] = color[1]
    rgb[:, :, 2] = color[2]

    if shaded:
        # Smooth distance-from-boundary dome shading + directional top-left light
        # Fast approximate interior distance via progressive Gaussian blurs of mask
        m_blur1 = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(radius=14)), dtype=np.float32) / 255.0
        m_blur2 = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(radius=30)), dtype=np.float32) / 255.0
        # Gradient of blurred mask approximates surface normal (nx, ny)
        gy, gx = np.gradient(0.6 * m_blur1 + 0.4 * m_blur2)
        nz = np.sqrt(np.clip(1.0 - (gx * 16.0) ** 2 - (gy * 16.0) ** 2, 0.08, 1.0))
        # Directional light from top-left (-0.55, -0.55, 0.63)
        ndotl = np.clip(-gx * 12.0 * 0.55 - gy * 12.0 * 0.55 + nz * 0.72, 0.25, 1.18)
        # Subtle specular lift in center
        spec = np.clip((ndotl - 0.85) * 0.8, 0.0, 0.25)
        for c in range(3):
            rgb[:, :, c] = np.clip(rgb[:, :, c] * (0.58 + 0.52 * ndotl) + spec * 75.0, 0, 255)
    else:
        # Blend crisp anime dark ink outline on L3_Flat2D
        for c in range(3):
            rgb[:, :, c] = rgb[:, :, c] * (1.0 - edge) + INK[c] * edge

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA"), (mask > 0.85)


def _compose_two_layers(
    size: int,
    layer_back: Image.Image,
    layer_front: Image.Image,
    shaded: bool,
) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (*BG, 255))
    canvas = Image.alpha_composite(canvas, layer_back)
    if shaded:
        # Soft directional contact shadow cast by front onto back & canvas
        alpha_front = layer_front.split()[3]
        shadow = Image.new("RGBA", (size, size), (22, 18, 32, 0))
        blurred = alpha_front.filter(ImageFilter.GaussianBlur(radius=11))
        arr = (np.asarray(blurred, dtype=np.float32) * 0.42).astype(np.uint8)
        shadow_alpha = Image.fromarray(arr, mode="L")
        shadow.putalpha(shadow_alpha)
        shifted = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        shifted.paste(shadow, (9, 10))
        canvas = Image.alpha_composite(canvas, shifted)
    canvas = Image.alpha_composite(canvas, layer_front)
    return canvas.convert("RGB")


def _pick_safe_interior_point(
    mask_self: np.ndarray, mask_other: np.ndarray, rng: random.Random, min_margin: int = 16
) -> tuple[int, int]:
    """Pick a point deep inside mask_self and far from mask_other and image borders."""
    h, w = mask_self.shape
    # Erode self & dilate other via simple box sum / PIL MaxFilter/MinFilter
    self_img = Image.fromarray((mask_self.astype(np.uint8) * 255), mode="L")
    other_img = Image.fromarray((mask_other.astype(np.uint8) * 255), mode="L")
    self_eroded = np.asarray(self_img.filter(ImageFilter.MinFilter(min_margin * 2 + 1))) > 200
    other_dilated = np.asarray(other_img.filter(ImageFilter.MaxFilter(min_margin * 2 + 1))) > 50
    valid = self_eroded & (~other_dilated)
    valid[:24, :] = False
    valid[-24:, :] = False
    valid[:, :24] = False
    valid[:, -24:] = False
    ys, xs = np.where(valid)
    if len(xs) == 0:
        # Fallback to slightly smaller margin
        self_eroded = np.asarray(self_img.filter(ImageFilter.MinFilter(13))) > 200
        other_dilated = np.asarray(other_img.filter(ImageFilter.MaxFilter(15))) > 50
        ys, xs = np.where(self_eroded & (~other_dilated))
    # Pick near the centroid of valid pixels (or closest valid pixel to crossing-adjacent region)
    cx, cy = w / 2.0, h / 2.0
    dists = (xs - cx) ** 2 + (ys - cy) ** 2
    # Prefer points ~80-160px from center (not at extreme corner, not touching overlap)
    score = np.abs(np.sqrt(dists) - 115.0)
    idx = int(np.argmin(score))
    return int(xs[idx]), int(ys[idx])


def _make_curved_weave_pair(size: int, rng: random.Random) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    cx = size * 0.5 + rng.uniform(-24, 24)
    cy = size * 0.5 + rng.uniform(-24, 24)
    ang_a = rng.uniform(-0.75, 0.75)
    ang_b = ang_a + rng.uniform(0.95, 2.15)
    L = size * 0.44
    hw_a = rng.uniform(30, 44)
    hw_b = rng.uniform(30, 44)

    def curve_poly(ang: float, hw: float, bend: float) -> list[tuple[float, float]]:
        dx, dy = math.cos(ang), math.sin(ang)
        nx, ny = -dy, dx
        p0 = (cx - dx * L + nx * bend * 0.35, cy - dy * L + ny * bend * 0.35)
        p1 = (cx - nx * bend, cy - ny * bend)
        p2 = (cx + dx * L + nx * bend * 0.35, cy + dy * L + ny * bend * 0.35)
        cl = _bezier_points(p0, p1, p2, n=72)
        return _stroke_ribbon_polygon(cl, half_width=hw, taper=rng.uniform(0.18, 0.38))

    poly_a = curve_poly(ang_a, hw_a, rng.uniform(-78, 78))
    poly_b = curve_poly(ang_b, hw_b, rng.uniform(-78, 78))
    return poly_a, poly_b


def _rounded_rect_poly(cx: float, cy: float, rx: float, ry: float, angle: float, n: int = 64) -> list[tuple[float, float]]:
    # Superellipse (squircle) rotated by angle
    pts = []
    ca, sa = math.cos(angle), math.sin(angle)
    for i in range(n):
        t = 2.0 * math.pi * i / n
        ct, st = math.cos(t), math.sin(t)
        x0 = math.copysign(abs(ct) ** 0.55, ct) * rx
        y0 = math.copysign(abs(st) ** 0.55, st) * ry
        x = cx + x0 * ca - y0 * sa
        y = cy + x0 * sa + y0 * ca
        pts.append((x, y))
    return pts


def _make_cel_shapes_pair(size: int, rng: random.Random) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    cx, cy = size * 0.5, size * 0.5
    sep_ang = rng.uniform(0, 2 * math.pi)
    sep = rng.uniform(52, 74)
    ax = cx - math.cos(sep_ang) * sep
    ay = cy - math.sin(sep_ang) * sep
    bx = cx + math.cos(sep_ang) * sep
    by = cy + math.sin(sep_ang) * sep
    poly_a = _rounded_rect_poly(ax, ay, rng.uniform(86, 118), rng.uniform(76, 108), rng.uniform(-0.8, 0.8))
    poly_b = _rounded_rect_poly(bx, by, rng.uniform(86, 118), rng.uniform(76, 108), rng.uniform(-0.8, 0.8))
    return poly_a, poly_b


def generate_ood_suite(out_dir: Path, size: int = 512, n_per_family: int = 20, seed: int = 2026) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    stim_dir = out_dir / "stimuli"
    stim_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for family in ["curved_weave", "cel_shapes"]:
        for idx in range(n_per_family):
            pair_id = f"{family}_{idx:02d}"
            regime = "identical_color" if idx < (n_per_family // 2) else "distinct_color"
            c1 = rng.choice(PALETTE)
            if regime == "identical_color":
                c2 = c1
            else:
                c2 = rng.choice([c for c in PALETTE if c != c1])

            if family == "curved_weave":
                poly_a, poly_b = _make_curved_weave_pair(size, rng)
            else:
                poly_a, poly_b = _make_cel_shapes_pair(size, rng)

            # Precompute masks once to pick exact identical sample points across all 4 variants
            _, mask_a = _render_polygon_layer(size, poly_a, c1, shaded=False)
            _, mask_b = _render_polygon_layer(size, poly_b, c2, shaded=False)
            pt_a = _pick_safe_interior_point(mask_a, mask_b, rng)
            pt_b = _pick_safe_interior_point(mask_b, mask_a, rng)

            for render_level, shaded in [("L1_Shaded", True), ("L3_Flat2D", False)]:
                layer_a, _ = _render_polygon_layer(size, poly_a, c1, shaded=shaded)
                layer_b, _ = _render_polygon_layer(size, poly_b, c2, shaded=shaded)
                for order, front_gt in [("A_over_B", "A"), ("B_over_A", "B")]:
                    if order == "A_over_B":
                        img = _compose_two_layers(size, layer_b, layer_a, shaded=shaded)
                    else:
                        img = _compose_two_layers(size, layer_a, layer_b, shaded=shaded)
                    img_path = stim_dir / f"{pair_id}_{render_level}_{order}.png"
                    img.save(img_path)

                    # Dot-annotated copy for VLM eval
                    dots = img.copy()
                    d = ImageDraw.Draw(dots)
                    ax, ay = pt_a
                    bx, by = pt_b
                    d.ellipse([ax - 11, ay - 11, ax + 11, ay + 11], fill=(235, 36, 36), outline=(255, 255, 255), width=3)
                    d.ellipse([bx - 11, by - 11, bx + 11, by + 11], fill=(32, 108, 240), outline=(255, 255, 255), width=3)
                    dots_path = stim_dir / f"{pair_id}_{render_level}_{order}_dots.png"
                    dots.save(dots_path)

                    records.append(
                        {
                            "pair_id": pair_id,
                            "family": family,
                            "regime": regime,
                            "render_level": render_level,
                            "order": order,
                            "front_gt": front_gt,
                            "pt_a": list(pt_a),
                            "pt_b": list(pt_b),
                            "image_path": str(img_path),
                            "dots_image_path": str(dots_path),
                        }
                    )

    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"Generated {len(records)} OOD stimuli -> {manifest}")
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-per-family", type=int, default=20)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generate_ood_suite(args.out_dir, n_per_family=args.n_per_family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
