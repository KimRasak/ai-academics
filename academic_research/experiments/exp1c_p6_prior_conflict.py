#!/usr/bin/env python3
"""
Exp 1C (P6 Sharpened Stress Test): When Does Flat-2D T-Junction Perception Break?

Wave-1 (exp1_p6_tjunction_pilot.py) falsified the naive hypothesis:
  Depth-Anything-V2-Large / Metric / Small all scored 100% (20/20) on isolated
  high-contrast thick-ink 2-ribbon crossings, BUT their signed flip depth delta
  collapsed by -55% to -63% when shading was removed (L1 -> L3).

This script directly tests 5 realistic 2D-anime stress conditions where a weakened
T-junction signal may be overridden by competing monocular priors or cel-style traits:
  1. `lineless_flat2d`: Distinct-color flat-2D ribbons with ZERO dark ink outline
     (tests whether DA2 relied on dark ink continuity rather than true T-junctions).
  2. `thin_ink_samecolor`: Identical-color flat-2D ribbons with thin 1.5px anime lineart.
  3. `size_conflict`: Small flat-2D shape (r=44) vs Large flat-2D shape (r=135) overlapping.
     In real 3D scenes, larger apparent size strongly implies closer; when small is IN FRONT
     of large (A_over_B), flat T-junction must override the strong size prior.
  4. `elevation_conflict`: Upper shape vs Lower shape overlapping vertically (in real 3D
     scenes, lower-in-image = closer ground-plane prior; when upper is IN FRONT of lower,
     flat T-junction must override the ground-plane prior).
  5. `curved_anime_weave`: Multi-crossing curved tapered 2D anime hair-like ribbons
     (S-curves with cel ink), testing paired flip under complex geometry.
  Plus every condition has an exact matched `L1_Shaded` control!
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
    (226, 118, 138),
    (112, 168, 224),
    (142, 198, 138),
    (234, 184, 102),
    (176, 138, 218),
    (232, 146, 104),
]
INK = (34, 30, 42)
BG = (244, 242, 238)


def _rect_poly(cx: float, cy: float, length: float, width: float, angle_deg: float) -> list[tuple[float, float]]:
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    hl, hw = length / 2.0, width / 2.0
    corners = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in corners]


def _squircle_poly(cx: float, cy: float, rx: float, ry: float, angle_deg: float = 0.0, n: int = 72) -> list[tuple[float, float]]:
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        ct, st = math.cos(t), math.sin(t)
        x0 = math.copysign(abs(ct) ** 0.65, ct) * rx
        y0 = math.copysign(abs(st) ** 0.65, st) * ry
        pts.append((cx + x0 * ca - y0 * sa, cy + x0 * sa + y0 * ca))
    return pts


def _curved_ribbon_poly(
    cx: float, cy: float, angle_deg: float, half_len: float, half_w: float, bend: float
) -> list[tuple[float, float]]:
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    nx, ny = -dy, dx
    p0 = (cx - dx * half_len + nx * bend * 0.4, cy - dy * half_len + ny * bend * 0.4)
    p1 = (cx - nx * bend, cy - ny * bend)
    p2 = (cx + dx * half_len + nx * bend * 0.4, cy + dy * half_len + ny * bend * 0.4)
    cl = []
    n = 72
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        cl.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0], u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))
    left, right = [], []
    for i, (x, y) in enumerate(cl):
        if i == 0:
            tx, ty = cl[1][0] - x, cl[1][1] - y
        elif i == n:
            tx, ty = x - cl[i - 1][0], y - cl[i - 1][1]
        else:
            tx, ty = cl[i + 1][0] - cl[i - 1][0], cl[i + 1][1] - cl[i - 1][1]
        nm = math.hypot(tx, ty) + 1e-8
        px, py = -ty / nm, tx / nm
        s = i / n
        w = half_w * (1.0 - 0.30 * ((2.0 * s - 1.0) ** 2))
        left.append((x + px * w, y + py * w))
        right.append((x - px * w, y - py * w))
    return left + list(reversed(right))


def _render_layer(
    size: int,
    poly: list[tuple[float, float]],
    color: tuple[int, int, int],
    shaded: bool,
    outline_w: float,
) -> tuple[Image.Image, np.ndarray]:
    ss = 2
    S = size * ss
    poly_ss = [(x * ss, y * ss) for x, y in poly]

    mask_ss = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask_ss).polygon(poly_ss, fill=255)
    mask_img = mask_ss.resize((size, size), Image.BILINEAR)
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0

    rgb = np.zeros((size, size, 3), dtype=np.float32)
    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = color

    if shaded:
        m1 = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(radius=14)), dtype=np.float32) / 255.0
        m2 = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(radius=32)), dtype=np.float32) / 255.0
        gy, gx = np.gradient(0.6 * m1 + 0.4 * m2)
        nz = np.sqrt(np.clip(1.0 - (gx * 16.0) ** 2 - (gy * 16.0) ** 2, 0.08, 1.0))
        ndotl = np.clip(-gx * 12.0 * 0.55 - gy * 12.0 * 0.55 + nz * 0.72, 0.25, 1.2)
        spec = np.clip((ndotl - 0.85) * 0.85, 0.0, 0.28)
        for c in range(3):
            rgb[:, :, c] = np.clip(rgb[:, :, c] * (0.55 + 0.55 * ndotl) + spec * 80.0, 0, 255)
    elif outline_w > 0:
        edge_ss = Image.new("L", (S, S), 0)
        ImageDraw.Draw(edge_ss).polygon(poly_ss, outline=255, width=max(1, int(round(outline_w * ss))))
        edge = (np.asarray(edge_ss.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0) * (mask > 0.05)
        for c in range(3):
            rgb[:, :, c] = rgb[:, :, c] * (1.0 - edge) + INK[c] * edge

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA"), (mask > 0.85)


def _compose(size: int, back: Image.Image, front: Image.Image, shaded: bool) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (*BG, 255))
    canvas = Image.alpha_composite(canvas, back)
    if shaded:
        alpha_f = front.split()[3]
        sh = Image.new("RGBA", (size, size), (22, 18, 32, 0))
        bl = alpha_f.filter(ImageFilter.GaussianBlur(radius=12))
        sh.putalpha(Image.fromarray((np.asarray(bl, dtype=np.float32) * 0.45).astype(np.uint8), mode="L"))
        shifted = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        shifted.paste(sh, (10, 11))
        canvas = Image.alpha_composite(canvas, shifted)
    canvas = Image.alpha_composite(canvas, front)
    return canvas.convert("RGB")


def _safe_point(mask_self: np.ndarray, mask_other: np.ndarray, min_margin: int = 12) -> tuple[int, int]:
    h, w = mask_self.shape
    for m in [min_margin, 8, 5, 3]:
        s_img = Image.fromarray((mask_self.astype(np.uint8) * 255), mode="L")
        o_img = Image.fromarray((mask_other.astype(np.uint8) * 255), mode="L")
        s_er = np.asarray(s_img.filter(ImageFilter.MinFilter(m * 2 + 1))) > 200
        o_di = np.asarray(o_img.filter(ImageFilter.MaxFilter(m * 2 + 1))) > 50
        valid = s_er & (~o_di)
        valid[:16, :] = False
        valid[-16:, :] = False
        valid[:, :16] = False
        valid[:, -16:] = False
        ys, xs = np.where(valid)
        if len(xs) > 0:
            # Pick pixel closest to centroid of valid region
            mx, my = float(np.mean(xs)), float(np.mean(ys))
            idx = int(np.argmin((xs - mx) ** 2 + (ys - my) ** 2))
            return int(xs[idx]), int(ys[idx])
    ys, xs = np.where(mask_self & (~mask_other))
    idx = len(xs) // 2
    return int(xs[idx]), int(ys[idx])


def generate_suite(out_dir: Path, n_per_cond: int = 16, size: int = 512, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    stim_dir = out_dir / "stimuli"
    stim_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    conditions = [
        "lineless_flat2d",
        "thin_ink_samecolor",
        "size_conflict",
        "elevation_conflict",
        "curved_anime_weave",
    ]

    for cond in conditions:
        for i in range(n_per_cond):
            pair_id = f"{cond}_{i:02d}"
            c1 = rng.choice(PALETTE)
            c2 = rng.choice([c for c in PALETTE if c != c1])
            outline_w = 3.5

            if cond == "lineless_flat2d":
                ang_a = rng.uniform(-55, 55)
                ang_b = ang_a + rng.uniform(55, 125)
                poly_a = _rect_poly(256, 256, 400, rng.uniform(62, 86), ang_a)
                poly_b = _rect_poly(256, 256, 400, rng.uniform(62, 86), ang_b)
                outline_w = 0.0
            elif cond == "thin_ink_samecolor":
                c2 = c1
                ang_a = rng.uniform(-55, 55)
                ang_b = ang_a + rng.uniform(55, 125)
                poly_a = _rect_poly(256, 256, 400, rng.uniform(60, 84), ang_a)
                poly_b = _rect_poly(256, 256, 400, rng.uniform(60, 84), ang_b)
                outline_w = 1.5
            elif cond == "size_conflict":
                # A is SMALL (r~48), B is LARGE (r~142); overlap near edge of B
                ang = rng.uniform(0, 2 * math.pi)
                bx, by = 256 - math.cos(ang) * 32, 256 - math.sin(ang) * 32
                ax, ay = bx + math.cos(ang) * 120, by + math.sin(ang) * 120
                poly_a = _squircle_poly(ax, ay, rng.uniform(44, 54), rng.uniform(42, 52), rng.uniform(-30, 30))
                poly_b = _squircle_poly(bx, by, rng.uniform(132, 152), rng.uniform(128, 146), rng.uniform(-30, 30))
                outline_w = 0.0 if (i % 2 == 0) else 2.5
            elif cond == "elevation_conflict":
                # A is UPPER (y~165), B is LOWER (y~335); ground-plane prior says B (lower) is closer
                dx = rng.uniform(-32, 32)
                poly_a = _squircle_poly(256 - dx, 178, rng.uniform(96, 118), rng.uniform(86, 102), rng.uniform(-15, 15))
                poly_b = _squircle_poly(256 + dx, 330, rng.uniform(96, 118), rng.uniform(86, 102), rng.uniform(-15, 15))
                outline_w = 0.0 if (i % 2 == 0) else 2.5
            elif cond == "curved_anime_weave":
                if i < n_per_cond // 2:
                    c2 = c1
                ang_a = rng.uniform(-50, 50)
                ang_b = ang_a + rng.uniform(60, 120)
                poly_a = _curved_ribbon_poly(256, 256, ang_a, 225, rng.uniform(32, 44), rng.uniform(-85, 85))
                poly_b = _curved_ribbon_poly(256, 256, ang_b, 225, rng.uniform(32, 44), rng.uniform(-85, 85))
                outline_w = 2.5
            else:
                raise ValueError(cond)

            _, mask_a = _render_layer(size, poly_a, c1, shaded=False, outline_w=0.0)
            _, mask_b = _render_layer(size, poly_b, c2, shaded=False, outline_w=0.0)
            pt_a = _safe_point(mask_a, mask_b)
            pt_b = _safe_point(mask_b, mask_a)

            for render_level, shaded in [("L1_Shaded", True), ("L3_Flat2D", False)]:
                la, _ = _render_layer(size, poly_a, c1, shaded=shaded, outline_w=outline_w)
                lb, _ = _render_layer(size, poly_b, c2, shaded=shaded, outline_w=outline_w)
                for order, front_gt in [("A_over_B", "A"), ("B_over_A", "B")]:
                    img = _compose(size, lb, la, shaded=shaded) if order == "A_over_B" else _compose(size, la, lb, shaded=shaded)
                    img_path = stim_dir / f"{pair_id}__{render_level}__{order}.png"
                    img.save(img_path)

                    dots = img.copy()
                    d = ImageDraw.Draw(dots)
                    ax, ay = pt_a
                    bx, by = pt_b
                    d.ellipse([ax - 10, ay - 10, ax + 10, ay + 10], fill=(235, 36, 36), outline=(255, 255, 255), width=3)
                    d.ellipse([bx - 10, by - 10, bx + 10, by + 10], fill=(32, 108, 240), outline=(255, 255, 255), width=3)
                    dots_path = stim_dir / f"{pair_id}__{render_level}__{order}__dots.png"
                    dots.save(dots_path)

                    records.append(
                        {
                            "pair_id": pair_id,
                            "condition": cond,
                            "outline_w": outline_w,
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
    print(f"Generated {len(records)} stress-test stimuli -> {manifest}")
    return records


def _sample_patch_median(depth_map: np.ndarray, pt: tuple[int, int], radius: int = 6) -> float:
    h, w = depth_map.shape[:2]
    x, y = pt
    return float(np.median(depth_map[max(0, y - radius) : min(h, y + radius + 1), max(0, x - radius) : min(w, x + radius + 1)]))


def eval_depth(records: list[dict[str, Any]], model_id: str, out_dir: Path, higher_is_closer: bool = True) -> None:
    import torch
    from transformers import pipeline

    pipe = pipeline("depth-estimation", model=model_id, device=0 if torch.cuda.is_available() else -1)
    preds: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        depth = np.squeeze(np.asarray(pipe(img)["predicted_depth"], dtype=np.float32))
        if depth.shape != (img.height, img.width):
            depth = np.asarray(Image.fromarray(depth, mode="F").resize((img.width, img.height), resample=Image.BILINEAR), dtype=np.float32)
        da = _sample_patch_median(depth, tuple(rec["pt_a"]))
        db = _sample_patch_median(depth, tuple(rec["pt_b"]))
        pred = "A" if ((da > db) if higher_is_closer else (da < db)) else "B"
        preds[(rec["pair_id"], rec["render_level"], rec["order"])] = {
            "da": da,
            "db": db,
            "delta_ab": da - db,
            "pred": pred,
            "correct": pred == rec["front_gt"],
            "condition": rec["condition"],
        }

    summary: dict[str, Any] = {"model_id": model_id, "higher_is_closer": higher_is_closer, "slices": {}}
    conds = sorted({r["condition"] for r in records})
    for cond in conds:
        for rl in ["L1_Shaded", "L3_Flat2D"]:
            pids = sorted({r["pair_id"] for r in records if r["condition"] == cond and r["render_level"] == rl})
            s_ok, f_ok, a_ok, b_ok = 0, 0, 0, 0
            s_flips = []
            sign = 1.0 if higher_is_closer else -1.0
            for pid in pids:
                p1 = preds[(pid, rl, "A_over_B")]
                p2 = preds[(pid, rl, "B_over_A")]
                a_ok += int(p1["correct"])
                b_ok += int(p2["correct"])
                s_ok += int(p1["correct"]) + int(p2["correct"])
                if p1["correct"] and p2["correct"]:
                    f_ok += 1
                s_flips.append(sign * (p1["delta_ab"] - p2["delta_ab"]))
            summary["slices"][f"{cond}__{rl}"] = {
                "n_pairs": len(pids),
                "acc_A_over_B": round(a_ok / max(1, len(pids)), 4),
                "acc_B_over_A": round(b_ok / max(1, len(pids)), 4),
                "single_acc": round(s_ok / max(1, 2 * len(pids)), 4),
                "paired_flip": round(f_ok / max(1, len(pids)), 4),
                "signed_flip_delta": round(float(np.mean(s_flips)), 5),
            }

    out_path = out_dir / f"stress_depth_{model_id.replace('/', '_')}.json"
    out_path.write_text(json.dumps({"summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--eval-depth", type=str, default=None)
    ap.add_argument("--lower-is-closer", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.jsonl"
    if args.skip_gen and manifest.exists():
        records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        records = generate_suite(args.out_dir)
    if args.eval_depth:
        eval_depth(records, args.eval_depth, args.out_dir, higher_is_closer=not args.lower_is_closer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
