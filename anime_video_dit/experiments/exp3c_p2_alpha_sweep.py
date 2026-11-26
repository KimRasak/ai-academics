#!/usr/bin/env python3
"""
Exp 3C (P2 Ablation): Normal-Flattening Control (alpha in {0.0, 0.35, 0.70, 1.0})
& Macro-Cheek + Hair-Bang Cast Shadow Sweep on Full-Color TV-Anime Portraits.

Generates side-by-side ablation sheets showing:
  orig | Qwen-Image-Edit-2511 (3D airbrush collapse) | alpha=0.0 (bumpy 3D nose cel)
  | alpha=0.55 (studio balanced cel) | alpha=0.85 (flat-face anime cel + bang cast shadow)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import pipeline

from exp3b_p2_cel_relight_opt import _blur_f32, _rgb_to_luma, compute_cel_metrics, depth_guided_cel_relight


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portraits-dir", type=Path, required=True)
    ap.add_argument("--edit-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=0 if torch.cuda.is_available() else -1)

    for p in sorted(args.portraits_dir.glob("*.png"))[:4]:
        cid = p.stem
        orig = Image.open(p).convert("RGB").resize((768, 768), Image.LANCZOS)
        depth = np.squeeze(np.asarray(pipe(orig)["predicted_depth"], dtype=np.float32))
        if depth.shape != (768, 768):
            depth = np.asarray(Image.fromarray(depth, mode="F").resize((768, 768), Image.BILINEAR), dtype=np.float32)
        depth_norm = (depth - float(depth.min())) / max(1e-6, float(depth.max() - depth.min()))

        panels = [("orig", orig)]
        qw_path = args.edit_dir / f"{cid}__p2_relight_cel_right.png"
        if qw_path.exists():
            panels.append(("Qwen-Edit (3D collapse)", Image.open(qw_path).convert("RGB")))

        for alpha, term, lab in [
            (0.0, 0.36, "alpha=0.0 (3D noisy cel)"),
            (0.45, 0.36, "alpha=0.45 (balanced)"),
            (0.82, 0.36, "alpha=0.82 (flat anime)"),
        ]:
            relit = depth_guided_cel_relight(
                orig,
                depth_norm,
                light_dir=(0.86, -0.22, 0.45),
                flatten_alpha=alpha,
                terminator=term,
                cast_shadow_steps=20,
            )
            panels.append((lab, relit))

        sheet = Image.new("RGB", (256 * len(panels), 256))
        for i, (lab, im) in enumerate(panels):
            t = im.resize((256, 256), Image.LANCZOS)
            ImageDraw.Draw(t).rectangle([0, 0, 256, 24], fill=(20, 20, 28))
            ImageDraw.Draw(t).text((6, 5), lab, fill=(255, 255, 255))
            sheet.paste(t, (i * 256, 0))
        sheet.save(args.out_dir / f"{cid}__alpha_sweep.jpg", quality=92)
        print(f"Saved alpha sweep -> {cid}__alpha_sweep.jpg", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
