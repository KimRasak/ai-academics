#!/usr/bin/env python3
"""
Exp 3D (P2 Hard Case — Pre-Lit Anime Double-Shadow Removal Test):
  1. Generates 6 full-color TV-anime portraits WITH strong pre-existing left-side
     cel shadows (using Animagine-XL-4.0).
  2. Relights them from the OPPOSITE direction (key light from the LEFT, shadow on
     the RIGHT — directly conflicting with the pre-existing baked cel shadow!).
  3. Compares:
       - `naive_depth_cel`: casts new right-side cel shadow WITHOUT delighting
         (quantifies & visualizes the Double-Shadow artifact!)
       - `albedo_norm_depth_cel`: performs chroma-preserving 2D cel delighting
         (lifting cool-shaded cel plateaus toward each segment's lit base chroma
         while preserving dark ink lines & eyes) BEFORE casting the new right-side
         normal-flattened cel shadow!
       - Measures shadow-flip consistency (left_luma - right_luma sign reversal!)
         + ink_edge_retention + plateau_gradient_pollution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageDraw
from transformers import pipeline

from exp3b_p2_cel_relight_opt import _blur_f32, _rgb_to_luma, compute_cel_metrics, depth_guided_cel_relight


PRELIT_PROMPTS = [
    ("pl_00_blue_girl", "1girl, solo, blue bob hair, cyan eyes, upper body portrait, dramatic side lighting from right, strong cel shadow on left half of face, anime coloring, vibrant full color, simple white background, masterpiece, high score"),
    ("pl_01_pink_girl", "1girl, solo, pink twin tails hair, green eyes, upper body portrait, dramatic side lighting from right, strong cel shadow on left half of face, anime coloring, vibrant full color, simple white background, masterpiece, high score"),
    ("pl_02_silver_boy", "1boy, solo, silver hair, amber eyes, upper body portrait, dramatic side lighting from right, strong cel shadow on left half of face, anime coloring, vibrant full color, simple white background, masterpiece, high score"),
    ("pl_03_blonde_girl", "1girl, solo, golden blonde hair, blue eyes, upper body portrait, dramatic side lighting from right, strong cel shadow on left half of face, anime coloring, vibrant full color, simple white background, masterpiece, high score"),
]
NEG = "sketch, lineart, monochrome, 3d, photorealistic, blurry, bad anatomy, text, watermark"


def delight_anime_cel_albedo(img: Image.Image, depth_norm: np.ndarray) -> Image.Image:
    """
    Lightweight 2D anime cel shadow lifter (albedo normalization):
      Identifies non-ink pixels that are noticeably darker than the local macro-lit
      envelope of similar hue (pre-baked cel shadow bands on skin/hair/clothes)
      and softly lifts them toward the base lit chroma while strictly protecting
      dark ink contours (luma < 65) and dark eye pupils!
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    luma = _rgb_to_luma(arr)

    # Local high-percentile / smooth bright envelope via max-pooled smoothed luma
    # Approximates the unshaded base luminance of the surrounding material region
    luma_smooth = _blur_f32(luma, radius=28.0)
    # Estimate local lit reference as smooth(max(luma, luma_smooth))
    lit_env = _blur_f32(np.maximum(luma, luma_smooth * 1.06), radius=24.0)
    lit_env = np.maximum(lit_env, 1.0)

    # Protect dark ink outlines, black clothing, and very bright background
    ink_mask = np.clip((luma - 68.0) / 38.0, 0.0, 1.0)
    bg_mask = np.clip((242.0 - luma) / 15.0, 0.0, 1.0)
    # Only lift moderate cel-shadow dips (ratio in [0.55, 0.96]) — never blow out true dark hair/pupils!
    ratio = np.clip(luma / lit_env, 0.2, 1.2)
    is_cel_shadow = np.clip((0.96 - ratio) / 0.14, 0.0, 1.0) * np.clip((ratio - 0.50) / 0.15, 0.0, 1.0)

    # Gain to bring pre-baked shadow band back toward lit base albedo (capped at 1.32x to keep chroma clean)
    gain = 1.0 + is_cel_shadow * ink_mask * bg_mask * np.clip((lit_env / np.maximum(luma, 1.0)) - 1.0, 0.0, 0.34)
    # Slight warm de-tinting when lifting cool-tinted anime shadows
    out = np.empty_like(arr)
    out[:, :, 0] = np.clip(arr[:, :, 0] * (gain ** 1.08), 0, 255)
    out[:, :, 1] = np.clip(arr[:, :, 1] * (gain ** 1.02), 0, 255)
    out[:, :, 2] = np.clip(arr[:, :, 2] * (gain ** 0.94), 0, 255)
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animagine-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/animagine-xl-4.0")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = args.out_dir / "prelit_inputs"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate 4 strongly pre-lit anime portraits (key light from RIGHT, baked shadow on LEFT)
    existing = sorted(gen_dir.glob("pl_*.png"))
    if len(existing) < len(PRELIT_PROMPTS):
        pipe_gen = DiffusionPipeline.from_pretrained(args.animagine_dir, torch_dtype=torch.float16, use_safetensors=True)
        pipe_gen.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_gen.scheduler.config)
        pipe_gen = pipe_gen.to("cuda")
        for idx, (cid, prompt) in enumerate(PRELIT_PROMPTS):
            img = pipe_gen(
                prompt=prompt, negative_prompt=NEG, width=768, height=768,
                guidance_scale=5.5, num_inference_steps=26,
                generator=torch.Generator(device="cuda").manual_seed(9001 + idx * 19),
            ).images[0]
            img.save(gen_dir / f"{cid}.png")
        del pipe_gen
        torch.cuda.empty_cache()

    # 2. Relight from OPPOSITE side (key light from LEFT -> light_dir=(-0.86, -0.22, 0.45))
    depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=0)
    rows = []
    for p in sorted(gen_dir.glob("pl_*.png")):
        cid = p.stem
        orig = Image.open(p).convert("RGB").resize((768, 768), Image.LANCZOS)
        depth = np.squeeze(np.asarray(depth_pipe(orig)["predicted_depth"], dtype=np.float32))
        if depth.shape != (768, 768):
            depth = np.asarray(Image.fromarray(depth, mode="F").resize((768, 768), Image.BILINEAR), dtype=np.float32)
        d_norm = (depth - float(depth.min())) / max(1e-6, float(depth.max() - depth.min()))

        delit = delight_anime_cel_albedo(orig, d_norm)
        naive_flip = depth_guided_cel_relight(orig, d_norm, light_dir=(-0.86, -0.22, 0.45), flatten_alpha=0.80, terminator=0.36)
        albedo_flip = depth_guided_cel_relight(delit, d_norm, light_dir=(-0.86, -0.22, 0.45), flatten_alpha=0.80, terminator=0.36)

        # Measure left-minus-right face luminance balance (should flip from negative -> positive!)
        def lr_balance(im: Image.Image) -> float:
            l = _rgb_to_luma(np.asarray(im, dtype=np.float32))
            fg = (l < 235.0) & (l > 55.0)
            h, w = l.shape
            left = l[h // 4 : 3 * h // 4, w // 4 : w // 2][fg[h // 4 : 3 * h // 4, w // 4 : w // 2]]
            right = l[h // 4 : 3 * h // 4, w // 2 : 3 * w // 4][fg[h // 4 : 3 * h // 4, w // 2 : 3 * w // 4]]
            return round(float(left.mean() - right.mean()), 2)

        row = {
            "char_id": cid,
            "orig_L_minus_R": lr_balance(orig),
            "delit_L_minus_R": lr_balance(delit),
            "naive_flip_L_minus_R": lr_balance(naive_flip),
            "albedo_flip_L_minus_R": lr_balance(albedo_flip),
            "naive_metrics": compute_cel_metrics(orig, naive_flip),
            "albedo_metrics": compute_cel_metrics(orig, albedo_flip),
        }
        rows.append(row)

        panels = [("prelit_orig (R-lit)", orig), ("delit_albedo", delit), ("naive_L-lit (2x shd?)", naive_flip), ("albedo_norm_L-lit", albedo_flip)]
        sheet = Image.new("RGB", (256 * 4, 256))
        for i, (lab, im) in enumerate(panels):
            t = im.resize((256, 256), Image.LANCZOS)
            ImageDraw.Draw(t).rectangle([0, 0, 256, 24], fill=(20, 20, 28))
            ImageDraw.Draw(t).text((6, 5), lab, fill=(255, 255, 255))
            sheet.paste(t, (i * 256, 0))
        sheet.save(args.out_dir / f"{cid}__prelit_flip_sheet.jpg", quality=92)
        print(f"{cid}: orig={row['orig_L_minus_R']} -> naive={row['naive_flip_L_minus_R']} vs albedo_norm={row['albedo_flip_L_minus_R']}", flush=True)

    (args.out_dir / "prelit_double_shadow_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
