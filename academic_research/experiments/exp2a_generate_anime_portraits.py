#!/usr/bin/env python3
"""
Exp 2A / 3A: Generate 12 copyright-free, studio-quality 2D cel-shaded anime portraits
using open-weight Animagine XL 4.0 (/gluster_hdd_cv/user/jinzili/weights/animagine-xl-4.0).

These clean, license-safe synthetic 2D anime portraits serve BOTH:
  - Exp 2 (P3: Non-Biological Anime Affect Symbols — pixel-exact counterfactual toggles
    for eye highlights OFF, forehead dread shadow, blush hatch lines, sweat drop, placebo hairpin)
  - Exp 3 (P2: True 2D Cel-Shading Relighting vs IC-Light / 10-line strong baseline)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

CHARACTERS = [
    ("char_00_blue_bob", "1girl, solo, short navy blue bob hair, bright cyan anime eyes with clear white catchlights, cheerful bright open smile, school blazer collar"),
    ("char_01_pink_twintails", "1girl, solo, pastel pink twin tails hair, big emerald green anime eyes with clear white catchlights, happy gentle smile, white blouse"),
    ("char_02_silver_boy", "1boy, solo, messy silver hair, vivid amber anime eyes with clear white catchlights, warm friendly smile, dark jacket collar"),
    ("char_03_blonde_long", "1girl, solo, long golden blonde hair, sapphire blue anime eyes with clear white catchlights, bright joyful smile, ribbon tie"),
    ("char_04_black_hime", "1girl, solo, straight black hime-cut hair, ruby red anime eyes with clear white catchlights, gentle cheerful smile, cardigan"),
    ("char_05_orange_short", "1girl, solo, energetic orange short hair, bright teal anime eyes with clear white catchlights, happy beaming smile, hoodie"),
    ("char_06_violet_glasses", "1girl, solo, lavender wavy hair, violet anime eyes with clear white catchlights, sweet happy smile, sweater"),
    ("char_07_brown_boy", "1boy, solo, neat chestnut brown hair, warm hazel anime eyes with clear white catchlights, cheerful friendly smile, collared shirt"),
    ("char_08_mint_braid", "1girl, solo, mint green side-braid hair, golden yellow anime eyes with clear white catchlights, bright happy smile, sailor collar"),
    ("char_09_crimson_ponytail", "1girl, solo, crimson red high ponytail, bright blue anime eyes with clear white catchlights, confident happy smile, turtleneck"),
    ("char_10_white_short", "1girl, solo, snowy white short hair, rose pink anime eyes with clear white catchlights, soft joyful smile, scarf"),
    ("char_11_darkteal_boy", "1boy, solo, dark teal hair, bright turquoise anime eyes with clear white catchlights, cheerful open smile, varsity jacket"),
]

COMMON_POS = (
    "portrait, close-up face, front view, looking straight at viewer, "
    "clean 2d anime cel shading, crisp dark ink lineart, flat color regions, "
    "bright even frontal studio lighting, clean plain off-white background, "
    "masterpiece, high score, great score, absurdres"
)
COMMON_NEG = (
    "3d render, realistic, photorealistic, heavy shadow, dark background, side view, "
    "closed eyes, wink, sad, angry, crying, blush, sweat drop, text, watermark, "
    "lowres, bad anatomy, bad proportions, blurry, cropped head"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/animagine-xl-4.0")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=26)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = DiffusionPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    records = []
    for idx, (cid, desc) in enumerate(CHARACTERS):
        gen = torch.Generator(device="cuda").manual_seed(args.seed + idx * 17)
        prompt = f"{desc}, {COMMON_POS}"
        img = pipe(
            prompt=prompt,
            negative_prompt=COMMON_NEG,
            width=1024,
            height=1024,
            guidance_scale=5.5,
            num_inference_steps=args.steps,
            generator=gen,
        ).images[0]
        p = args.out_dir / f"{cid}.png"
        img.save(p)
        records.append({"char_id": cid, "prompt": prompt, "image_path": str(p)})
        print(f"Saved {p}", flush=True)

    (args.out_dir / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"Done: {len(records)} portraits -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
