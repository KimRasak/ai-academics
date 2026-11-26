#!/usr/bin/env python3
"""
Exp 2A-v2 / 3A-v2: Generate 16 FULL-COLOR broadcast TV-anime cel-shaded portraits
using Animagine XL 4.0 (fixing the 'lineart' tag artifact from v1).

Produces rich, vibrant, full-color 2D anime portraits with clear skin tone,
saturated hair, vivid irises with crisp white catchlights, cheerful smiling mouth,
and clean 2-tone TV-anime cel shading — ideal for BOTH:
  - Exp 2 (P3: Non-Biological Affect Symbols — eye highlights OFF, forehead shadow, blush, sweat drop)
  - Exp 3 (P2: True 2D Cel-Shading Relighting vs IC-Light / Qwen-Image-Edit / 10-line baseline)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

CHARACTERS = [
    ("fc_00_navy_girl", "1girl, solo, short navy blue bob hair, bright cyan eyes, joyful smile, red ribbon, school uniform"),
    ("fc_01_pink_girl", "1girl, solo, pastel pink twin tails hair, emerald green eyes, happy smile, white collar blouse"),
    ("fc_02_silver_boy", "1boy, solo, messy silver hair, vivid amber eyes, warm cheerful smile, dark blue jacket"),
    ("fc_03_blonde_girl", "1girl, solo, long golden blonde hair, sapphire blue eyes, bright happy smile, green cardigan"),
    ("fc_04_black_girl", "1girl, solo, long straight black hair, ruby red eyes, gentle smile, cream sweater"),
    ("fc_05_orange_girl", "1girl, solo, short bright orange hair, teal eyes, energetic happy smile, yellow hoodie"),
    ("fc_06_violet_girl", "1girl, solo, wavy lavender hair, violet eyes, sweet smile, lilac blouse"),
    ("fc_07_brown_boy", "1boy, solo, neat chestnut brown hair, warm golden eyes, friendly smile, white collared shirt"),
    ("fc_08_mint_girl", "1girl, solo, mint green hair, golden yellow eyes, cheerful smile, sailor uniform"),
    ("fc_09_crimson_girl", "1girl, solo, crimson red ponytail hair, bright blue eyes, confident happy smile, dark turtleneck"),
    ("fc_10_white_girl", "1girl, solo, snowy white short hair, rose pink eyes, soft happy smile, red scarf"),
    ("fc_11_teal_boy", "1boy, solo, dark teal hair, turquoise eyes, beaming cheerful smile, varsity jacket"),
    ("fc_12_auburn_girl", "1girl, solo, warm auburn hair, bright green eyes, happy smile, denim jacket"),
    ("fc_13_cobalt_boy", "1boy, solo, cobalt blue hair, bright orange eyes, cheerful grin, black hoodie"),
    ("fc_14_peach_girl", "1girl, solo, peach blonde bob hair, sky blue eyes, gentle joyful smile, striped shirt"),
    ("fc_15_plum_girl", "1girl, solo, deep plum hair, bright gold eyes, warm happy smile, beige blazer"),
]

COMMON_POS = (
    "anime coloring, tv anime screenshot, vibrant full color, upper body portrait, "
    "front view, looking at viewer, open eyes, bright detailed eyes, white eye highlights, "
    "clean cel shading, even studio lighting, simple background, light cream background, "
    "masterpiece, high score, great score, absurdres"
)
COMMON_NEG = (
    "sketch, lineart, monochrome, greyscale, unfinished, pale, washed out, muted colors, "
    "3d, photorealistic, realistic, heavy shadow, dark background, side view, closed eyes, "
    "sad, angry, crying, blush, sweat drop, shaded face, text, watermark, lowres, bad anatomy"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/animagine-xl-4.0")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=4201)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = DiffusionPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    records = []
    for idx, (cid, desc) in enumerate(CHARACTERS):
        gen = torch.Generator(device="cuda").manual_seed(args.seed + idx * 31)
        prompt = f"{desc}, {COMMON_POS}"
        img = pipe(
            prompt=prompt,
            negative_prompt=COMMON_NEG,
            width=1024,
            height=1024,
            guidance_scale=5.0,
            num_inference_steps=args.steps,
            generator=gen,
        ).images[0]
        p = args.out_dir / f"{cid}.png"
        img.save(p)
        records.append({"char_id": cid, "prompt": prompt, "image_path": str(p)})
        print(f"Saved {p}", flush=True)

    (args.out_dir / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"Done: {len(records)} full-color TV-anime portraits -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
