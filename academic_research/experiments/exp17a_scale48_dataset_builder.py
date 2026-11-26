#!/usr/bin/env python3
"""
Exp 17A (P3 Scale-Up to n=48 Characters x 9 Variants = 432 Pixel-Exact Counterfactuals
         + Ground-Truth General Non-Affect VQA Metadata for Zero-Forgetting Verification):

Generates 32 brand-new diverse full-color broadcast TV-anime portraits (fc_16..fc_47)
using Animagine-XL-4.0 (sharded across GPUs for ~25s generation), detects exact face/eye/mouth
boxes with Qwen2.5-VL-7B, and builds the complete 48-character x 9-variant (432-image)
pixel-exact counterfactual suite (mouth_l1=0.0) by merging fc_00..fc_15 + fc_16..fc_47.

Also records exact ground-truth non-affect VQA attributes (gender, hair_color) for every
character so Exp 17C can verify 0.00% general-VQA degradation under our 1-Pass Circuit Surgery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure sibling experiment scripts are importable (imported lazily in stage_build for conda diffusers compat)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 32 brand-new diverse TV-anime characters (fc_16..fc_47), paired with ground-truth VQA metadata
NEW_CHARACTERS = [
    ("fc_16_scarlet_boy", "1boy, solo, spiky scarlet red hair, emerald green eyes, bright cheerful smile, white athletic jacket", "boy", "red"),
    ("fc_17_indigo_girl", "1girl, solo, long braided indigo hair, amber yellow eyes, happy gentle smile, cream cardigan", "girl", "blue"),
    ("fc_18_gold_boy", "1boy, solo, wavy golden blonde hair, sky blue eyes, warm friendly grin, navy blazer", "boy", "blonde"),
    ("fc_19_jade_girl", "1girl, solo, shoulder-length jade green hair, bright violet eyes, joyful smile, white sailor collar", "girl", "green"),
    ("fc_20_coral_girl", "1girl, solo, wavy coral pink hair, turquoise blue eyes, sweet happy smile, yellow sundress", "girl", "pink"),
    ("fc_21_raven_boy", "1boy, solo, neat jet black hair, bright crimson red eyes, confident cheerful smile, grey school blazer", "boy", "black"),
    ("fc_22_honey_girl", "1girl, solo, short honey brown bob hair, sapphire blue eyes, energetic bright smile, red hoodie", "girl", "brown"),
    ("fc_23_ice_boy", "1boy, solo, layered ice blue hair, bright orange eyes, happy relaxed smile, dark green jacket", "boy", "blue"),
    ("fc_24_magenta_girl", "1girl, solo, high magenta ponytail hair, golden eyes, beaming joyful smile, white collared shirt", "girl", "pink"),
    ("fc_25_ash_girl", "1girl, solo, soft ash grey hair, bright teal eyes, warm gentle smile, maroon sweater", "girl", "white"),
    ("fc_26_copper_boy", "1boy, solo, messy copper orange hair, bright green eyes, enthusiastic smile, blue denim jacket", "boy", "orange"),
    ("fc_27_lilac_girl", "1girl, solo, twin braids lilac purple hair, rose pink eyes, happy sweet smile, pastel blue blouse", "girl", "purple"),
    ("fc_28_forest_boy", "1boy, solo, short forest green hair, warm amber eyes, friendly cheerful smile, tan cardigan", "boy", "green"),
    ("fc_29_ivory_girl", "1girl, solo, long straight ivory white hair, cobalt blue eyes, serene happy smile, black ribbon blouse", "girl", "white"),
    ("fc_30_mahogany_girl", "1girl, solo, curly mahogany brown hair, emerald green eyes, joyful smile, mustard sweater", "girl", "brown"),
    ("fc_31_cyan_boy", "1boy, solo, tousled bright cyan hair, golden yellow eyes, cheerful grin, white hoodie", "boy", "blue"),
    ("fc_32_rose_girl", "1girl, solo, short rose red bob hair, bright sky blue eyes, playful happy smile, navy sailor uniform", "girl", "red"),
    ("fc_33_obsidian_girl", "1girl, solo, sleek black ponytail hair, bright violet eyes, cheerful confident smile, pink cardigan", "girl", "black"),
    ("fc_34_wheat_boy", "1boy, solo, messy wheat blonde hair, emerald green eyes, bright warm smile, burgundy hoodie", "boy", "blonde"),
    ("fc_35_amethyst_girl", "1girl, solo, wavy amethyst purple hair, cyan eyes, sweet bright smile, cream vest", "girl", "purple"),
    ("fc_36_tangerine_girl", "1girl, solo, energetic tangerine orange twin tails hair, blue eyes, beaming smile, white polo shirt", "girl", "orange"),
    ("fc_37_slate_boy", "1boy, solo, medium slate blue hair, bright ruby eyes, friendly happy smile, charcoal sweater", "boy", "blue"),
    ("fc_38_cherry_girl", "1girl, solo, long cherry red hair, golden amber eyes, warm joyful smile, dark blue cardigan", "girl", "red"),
    ("fc_39_chestnut_girl", "1girl, solo, side ponytail chestnut brown hair, bright orange eyes, cheerful smile, green hoodie", "girl", "brown"),
    ("fc_40_pearl_boy", "1boy, solo, sleek pearl white hair, sapphire blue eyes, gentle happy smile, black turtleneck", "boy", "white"),
    ("fc_41_emerald_girl", "1girl, solo, long wavy emerald green hair, pink eyes, radiant smile, yellow cardigan", "girl", "green"),
    ("fc_42_peach_boy", "1boy, solo, short peach pink hair, teal eyes, upbeat cheerful smile, navy varsity jacket", "boy", "pink"),
    ("fc_43_midnight_girl", "1girl, solo, short midnight blue hair, bright yellow eyes, happy smile, white hoodie", "girl", "blue"),
    ("fc_44_saffron_girl", "1girl, solo, braided saffron blonde hair, violet eyes, gentle happy smile, teal blouse", "girl", "blonde"),
    ("fc_45_ebony_boy", "1boy, solo, spiky ebony black hair, bright turquoise eyes, confident grin, red athletic jacket", "boy", "black"),
    ("fc_46_orchid_girl", "1girl, solo, wavy orchid purple hair, emerald green eyes, sweet cheerful smile, beige cardigan", "girl", "purple"),
    ("fc_47_cinnamon_boy", "1boy, solo, neat cinnamon brown hair, bright blue eyes, warm friendly smile, olive jacket", "boy", "brown"),
]

# Ground-truth metadata for original 16 characters (fc_00..fc_15)
ORIG16_META = {
    "fc_00_navy_girl": ("girl", "blue"),
    "fc_01_pink_girl": ("girl", "pink"),
    "fc_02_silver_boy": ("boy", "white"),
    "fc_03_blonde_girl": ("girl", "blonde"),
    "fc_04_black_girl": ("girl", "black"),
    "fc_05_orange_girl": ("girl", "orange"),
    "fc_06_violet_girl": ("girl", "purple"),
    "fc_07_brown_boy": ("boy", "brown"),
    "fc_08_mint_girl": ("girl", "green"),
    "fc_09_crimson_girl": ("girl", "red"),
    "fc_10_white_girl": ("girl", "white"),
    "fc_11_teal_boy": ("boy", "blue"),
    "fc_12_auburn_girl": ("girl", "brown"),
    "fc_13_cobalt_boy": ("boy", "blue"),
    "fc_14_peach_girl": ("girl", "blonde"),
    "fc_15_plum_girl": ("girl", "purple"),
}

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


def stage_gen(args) -> int:
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    portraits_dir = args.out_dir / "portraits_new32"
    portraits_dir.mkdir(parents=True, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(
        args.sdxl_dir,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    items = [c for i, c in enumerate(NEW_CHARACTERS) if i % args.num_shards == args.shard]
    for cid, desc, _g, _hc in items:
        idx = int(cid.split("_")[1])
        p = portraits_dir / f"{cid}.png"
        if p.exists():
            print(f"[shard {args.shard}] skip existing {cid}", flush=True)
            continue
        gen = torch.Generator(device="cuda").manual_seed(5100 + idx * 37)
        img = pipe(
            prompt=f"{desc}, {COMMON_POS}",
            negative_prompt=COMMON_NEG,
            width=1024,
            height=1024,
            guidance_scale=5.0,
            num_inference_steps=28,
            generator=gen,
        ).images[0]
        img.save(p)
        print(f"[shard {args.shard}] saved {p.name}", flush=True)
    return 0


def stage_build(args) -> int:
    import exp2b_p3_counterfactual_suite as exp2b
    import exp8_p3_bio_vs_nonbio_matched_control as exp8

    portraits_new = args.out_dir / "portraits_new32"
    new_paths = sorted(portraits_new.glob("fc_*.png"))
    assert len(new_paths) == 32, f"Expected 32 new portraits, found {len(new_paths)}"

    # 1. Detect face/eye/mouth boxes for the 32 new characters using Qwen2.5-VL-7B (exact same detector as exp2b)
    boxes_file = args.out_dir / "face_boxes_new32.json"
    if boxes_file.exists():
        boxes_new = json.loads(boxes_file.read_text(encoding="utf-8"))
    else:
        boxes_new = exp2b._detect_face_boxes_vlm(new_paths, args.qwen_dir)
        boxes_file.write_text(json.dumps(boxes_new, indent=2), encoding="utf-8")

    # 2. Load existing 16-char 9-variant records
    orig_records = [
        json.loads(l)
        for l in args.orig_manifest9.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    # Add VQA metadata to orig 16
    all_records = []
    for r in orig_records:
        g, hc = ORIG16_META[r["char_id"]]
        r2 = dict(r)
        r2["gender"] = g
        r2["hair_color"] = hc
        all_records.append(r2)

    meta_new = {cid: (g, hc) for cid, _d, g, hc in NEW_CHARACTERS}

    # 3. Generate all 9 pixel-exact counterfactuals for each of the 32 new characters
    vars_dir = args.out_dir / "variants_new32"
    sheets_dir = args.out_dir / "qc_sheets"
    vars_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    max_mouth_l1 = 0.0
    for idx, p in enumerate(new_paths):
        cid = p.stem
        boxes = boxes_new[cid]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        mx1, my1, mx2, my2 = exp2b._to_px(boxes["mouth"], w, h)
        orig_arr = np.asarray(img, dtype=np.float32)
        orig_mouth = orig_arr[my1:my2, mx1:mx2]

        img_placebo, _ = exp2b._add_placebo_hairpin(img, boxes)
        img_eye, _ = exp2b._remove_eye_highlights(img, boxes)
        img_shd, _ = exp2b._add_forehead_shadow(img, boxes)
        img_ds, _ = exp2b._add_forehead_shadow(img_eye, boxes)
        img_bls, _ = exp2b._add_blush_lines(img, boxes)
        img_swt, _ = exp2b._add_sweat_drop(img, boxes)
        img_brw = exp8._add_bio_angry_brows(img, boxes)
        img_vein = exp8._add_anger_vein_mark(img, boxes)

        variants = [
            ("orig", img),
            ("placebo_hairpin", img_placebo),
            ("eye_highlight_off", img_eye),
            ("forehead_shadow_on", img_shd),
            ("dead_eyes_and_shadow", img_ds),
            ("blush_lines_on", img_bls),
            ("sweat_drop_on", img_swt),
            ("bio_angry_brows", img_brw),
            ("anger_vein_on", img_vein),
        ]

        g, hc = meta_new[cid]
        cdir = vars_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        for vname, vimg in variants:
            out_p = cdir / f"{vname}.png"
            vimg.save(out_p)
            varr = np.asarray(vimg, dtype=np.float32)
            mouth_l1 = float(np.mean(np.abs(varr[my1:my2, mx1:mx2] - orig_mouth)))
            max_mouth_l1 = max(max_mouth_l1, mouth_l1)
            px_pct = float(np.mean(np.any(np.abs(varr - orig_arr) > 3.0, axis=2)) * 100.0)
            all_records.append({
                "char_id": cid,
                "variant": vname,
                "image_path": str(out_p),
                "mouth_l1": round(mouth_l1, 5),
                "changed_px_pct": round(px_pct, 4),
                "gender": g,
                "hair_color": hc,
            })

        # Save QC sheet for first 4 new characters
        if idx < 4:
            tile = 320
            sheet = Image.new("RGB", (tile * 9, tile), (255, 255, 255))
            for vi, (_vn, vimg) in enumerate(variants):
                sheet.paste(vimg.resize((tile, tile), Image.LANCZOS), (vi * tile, 0))
            sheet.save(sheets_dir / f"{cid}_9var_sheet.png")

    manifest_path = args.out_dir / "manifest_scale48_9var.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r) for r in all_records) + "\n", encoding="utf-8")
    n_chars = len({r["char_id"] for r in all_records})
    print(f"DONE: {n_chars} chars x 9 variants = {len(all_records)} images | max_mouth_l1={max_mouth_l1:.5f} -> {manifest_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "build"], required=True)
    ap.add_argument("--sdxl-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/animagine-xl-4.0")
    ap.add_argument("--qwen-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--orig-manifest9", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp2_p3_v2/manifest_9var.jsonl"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    args = ap.parse_args()
    if args.stage == "gen":
        return stage_gen(args)
    return stage_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
