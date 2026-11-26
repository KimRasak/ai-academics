#!/usr/bin/env python3
"""
Exp 25A (NEW PROBLEM PILOT `P7` — The 2D Anime "Color-Palette Shortcut" in Character Re-Identification!):

Tests a fundamental unsolved question in 2D Anime Computer Vision (IP consistency, anime multi-panel
continuity, video Re-ID):
  Because 2D anime faces have radically simplified geometry ("same-face syndrome"), do SOTA Multimodal
  VLMs (`InternVL3-8B`, `Qwen2.5-VL-7B`, `LLaVA-OV-7B`, `Qwen3-VL-32B`) identify anime characters by
  structural identity (hair silhouette, bangs, eye shape, facial proportions), OR do they collapse onto
  a shallow RGB color-palette shortcut?!

Constructs 4 controlled 2-Alternative Forced Choice (2AFC) Identity Triplet Tests from our 48 diverse
TV-anime characters (fc_00..fc_47, balanced A/B position to eliminate position bias!):
  Given Reference Image (Character i, original color), pick which candidate is the SAME character:
    1. `T1_BWManga_vs_SameColorOther`:
         - True Same: Character i converted to clean B&W Manga Lineart/Screentone (0% color, 100% geometry!)
         - Hard Neg:  DIFFERENT Character j (same gender + SAME hair color as i, full color!)
    2. `T2_LightingTint_vs_SameColorOther`:
         - True Same: Character i under strong sunset/cyan-night anime lighting color cast (same geometry!)
         - Hard Neg:  DIFFERENT Character j (same gender + SAME hair color as i, neutral daylight!)
    3. `T3_DiffOutfitPose_vs_SameColorOther` (generated natively via Animagine-XL-4.0!):
         - True Same: Character i in a DIFFERENT pose & outfit (same hair silhouette + eye color + hair color)
         - Hard Neg:  DIFFERENT Character j (same gender + SAME hair color!)
    4. `T4_HueShift_vs_SameColorOther`:
         - True Same: Character i with hair/clothes hue shifted +120 deg (exact same pixel geometry/silhouette!)
         - Hard Neg:  DIFFERENT Character j (same gender + SAME original hair color!)

If SOTA VLMs fail T1 / T2 / T4 (choosing the WRONG character just because they share the RGB hair color!),
we have discovered a clean, quantifiable, high-impact CVPR/ICCV failure mode for 2D anime character Re-ID!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps


def _to_bw_manga(img: Image.Image) -> Image.Image:
    """Clean high-contrast B&W manga lineart + crisp ink conversion (0% RGB color, 100% structural geometry!)."""
    gray = ImageOps.grayscale(img)
    blur = gray.filter(ImageFilter.GaussianBlur(radius=3))
    g = np.asarray(gray, dtype=np.float32)
    b = np.asarray(blur, dtype=np.float32)
    # Color-dodge pencil/manga line extraction + gentle tone quantization
    line = np.clip(g * 255.0 / np.maximum(255.0 - b + 1.0, 1.0), 0, 255)
    # Blend crisp ink edges with 3-level manga screentone gray
    tone = np.where(g < 85, 65.0, np.where(g < 170, 175.0, 245.0))
    out = np.clip(0.55 * line + 0.45 * tone, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L").convert("RGB")


def _to_cyan_night_lighting(img: Image.Image) -> Image.Image:
    """Strong anime moonlight / cyan-indigo environmental color cast (shifts RGB palette strongly, keeps 100% geometry!)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Cool cyan-blue night tint matrix (classic anime night scene grading)
    r = np.clip(arr[:, :, 0] * 0.42 + 0.04, 0, 1)
    g = np.clip(arr[:, :, 1] * 0.72 + 0.06, 0, 1)
    b = np.clip(arr[:, :, 2] * 1.18 + 0.12, 0, 1)
    out = np.stack([r, g, b], axis=2)
    return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")


def _to_hue_shifted(img: Image.Image, shift: int = 95) -> Image.Image:
    """Rotate HSV hue by +135 deg (changes hair/clothing color while keeping 100% pixel-exact silhouette & facial geometry!)."""
    hsv = np.asarray(img.convert("HSV"), dtype=np.int32).copy()
    # Keep skin tones mostly natural? Even cleaner: rotate all high-saturation pixels (hair/eyes/clothes)!
    sat = hsv[:, :, 1]
    mask = sat > 55
    hsv[:, :, 0] = np.where(mask, (hsv[:, :, 0] + shift) % 256, hsv[:, :, 0])
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


PROMPT_REID = (
    "You are given THREE 2D anime character portraits:\n"
    "- Image 1 is the REFERENCE character.\n"
    "- Image 2 is Candidate A.\n"
    "- Image 3 is Candidate B.\n"
    "One candidate is the EXACT SAME character as the Reference (possibly drawn in black-and-white manga style, "
    "under different colored lighting, or with dyed hair/recolored palette), while the other is a DIFFERENT character "
    "who happens to have a similar hair color.\n"
    "Based on structural character identity (hairstyle silhouette, bangs, eye shape, facial structure), "
    "which candidate is the SAME character as Image 1?\n"
    "Answer with ONLY a single letter: A or B."
)


@torch.no_grad()
def main() -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest48", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    orig48 = [r for r in recs if r["variant"] == "orig"]

    # Build hard-negative pairs: match each char i with a DIFFERENT char j sharing the EXACT SAME (gender, hair_color)!
    by_gh: dict[tuple[str, str], list[dict]] = {}
    for r in orig48:
        by_gh.setdefault((r["gender"], r["hair_color"]), []).append(r)

    pairs = []
    for (g, hc), group in sorted(by_gh.items()):
        if len(group) < 2:
            continue
        for idx, r in enumerate(group):
            neg = group[(idx + 1) % len(group)]
            pairs.append((r, neg))
    print(f"Built {len(pairs)} exact (same-gender, same-hair-color) Hard-Negative anime Re-ID pairs!", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ab_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in ["A", "B"]]

    def ask_2afc(ref: Image.Image, candA: Image.Image, candB: Image.Image) -> tuple[str, float]:
        m = [{
            "role": "user",
            "content": [
                {"type": "image", "image": ref},
                {"type": "image", "image": candA},
                {"type": "image", "image": candB},
                {"type": "text", "text": PROMPT_REID},
            ],
        }]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[ref, candA, candB], return_tensors="pt").to(model.device)
        lg = model(**inp).logits[0, -1, ab_ids].float().cpu().numpy()
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        return ("A" if lg[0] >= lg[1] else "B"), float(pr[0])

    transforms = [
        ("T0_ExactIdentical_Sanity", lambda im: im),
        ("T1_BWManga_vs_SameHairColor", _to_bw_manga),
        ("T2_CyanNightLight_vs_SameHairColor", _to_cyan_night_lighting),
        ("T3_HueShift135_vs_SameHairColor", lambda im: _to_hue_shifted(im, 95)),
    ]

    summary = {}
    print(f"\n=== EXP 25A (P7 PILOT) ANIME COLOR-PALETTE SHORTCUT RE-ID: {Path(args.model_dir).name} (n={len(pairs)} pairs x 2 balanced orders) ===", flush=True)
    for tname, fn in transforms:
        accs, pair_both_ok = [], []
        for r_pos, r_neg in pairs:
            im_ref = Image.open(r_pos["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
            im_pos = fn(im_ref)
            im_neg = Image.open(r_neg["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
            # Run BOTH (A=pos, B=neg) and (A=neg, B=pos) for 100% position-debiased paired consistency!
            p1, _ = ask_2afc(im_ref, im_pos, im_neg)
            p2, _ = ask_2afc(im_ref, im_neg, im_pos)
            ok1, ok2 = int(p1 == "A"), int(p2 == "B")
            accs.extend([ok1, ok2])
            pair_both_ok.append(int(ok1 == 1 and ok2 == 1))
        summary[tname] = {
            "acc_balanced": round(float(np.mean(accs)), 4),
            "paired_consistent_acc": round(float(np.mean(pair_both_ok)), 4),
        }
        print(f"  {tname:36s} | Balanced Acc={summary[tname]['acc_balanced']:.3f} | Paired Both-Order={summary[tname]['paired_consistent_acc']:.3f}", flush=True)

    out_p = args.out_dir / f"exp25a_p7_reid_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "n_pairs": len(pairs), "summary": summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
