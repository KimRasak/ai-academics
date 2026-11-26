#!/usr/bin/env python3
"""
Exp 26B (Track `P7` — 100% DECONFOUNDED Cross-Pose Anime Re-ID Benchmark:
         XPose(Char i) vs. XPose(Char j, Same Gender & Same Hair Color)!):

Eliminates 100% of background, framing, and outfit confounds discovered during visual QC of Exp 26A:
  - Reference:       Front studio portrait of Character i
  - True Positive:   3/4-view outdoor dark-winter-coat portrait of Character i (`xpose_i`)
  - Hard Negative:   3/4-view outdoor dark-winter-coat portrait of DIFFERENT Character j (`xpose_j`)
                     who shares the EXACT SAME gender + hair color (and exact same pose/coat/background!)

Tests BOTH Vision Encoders (`CLIP-ViT-L-14`, `DINOv2-Large`, `SigLIP-SO400M`) AND Frontier VLMs
(`Qwen2.5-VL-7B`, `Qwen3-VL-32B`, `InternVL3-8B`) across 4 regimes:
  1. `XPose_SameColor`  (deconfounded clean cross-pose Re-ID!)
  2. `XPose_BWManga`    (True Pos is clean Luminance Grayscale/Manga; Hard Neg is full color!)
  3. `XPose_NightLight` (True Pos under strong Cyan Moonlight grading; Hard Neg is neutral daylight!)
  4. `XPose_HueShift`   (True Pos has hair hue rotated +135 deg; Hard Neg matches Ref's original hair color!)

And tests TWO clean zero-training remedies (fixing Exp 26A's washed-out color-dodge artifact!):
  - Remedy 1: **Clean Luminance-Canonicalized Matching** (standard ITU-R BT.601 grayscale + mild contrast
    normalization on BOTH Reference & Candidates — preserves full hair/eye tonal detail, eliminates RGB shortcut!)
  - Remedy 2: **Upper-Head Crop + Luminance-Canonicalized Matching** (crops tightly around head/hair/face
    to remove coat/street distractions!)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp25a_p7_hair_color_shortcut_pilot as e25a


def _clean_luma_gray(img: Image.Image) -> Image.Image:
    """Clean autocontrasted luminance grayscale (0% RGB hue, 100% tonal & structural fidelity — NO color-dodge washout!)."""
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)
    return g.convert("RGB")


def _head_crop(img: Image.Image) -> Image.Image:
    """Tight upper-head/hair/face crop removing torso/coat/lower background."""
    w, h = img.size
    return img.crop((int(w * 0.14), int(h * 0.02), int(w * 0.86), int(h * 0.68))).resize((448, 448), Image.LANCZOS)


def _build_deconfounded_pairs(manifest48: Path, xdir: Path) -> list[tuple[dict, dict, Path, Path]]:
    recs = [json.loads(l) for l in manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    orig_by_id = {r["char_id"]: r for r in recs if r["variant"] == "orig"}
    avail_ids = sorted(p.stem.replace("_xpose", "") for p in xdir.glob("*_xpose.png"))

    by_gh: dict[tuple[str, str], list[str]] = {}
    for cid in avail_ids:
        r = orig_by_id[cid]
        by_gh.setdefault((r["gender"], r["hair_color"]), []).append(cid)

    pairs = []
    for (_g, _hc), group in sorted(by_gh.items()):
        if len(group) < 2:
            continue
        for idx, cid in enumerate(group):
            neg_cid = group[(idx + 1) % len(group)]
            pairs.append((
                orig_by_id[cid],
                orig_by_id[neg_cid],
                xdir / f"{cid}_xpose.png",
                xdir / f"{neg_cid}_xpose.png",
            ))
    # Also pair remaining available chars with closest same-gender char if group size == 1
    paired_cids = {p[0]["char_id"] for p in pairs}
    by_g: dict[str, list[str]] = {}
    for cid in avail_ids:
        by_g.setdefault(orig_by_id[cid]["gender"], []).append(cid)
    # Stick strictly to EXACT same-gender + same-hair-color pairs if >= 12, else print exact count!
    print(f"Built {len(pairs)} STRICT 100%-deconfounded (same-gender, same-hair-color, BOTH-in-3/4-winter-coat) pairs!", flush=True)
    return pairs


@torch.no_grad()
def stage_eval_embed(args) -> int:
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    xdir = args.xpose_dir / "crosspose_imgs"
    pairs = _build_deconfounded_pairs(args.manifest48, xdir)

    encoders = [
        ("CLIP-ViT-L-14", "openai/clip-vit-large-patch14", "clip"),
        ("DINOv2-Large", "facebook/dinov2-large", "dino"),
        ("SigLIP-SO400M", "google/siglip-so400m-patch14-384", "siglip"),
    ]
    transforms = [
        ("XPose_SameColor", lambda im: im),
        ("XPose_BWGray", _clean_luma_gray),
        ("XPose_NightLight", e25a._to_cyan_night_lighting),
        ("XPose_HueShift135", lambda im: e25a._to_hue_shifted(im, 95)),
    ]

    all_res = {}
    for nice, repo, kind in encoders:
        if kind == "clip":
            proc = CLIPProcessor.from_pretrained(repo)
            mod = CLIPModel.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
            def emb_fn(im: Image.Image) -> np.ndarray:
                v = mod.get_image_features(**proc(images=im, return_tensors="pt").to("cuda"))
                v = (v if isinstance(v, torch.Tensor) else v.pooler_output)[0].float()
                return (v / (v.norm() + 1e-8)).cpu().numpy()
        elif kind == "siglip":
            proc = AutoImageProcessor.from_pretrained(repo)
            mod = AutoModel.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
            def emb_fn(im: Image.Image) -> np.ndarray:
                v = mod.get_image_features(**proc(images=im, return_tensors="pt").to("cuda"))
                v = (v if isinstance(v, torch.Tensor) else v.pooler_output)[0].float()
                return (v / (v.norm() + 1e-8)).cpu().numpy()
        else:
            proc = AutoImageProcessor.from_pretrained(repo)
            mod = AutoModel.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
            def emb_fn(im: Image.Image) -> np.ndarray:
                v = mod(**proc(images=im, return_tensors="pt").to("cuda")).last_hidden_state[0, 0].float()
                return (v / (v.norm() + 1e-8)).cpu().numpy()

        enc_res = {}
        print(f"\n=== EXP 26B DECONFOUNDED VISION ENCODER: {nice} (n={len(pairs)} strict XPose-vs-XPose pairs) ===", flush=True)
        for tname, tfn in transforms:
            r_ok, g_ok, hg_ok = [], [], []
            for r_pos, _r_neg, xp_pos, xp_neg in pairs:
                im_ref = Image.open(r_pos["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
                im_pos = tfn(Image.open(xp_pos).convert("RGB").resize((448, 448), Image.LANCZOS))
                im_neg = Image.open(xp_neg).convert("RGB").resize((448, 448), Image.LANCZOS)

                # 1. Raw RGB
                er, ep, en = emb_fn(im_ref), emb_fn(im_pos), emb_fn(im_neg)
                r_ok.append(int(np.dot(er, ep) > np.dot(er, en)))
                # 2. Clean Luminance-Canonicalized Grayscale
                er_g, ep_g, en_g = emb_fn(_clean_luma_gray(im_ref)), emb_fn(_clean_luma_gray(im_pos)), emb_fn(_clean_luma_gray(im_neg))
                g_ok.append(int(np.dot(er_g, ep_g) > np.dot(er_g, en_g)))
                # 3. Head-Cropped + Luminance-Canonicalized Grayscale
                er_h = emb_fn(_clean_luma_gray(_head_crop(im_ref)))
                ep_h = emb_fn(_clean_luma_gray(_head_crop(im_pos)))
                en_h = emb_fn(_clean_luma_gray(_head_crop(im_neg)))
                hg_ok.append(int(np.dot(er_h, ep_h) > np.dot(er_h, en_h)))

            enc_res[tname] = {
                "raw_rgb_acc": round(float(np.mean(r_ok)), 4),
                "luma_gray_acc": round(float(np.mean(g_ok)), 4),
                "head_luma_gray_acc": round(float(np.mean(hg_ok)), 4),
            }
            s = enc_res[tname]
            print(f"  {tname:20s} | Raw RGB={s['raw_rgb_acc']:.3f} -> Luma-Gray={s['luma_gray_acc']:.3f} -> Head+Luma={s['head_luma_gray_acc']:.3f}", flush=True)
        all_res[nice] = enc_res
        del mod
        torch.cuda.empty_cache()

    (args.out_dir / "exp26b_deconfounded_encoders.json").write_text(json.dumps(all_res, indent=2), encoding="utf-8")
    return 0


@torch.no_grad()
def stage_eval_vlm(args) -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    xdir = args.xpose_dir / "crosspose_imgs"
    pairs = _build_deconfounded_pairs(args.manifest48, xdir)

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ab_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in ["A", "B"]]

    def ask_2afc(ref: Image.Image, cA: Image.Image, cB: Image.Image) -> str:
        m = [{"role": "user", "content": [
            {"type": "image", "image": ref},
            {"type": "image", "image": cA},
            {"type": "image", "image": cB},
            {"type": "text", "text": e25a.PROMPT_REID},
        ]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[ref, cA, cB], return_tensors="pt").to(model.device)
        lg = model(**inp).logits[0, -1, ab_ids].float().cpu().numpy()
        return "A" if lg[0] >= lg[1] else "B"

    transforms = [
        ("XPose_SameColor", lambda im: im),
        ("XPose_BWGray", _clean_luma_gray),
        ("XPose_NightLight", e25a._to_cyan_night_lighting),
        ("XPose_HueShift135", lambda im: e25a._to_hue_shifted(im, 95)),
    ]

    summary = {}
    print(f"\n=== EXP 26B DECONFOUNDED VLM: {Path(args.model_dir).name} (n={len(pairs)} pairs x 2 orders) ===", flush=True)
    for tname, tfn in transforms:
        r_a, r_b, g_a, g_b, h_a, h_b = [], [], [], [], [], []
        for r_pos, _r_neg, xp_pos, xp_neg in pairs:
            im_ref = Image.open(r_pos["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
            im_pos = tfn(Image.open(xp_pos).convert("RGB").resize((448, 448), Image.LANCZOS))
            im_neg = Image.open(xp_neg).convert("RGB").resize((448, 448), Image.LANCZOS)

            for (ri, pi, ni, acc_l, both_l) in [
                (im_ref, im_pos, im_neg, r_a, r_b),
                (_clean_luma_gray(im_ref), _clean_luma_gray(im_pos), _clean_luma_gray(im_neg), g_a, g_b),
                (_clean_luma_gray(_head_crop(im_ref)), _clean_luma_gray(_head_crop(im_pos)), _clean_luma_gray(_head_crop(im_neg)), h_a, h_b),
            ]:
                o1, o2 = int(ask_2afc(ri, pi, ni) == "A"), int(ask_2afc(ri, ni, pi) == "B")
                acc_l.extend([o1, o2]); both_l.append(int(o1 and o2))

        summary[tname] = {
            "raw_bal": round(float(np.mean(r_a)), 4), "raw_both": round(float(np.mean(r_b)), 4),
            "luma_bal": round(float(np.mean(g_a)), 4), "luma_both": round(float(np.mean(g_b)), 4),
            "head_luma_bal": round(float(np.mean(h_a)), 4), "head_luma_both": round(float(np.mean(h_b)), 4),
        }
        s = summary[tname]
        print(f"  {tname:20s} | Raw={s['raw_bal']:.3f}(b={s['raw_both']:.3f}) -> Luma={s['luma_bal']:.3f}(b={s['luma_both']:.3f}) -> Head+Luma={s['head_luma_bal']:.3f}(b={s['head_luma_both']:.3f})", flush=True)

    (args.out_dir / f"exp26b_deconfounded_vlm_{Path(args.model_dir).name}.json").write_text(json.dumps({"model": Path(args.model_dir).name, "n_pairs": len(pairs), "summary": summary}, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["eval_embed", "eval_vlm"], required=True)
    ap.add_argument("--manifest48", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--xpose-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp26a_p7_crosspose"))
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "eval_embed":
        return stage_eval_embed(args)
    return stage_eval_vlm(args)


if __name__ == "__main__":
    raise SystemExit(main())
