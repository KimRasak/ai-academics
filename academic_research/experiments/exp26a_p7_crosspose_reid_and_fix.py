#!/usr/bin/env python3
"""
Exp 26A (Track `P7` — Real Cross-Pose / Cross-Outfit 2D Anime Character Re-ID:
         Exposing the "Color-Palette Shortcut" in CLIP-I, DINOv2, SigLIP & Frontier VLMs
         + Zero-Training Structure-Canonicalized Fix!):

Every CVPR/ICCV/SIGGRAPH anime generation & video consistency paper measures "Character Identity"
via CLIP-I, DINOv2, or SigLIP cosine similarity (or VLM Judge).
We test whether these metrics actually measure STRUCTURAL character identity (hair silhouette, bangs,
eye shape, facial geometry) OR collapse onto a shallow RGB Color-Palette Shortcut!

  Stage 1 (`--stage gen`):
    Uses Qwen-Image-Edit-2511 across 4 H100s to generate genuine CROSS-POSE + DIFFERENT-OUTFIT
    3/4-view portraits for 24 diverse characters (each paired with a DIFFERENT character sharing
    the EXACT SAME gender + hair_color as Hard Negative!).
  Stage 2 (`--stage eval_embed`):
    Evaluates `CLIP-ViT-L`, `DINOv2-Large`, and `SigLIP-SO400M` cosine similarity 2AFC accuracy
    across 4 real Cross-Pose regimes:
      - `XPose_SameColor`   (Same Char 3/4 new outfit, same color vs HardNeg same hair color)
      - `XPose_BWManga`     (Same Char 3/4 new outfit, B&W Manga vs HardNeg full color!)
      - `XPose_NightLight`  (Same Char 3/4 new outfit, Cyan Night Lighting vs HardNeg daylight!)
      - `XPose_HueShift`    (Same Char 3/4 new outfit, Recolored Hair vs HardNeg original color!)
    AND compares Raw RGB vs. OURS Zero-Training Structure-Canonicalized Matching (matching BOTH
    Reference and Candidates in normalized B&W Manga / Grayscale structural space!).
  Stage 3 (`--stage eval_vlm`):
    Evaluates `Qwen2.5-VL-7B` and `Qwen3-VL-32B` (balanced both orders A/B) on Raw RGB vs.
    OURS Structure-Canonicalized Matching across all 4 Cross-Pose regimes!
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp25a_p7_hair_color_shortcut_pilot as e25a


def _build_pairs(manifest48: Path, max_pairs: int = 24) -> list[tuple[dict, dict]]:
    recs = [json.loads(l) for l in manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    orig48 = [r for r in recs if r["variant"] == "orig"]
    by_gh: dict[tuple[str, str], list[dict]] = {}
    for r in orig48:
        by_gh.setdefault((r["gender"], r["hair_color"]), []).append(r)
    pairs = []
    for (_g, _hc), group in sorted(by_gh.items()):
        if len(group) < 2:
            continue
        for idx, r in enumerate(group):
            pairs.append((r, group[(idx + 1) % len(group)]))
    # Deterministic diverse subset of 24 pairs
    pairs.sort(key=lambda x: x[0]["char_id"])
    return pairs[:max_pairs]


def stage_gen(args) -> int:
    from diffusers import QwenImageEditPlusPipeline

    pairs = _build_pairs(args.manifest48, max_pairs=args.max_pairs)
    xdir = args.out_dir / "crosspose_imgs"
    xdir.mkdir(parents=True, exist_ok=True)

    pipe = QwenImageEditPlusPipeline.from_pretrained(args.edit_dir, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    prompt = (
        "Generate a 2D Japanese TV anime screenshot of this EXACT SAME anime character turned to a three-quarter view, "
        "wearing a completely different outfit (a plain dark grey winter jacket with scarf) outdoors with a soft blurred city street background. "
        "Strictly preserve the character's exact hairstyle silhouette, bangs, eye shape, iris color, hair color, and facial identity in clean 2D anime cel shading."
    )

    my_items = [(i, p) for i, p in enumerate(pairs) if i % args.num_shards == args.shard]
    for i, (r_pos, _r_neg) in my_items:
        cid = r_pos["char_id"]
        out_p = xdir / f"{cid}_xpose.png"
        if out_p.exists():
            continue
        im = Image.open(r_pos["image_path"]).convert("RGB").resize((768, 768), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(8100 + i * 29)
        res = pipe(
            image=im,
            prompt=prompt,
            num_inference_steps=20,
            true_cfg_scale=4.0,
            generator=gen,
        ).images[0]
        res.save(out_p)
        print(f"[xpose shard {args.shard}] saved {out_p.name}", flush=True)

    # Shard 0 also builds a 4-row QC contact sheet after all shards (or best-effort)
    return 0


def stage_sheet(args) -> int:
    pairs = _build_pairs(args.manifest48, max_pairs=args.max_pairs)
    xdir = args.out_dir / "crosspose_imgs"
    tile = 280
    # 6 columns: Ref | XPose_SameColor | XPose_BWManga | XPose_Night | XPose_HueShift | HardNeg_SameHairColor
    sheet = Image.new("RGB", (tile * 6, tile * 4), (255, 255, 255))
    for row_i, (r_pos, r_neg) in enumerate(pairs[:4]):
        im_ref = Image.open(r_pos["image_path"]).convert("RGB").resize((tile, tile), Image.LANCZOS)
        im_xp = Image.open(xdir / f"{r_pos['char_id']}_xpose.png").convert("RGB").resize((tile, tile), Image.LANCZOS)
        im_neg = Image.open(r_neg["image_path"]).convert("RGB").resize((tile, tile), Image.LANCZOS)
        cols = [
            im_ref,
            im_xp,
            e25a._to_bw_manga(im_xp),
            e25a._to_cyan_night_lighting(im_xp),
            e25a._to_hue_shifted(im_xp, 95),
            im_neg,
        ]
        for ci, im in enumerate(cols):
            sheet.paste(im, (ci * tile, row_i * tile))
    out_s = args.out_dir / "p7_crosspose_qc_sheet.png"
    sheet.save(out_s)
    print(f"Saved QC sheet -> {out_s}", flush=True)
    return 0


@torch.no_grad()
def stage_eval_embed(args) -> int:
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    pairs = _build_pairs(args.manifest48, max_pairs=args.max_pairs)
    xdir = args.out_dir / "crosspose_imgs"

    encoders = [
        ("CLIP-ViT-L-14", "openai/clip-vit-large-patch14", "clip"),
        ("DINOv2-Large", "facebook/dinov2-large", "dino"),
        ("SigLIP-SO400M", "google/siglip-so400m-patch14-384", "siglip"),
    ]
    transforms = [
        ("XPose_SameColor", lambda im: im),
        ("XPose_BWManga", e25a._to_bw_manga),
        ("XPose_NightLight", e25a._to_cyan_night_lighting),
        ("XPose_HueShift135", lambda im: e25a._to_hue_shifted(im, 95)),
    ]

    all_res = {}
    for nice, repo, kind in encoders:
        print(f"\nLoading vision encoder {nice} ({repo})...", flush=True)
        if kind == "clip":
            proc = CLIPProcessor.from_pretrained(repo)
            mod = CLIPModel.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
            def emb_fn(im: Image.Image) -> np.ndarray:
                inp = proc(images=im, return_tensors="pt").to("cuda")
                v = mod.get_image_features(**inp)
                if not isinstance(v, torch.Tensor):
                    v = v.pooler_output
                v = v[0].float()
                return (v / (v.norm() + 1e-8)).cpu().numpy()
        elif kind == "siglip":
            proc = AutoImageProcessor.from_pretrained(repo)
            mod = AutoModel.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
            def emb_fn(im: Image.Image) -> np.ndarray:
                inp = proc(images=im, return_tensors="pt").to("cuda")
                v = mod.get_image_features(**inp)
                if not isinstance(v, torch.Tensor):
                    v = v.pooler_output
                v = v[0].float()
                return (v / (v.norm() + 1e-8)).cpu().numpy()
        else:
            proc = AutoImageProcessor.from_pretrained(repo)
            mod = AutoModel.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
            def emb_fn(im: Image.Image) -> np.ndarray:
                inp = proc(images=im, return_tensors="pt").to("cuda")
                v = mod(**inp).last_hidden_state[0, 0].float()
                return (v / (v.norm() + 1e-8)).cpu().numpy()

        enc_res = {}
        print(f"=== EXP 26A VISION ENCODER CROSS-POSE RE-ID: {nice} (n={len(pairs)}) ===", flush=True)
        for tname, tfn in transforms:
            raw_ok, struct_ok, raw_margins, struct_margins = [], [], [], []
            for r_pos, r_neg in pairs:
                im_ref = Image.open(r_pos["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
                im_xp = Image.open(xdir / f"{r_pos['char_id']}_xpose.png").convert("RGB").resize((448, 448), Image.LANCZOS)
                im_pos = tfn(im_xp)
                im_neg = Image.open(r_neg["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)

                # 1. Standard Raw RGB Cosine Similarity
                er, ep, en = emb_fn(im_ref), emb_fn(im_pos), emb_fn(im_neg)
                m_raw = float(np.dot(er, ep) - np.dot(er, en))
                raw_ok.append(int(m_raw > 0))
                raw_margins.append(m_raw)

                # 2. OURS Zero-Training Structure-Canonicalized Matching (B&W Manga lineart-domain on BOTH sides!)
                er_s = emb_fn(e25a._to_bw_manga(im_ref))
                ep_s = emb_fn(e25a._to_bw_manga(im_pos))
                en_s = emb_fn(e25a._to_bw_manga(im_neg))
                m_st = float(np.dot(er_s, ep_s) - np.dot(er_s, en_s))
                struct_ok.append(int(m_st > 0))
                struct_margins.append(m_st)

            enc_res[tname] = {
                "raw_2afc_acc": round(float(np.mean(raw_ok)), 4),
                "raw_mean_margin": round(float(np.mean(raw_margins)), 4),
                "struct_canon_2afc_acc": round(float(np.mean(struct_ok)), 4),
                "struct_canon_margin": round(float(np.mean(struct_margins)), 4),
            }
            print(f"  {tname:20s} | Raw RGB={enc_res[tname]['raw_2afc_acc']:.3f} (m={enc_res[tname]['raw_mean_margin']:+.3f}) -> OURS Struct-Canon={enc_res[tname]['struct_canon_2afc_acc']:.3f} (m={enc_res[tname]['struct_canon_margin']:+.3f})", flush=True)

        all_res[nice] = enc_res
        del mod
        torch.cuda.empty_cache()

    out_p = args.out_dir / "exp26a_p7_vision_encoders.json"
    out_p.write_text(json.dumps(all_res, indent=2), encoding="utf-8")
    return 0


@torch.no_grad()
def stage_eval_vlm(args) -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    pairs = _build_pairs(args.manifest48, max_pairs=args.max_pairs)
    xdir = args.out_dir / "crosspose_imgs"

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
        ("XPose_BWManga", e25a._to_bw_manga),
        ("XPose_NightLight", e25a._to_cyan_night_lighting),
        ("XPose_HueShift135", lambda im: e25a._to_hue_shifted(im, 95)),
    ]

    summary = {}
    print(f"\n=== EXP 26A VLM CROSS-POSE RE-ID: {Path(args.model_dir).name} (n={len(pairs)} pairs x 2 orders) ===", flush=True)
    for tname, tfn in transforms:
        raw_acc, raw_both, st_acc, st_both = [], [], [], []
        for r_pos, r_neg in pairs:
            im_ref = Image.open(r_pos["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
            im_xp = Image.open(xdir / f"{r_pos['char_id']}_xpose.png").convert("RGB").resize((448, 448), Image.LANCZOS)
            im_pos = tfn(im_xp)
            im_neg = Image.open(r_neg["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)

            o1, o2 = int(ask_2afc(im_ref, im_pos, im_neg) == "A"), int(ask_2afc(im_ref, im_neg, im_pos) == "B")
            raw_acc.extend([o1, o2]); raw_both.append(int(o1 and o2))

            # OURS Zero-Training Structure-Canonicalized Matching (B&W Manga lineart on all 3 images!)
            sr, sp, sn = e25a._to_bw_manga(im_ref), e25a._to_bw_manga(im_pos), e25a._to_bw_manga(im_neg)
            s1, s2 = int(ask_2afc(sr, sp, sn) == "A"), int(ask_2afc(sr, sn, sp) == "B")
            st_acc.extend([s1, s2]); st_both.append(int(s1 and s2))

        summary[tname] = {
            "raw_balanced": round(float(np.mean(raw_acc)), 4),
            "raw_paired_both": round(float(np.mean(raw_both)), 4),
            "struct_balanced": round(float(np.mean(st_acc)), 4),
            "struct_paired_both": round(float(np.mean(st_both)), 4),
        }
        s = summary[tname]
        print(f"  {tname:20s} | Raw RGB={s['raw_balanced']:.3f}(both={s['raw_paired_both']:.3f}) -> OURS Struct-Canon={s['struct_balanced']:.3f}(both={s['struct_paired_both']:.3f})", flush=True)

    out_p = args.out_dir / f"exp26a_p7_vlm_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "n_pairs": len(pairs), "summary": summary}, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "sheet", "eval_embed", "eval_vlm"], required=True)
    ap.add_argument("--manifest48", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--edit-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/hg-Qwen-Image-Edit-2511")
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int, default=24)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    args = ap.parse_args()
    if args.stage == "gen":
        return stage_gen(args)
    if args.stage == "sheet":
        return stage_sheet(args)
    if args.stage == "eval_embed":
        return stage_eval_embed(args)
    return stage_eval_vlm(args)


if __name__ == "__main__":
    raise SystemExit(main())
