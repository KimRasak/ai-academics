#!/usr/bin/env python3
"""
Exp 30B (Track `P7` — Decomposing Color-Palette vs. Rendering-Style Shortcuts on 100% Native Hand-Inked B&W Manga!):

Exp 29B revealed a striking two-layer phenomenon on genuine hand-inked B&W manga panels:
  1. Inverse scaling is EVEN MORE SEVERE on native hand-inked manga (`Qwen3 8B->32B`: 76.1% -> 34.8%,
     `-41.3%`!; `Qwen2.5 7B->32B`: 65.2% -> 28.3%, `-36.9%`!; `InternVL3-8B`: 10.9%!).
  2. Plain `Luma` grayscale removes the Color shortcut (+9% to +15% on 32B/InternVL3) but stops at ~43.5%
     because `Ref` (smooth TV cel-anime) and `HardNeg` (smooth TV cel-anime) STILL share smooth shading
     and dark street backgrounds, whereas `NativeMangaPos` has crisp black ink lineart + white paper!

Tests whether **Style-Invariant Ink/Lineart Canonicalization** (`XDoG / Sketch-Tone` mapping ALL 3 images
into unified monochrome ink lineart + light tone, removing BOTH color AND smooth cel-shading texture)
— with and without Head-Crop — rescues ALL 4 VLMs AND Dense Encoders (`DINOv2 / CLIP / SigLIP`) on
100% Native Hand-Inked B&W Manga!
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp25a_p7_hair_color_shortcut_pilot as e25a
import exp26b_p7_deconfounded_xpose_benchmark as e26b
import exp27_p7_scale44_and_ssl_metric as e27


def _to_manga_ink_canonical(img: Image.Image) -> Image.Image:
    """
    Zero-training Difference-of-Gaussians (XDoG-style) + 3-level manga tone canonicalizer.
    Maps BOTH smooth cel-shaded anime AND hand-inked manga into the SAME crisp B&W ink+tone space!
    """
    g = ImageOps.autocontrast(ImageOps.grayscale(img), cutoff=1)
    g1 = np.asarray(g.filter(ImageFilter.GaussianBlur(radius=0.8)), dtype=np.float32)
    g2 = np.asarray(g.filter(ImageFilter.GaussianBlur(radius=2.2)), dtype=np.float32)
    # Crisp ink edges where fine < coarse (dark strokes)
    dog = g1 - 0.97 * g2
    ink = np.where(dog < -3.0, 0.0, np.clip(255.0 + 18.0 * dog, 0.0, 255.0))
    # Subtle quantized manga screentone shadow layer from luminance
    lum = np.asarray(g, dtype=np.float32)
    tone = np.where(lum < 75.0, 75.0, np.where(lum < 155.0, 185.0, 250.0))
    merged = np.minimum(ink, tone)
    return Image.fromarray(np.clip(merged, 0, 255).astype(np.uint8), mode="L").convert("RGB")


def _head_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((int(w * 0.10), int(h * 0.00), int(w * 0.90), int(h * 0.70))).resize((448, 448), Image.LANCZOS)


def _load_native_pairs(manifest48: Path, xdir: Path, mdir: Path):
    all_pairs = e26b._build_deconfounded_pairs(manifest48, xdir)
    pairs = []
    for r_pos, r_neg, _xp_pos, xp_neg in all_pairs:
        mp = mdir / f"{r_pos['char_id']}_manga.png"
        if mp.exists():
            pairs.append((r_pos, r_neg, mp, xp_neg))
    return pairs


@torch.no_grad()
def eval_vlm(args) -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    xdir = args.xpose_dir / "crosspose_imgs"
    mdir = args.manga_dir / "native_bw_manga_imgs"
    pairs = _load_native_pairs(args.manifest48, xdir, mdir)

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ab_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in ["A", "B"]]

    def pred_2afc(ref: Image.Image, cA: Image.Image, cB: Image.Image) -> int:
        m = [{"role": "user", "content": [
            {"type": "image", "image": ref},
            {"type": "image", "image": cA},
            {"type": "image", "image": cB},
            {"type": "text", "text": e25a.PROMPT_REID},
        ]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[ref, cA, cB], return_tensors="pt").to(model.device)
        lg = model(**inp).logits[0, -1, ab_ids].float().cpu().numpy()
        return int(np.argmax(lg))

    modes = [
        ("RawRGB", lambda im: im),
        ("LumaGray", e26b._clean_luma_gray),
        ("MangaInk_XDoG", _to_manga_ink_canonical),
        ("MangaInk_HeadCrop", lambda im: _to_manga_ink_canonical(_head_crop(im))),
    ]
    res_ok = {k: [] for k, _ in modes}
    for r_pos, _r_neg, mp_pos, xp_neg in pairs:
        ir = Image.open(r_pos["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
        ip = Image.open(mp_pos).convert("RGB").resize((448, 448), Image.LANCZOS)
        in_ = Image.open(xp_neg).convert("RGB").resize((448, 448), Image.LANCZOS)
        for mname, mfn in modes:
            tr, tp, tn = mfn(ir), mfn(ip), mfn(in_)
            res_ok[mname].append(int(pred_2afc(tr, tp, tn) == 0))
            res_ok[mname].append(int(pred_2afc(tr, tn, tp) == 1))

    summary = {}
    print(f"\n=== EXP 30B STYLE-INVARIANT MANGA-INK CANONICALIZATION: {Path(args.model_dir).name} (n={len(pairs)} pairs, {2*len(pairs)} Q) ===", flush=True)
    for mname, _ in modes:
        acc = float(np.mean(res_ok[mname]))
        p = e27._mcnemar_p(res_ok["RawRGB"], res_ok[mname]) if mname != "RawRGB" else 1.0
        summary[mname] = {"acc": round(acc, 4), "p_vs_raw": round(p, 6)}
        print(f"  {mname:20s} | Acc={acc:.3f} (p_vs_raw={p:.6f})", flush=True)

    out_p = args.out_dir / f"exp30b_mangaink_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "n_pairs": len(pairs), "summary": summary}, indent=2), encoding="utf-8")
    return 0


@torch.no_grad()
def eval_encoders(args) -> int:
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    xdir = args.xpose_dir / "crosspose_imgs"
    mdir = args.manga_dir / "native_bw_manga_imgs"
    pairs = _load_native_pairs(args.manifest48, xdir, mdir)

    modes = [
        ("RawRGB", lambda im: im),
        ("LumaGray", e26b._clean_luma_gray),
        ("MangaInk_XDoG", _to_manga_ink_canonical),
        ("MangaInk_HeadCrop", lambda im: _to_manga_ink_canonical(_head_crop(im))),
    ]

    results = {}

    def run_enc(name: str, enc_fn):
        results[name] = {}
        print(f"\n[{name}] Native Hand-Inked B&W Manga 2AFC (n={len(pairs)} strict pairs):", flush=True)
        for mname, mfn in modes:
            refs, poss, negs = [], [], []
            for r_pos, _r_neg, mp_pos, xp_neg in pairs:
                refs.append(mfn(Image.open(r_pos["image_path"]).convert("RGB").resize((384, 384), Image.LANCZOS)))
                poss.append(mfn(Image.open(mp_pos).convert("RGB").resize((384, 384), Image.LANCZOS)))
                negs.append(mfn(Image.open(xp_neg).convert("RGB").resize((384, 384), Image.LANCZOS)))
            er, ep, en = enc_fn(refs), enc_fn(poss), enc_fn(negs)
            sp = (er * ep).sum(dim=-1).cpu().numpy()
            sn = (er * en).sum(dim=-1).cpu().numpy()
            ok = (sp > sn).astype(int).tolist()
            acc = float(np.mean(ok))
            results[name][mname] = {"acc": round(acc, 4), "margin": round(float(np.mean(sp - sn)), 4)}
            print(f"  {mname:20s} | Acc={acc:.3f} | Margin={float(np.mean(sp - sn)):+.4f}", flush=True)

    # DINOv2-Large
    dp = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    dm = AutoModel.from_pretrained("facebook/dinov2-large", dtype=torch.bfloat16).to("cuda").eval()
    run_enc("DINOv2-Large", lambda ims: F.normalize(dm(**dp(images=ims, return_tensors="pt").to("cuda", dtype=torch.bfloat16)).last_hidden_state[:, 0].float(), dim=-1))
    del dm; torch.cuda.empty_cache()

    # CLIP-ViT-L/14
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.bfloat16).to("cuda").eval()
    def enc_clip(ims):
        f = cm.get_image_features(pixel_values=cp(images=ims, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16))
        if not isinstance(f, torch.Tensor):
            f = f.pooler_output
        return F.normalize(f.float(), dim=-1)
    run_enc("CLIP-ViT-L/14", enc_clip)
    del cm; torch.cuda.empty_cache()

    # SigLIP-SO400M
    sp_p = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    sm = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", dtype=torch.bfloat16).to("cuda").eval()
    def enc_siglip(ims):
        f = sm.get_image_features(pixel_values=sp_p(images=ims, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16))
        if not isinstance(f, torch.Tensor):
            f = f.pooler_output
        return F.normalize(f.float(), dim=-1)
    run_enc("SigLIP-SO400M", enc_siglip)

    out_p = args.out_dir / "exp30b_mangaink_encoders.json"
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["vlm", "encoders"])
    ap.add_argument("--manifest48", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--xpose-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp26a_p7_crosspose"))
    ap.add_argument("--manga-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp29_p7_retrieval_manga"))
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen3-VL-32B-Instruct")
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp30b_p7_mangaink"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return eval_vlm(args) if args.mode == "vlm" else eval_encoders(args)


if __name__ == "__main__":
    raise SystemExit(main())
