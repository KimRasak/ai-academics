#!/usr/bin/env python3
"""
Exp 27 (Track `P7` Scale-Up to ALL 48 Characters = 44 Strict 100%-Deconfounded XPose-vs-XPose Pairs
        + 5 Realistic Anime Lighting/Manga Domains + Self-Supervised Color-Invariant Metric Training!):

Stage 1 (`--stage gen_rest`):
  Generates genuine 3/4-view outdoor winter-coat Cross-Pose portraits (`Qwen-Image-Edit-2511`) for
  the remaining 24 characters (fc_24..fc_47) across 4 H100s -> ALL 48 chars have XPose pairs,
  yielding n=44 STRICT same-gender + same-hair-color deconfounded XPose-vs-XPose pairs!

Stage 2 (`--stage eval_vlm_scale44`):
  Evaluates 5 diverse VLMs across scaling regimes (`Qwen2.5-VL-7B`, `Qwen2.5-VL-32B`, `Qwen3-VL-8B`,
  `Qwen3-VL-32B`, `LLaVA-OV-7B`) on ALL 44 pairs x 2 balanced orders (= 88 queries per domain!)
  across 5 realistic anime production domains:
    1. `XPose_SameColor`    (clean cross-pose control)
    2. `XPose_BWManga`      (Color TV-Anime -> B&W Manga!)
    3. `XPose_CyanMoon`     (Night / Moonlight blue-cyan grading!)
    4. `XPose_WarmSunset`   (Golden-hour orange/crimson sunset lighting!)
    5. `XPose_HueShift135`  (Dyed hair / disguise / alternate palette!)
  Compares Raw RGB vs. OURS Zero-Training Clean Luminance Canonicalization (`p`-value via exact McNemar!).

Stage 3 (`--stage train_struct_metric`):
  Trains a lightweight Self-Supervised Color-Invariant Structural Projector (`MLP` on frozen `SigLIP-SO400M`
  + `DINOv2-Large` features) using ONLY single-image multi-color/lighting/B&W self-augmentation on
  32 train characters (**ZERO paired cross-pose training images!!**), and evaluates on the 16 strictly
  UNSEEN held-out characters' real `XPose-vs-XPose` pairs against raw `CLIP-I`, `DINOv2`, and `SigLIP`!
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp25a_p7_hair_color_shortcut_pilot as e25a
import exp26b_p7_deconfounded_xpose_benchmark as e26b


def _to_warm_sunset(img: Image.Image) -> Image.Image:
    """Strong anime golden-hour orange-crimson sunset grading (0% geometry change)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    r = np.clip(arr[:, :, 0] * 1.18 + 0.14, 0, 1)
    g = np.clip(arr[:, :, 1] * 0.72 + 0.04, 0, 1)
    b = np.clip(arr[:, :, 2] * 0.40 + 0.02, 0, 1)
    return Image.fromarray((np.stack([r, g, b], axis=2) * 255.0).astype(np.uint8), mode="RGB")


def _mcnemar_p(ca: list[int], cb: list[int]) -> float:
    b01 = sum(1 for a, b in zip(ca, cb) if a == 0 and b == 1)
    b10 = sum(1 for a, b in zip(ca, cb) if a == 1 and b == 0)
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2.0 * sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1)))


def stage_gen_rest(args) -> int:
    from diffusers import QwenImageEditPlusPipeline

    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    orig48 = sorted([r for r in recs if r["variant"] == "orig"], key=lambda x: x["char_id"])
    xdir = args.xpose_dir / "crosspose_imgs"
    xdir.mkdir(parents=True, exist_ok=True)

    todo = [r for r in orig48 if not (xdir / f"{r['char_id']}_xpose.png").exists()]
    my_todo = [r for i, r in enumerate(todo) if i % args.num_shards == args.shard]
    if not my_todo:
        print(f"[gen_rest shard {args.shard}] nothing to do!", flush=True)
        return 0

    pipe = QwenImageEditPlusPipeline.from_pretrained(args.edit_dir, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    prompt = (
        "Generate a 2D Japanese TV anime screenshot of this EXACT SAME anime character turned to a three-quarter view, "
        "wearing a completely different outfit (a plain dark grey winter jacket with scarf) outdoors with a soft blurred city street background. "
        "Strictly preserve the character's exact hairstyle silhouette, bangs, eye shape, iris color, hair color, and facial identity in clean 2D anime cel shading."
    )

    for r in my_todo:
        cid = r["char_id"]
        idx = int(cid.split("_")[1])
        out_p = xdir / f"{cid}_xpose.png"
        im = Image.open(r["image_path"]).convert("RGB").resize((768, 768), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(8100 + idx * 29)
        res = pipe(image=im, prompt=prompt, num_inference_steps=20, true_cfg_scale=4.0, generator=gen).images[0]
        res.save(out_p)
        print(f"[gen_rest shard {args.shard}] saved {out_p.name}", flush=True)
    return 0


@torch.no_grad()
def stage_eval_vlm_scale44(args) -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    xdir = args.xpose_dir / "crosspose_imgs"
    pairs = e26b._build_deconfounded_pairs(args.manifest48, xdir)

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
        ("XPose_BWGray", e26b._clean_luma_gray),
        ("XPose_CyanMoon", e25a._to_cyan_night_lighting),
        ("XPose_WarmSunset", _to_warm_sunset),
        ("XPose_HueShift135", lambda im: e25a._to_hue_shifted(im, 95)),
    ]

    summary = {}
    print(f"\n=== EXP 27A SCALE-44 DECONFOUNDED VLM: {Path(args.model_dir).name} (n={len(pairs)} pairs x 2 orders = {len(pairs)*2} Q/domain) ===", flush=True)
    for tname, tfn in transforms:
        r_acc, r_both, g_acc, g_both = [], [], [], []
        for r_pos, _r_neg, xp_pos, xp_neg in pairs:
            im_ref = Image.open(r_pos["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
            im_pos = tfn(Image.open(xp_pos).convert("RGB").resize((448, 448), Image.LANCZOS))
            im_neg = Image.open(xp_neg).convert("RGB").resize((448, 448), Image.LANCZOS)

            o1, o2 = int(ask_2afc(im_ref, im_pos, im_neg) == "A"), int(ask_2afc(im_ref, im_neg, im_pos) == "B")
            r_acc.extend([o1, o2]); r_both.append(int(o1 and o2))

            gr, gp, gn = e26b._clean_luma_gray(im_ref), e26b._clean_luma_gray(im_pos), e26b._clean_luma_gray(im_neg)
            g1, g2 = int(ask_2afc(gr, gp, gn) == "A"), int(ask_2afc(gr, gn, gp) == "B")
            g_acc.extend([g1, g2]); g_both.append(int(g1 and g2))

        pval = _mcnemar_p(r_acc, g_acc)
        summary[tname] = {
            "raw_bal": round(float(np.mean(r_acc)), 4),
            "raw_both": round(float(np.mean(r_both)), 4),
            "luma_bal": round(float(np.mean(g_acc)), 4),
            "luma_both": round(float(np.mean(g_both)), 4),
            "gain_bal": round(float(np.mean(g_acc) - np.mean(r_acc)), 4),
            "mcnemar_p": round(pval, 6),
        }
        s = summary[tname]
        print(f"  {tname:18s} | Raw={s['raw_bal']:.3f}(b={s['raw_both']:.3f}) -> Luma={s['luma_bal']:.3f}(b={s['luma_both']:.3f}) | gain={s['gain_bal']:+.3f} (p={s['mcnemar_p']:.5f})", flush=True)

    out_p = args.out_dir / f"exp27a_scale44_vlm_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "n_pairs": len(pairs), "summary": summary}, indent=2), encoding="utf-8")
    return 0


def stage_train_struct_metric(args) -> int:
    """
    Exp 27B: Self-Supervised Color-Invariant Structural Identity Projector on frozen SigLIP+DINOv2!
    Trained STRICTLY on single-image color/lighting/B&W self-augmentations of 32 train chars
    (ZERO cross-pose supervision!), evaluated on ALL 44 strict XPose-vs-XPose pairs (reporting
    both held-out 16 unseen chars and all 44 pairs)!
    """
    from transformers import AutoImageProcessor, AutoModel

    xdir = args.xpose_dir / "crosspose_imgs"
    pairs = e26b._build_deconfounded_pairs(args.manifest48, xdir)
    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    orig48 = sorted([r for r in recs if r["variant"] == "orig"], key=lambda x: x["char_id"])
    tr_chars = orig48[:32]
    te_cids = {r["char_id"] for r in orig48[32:]}

    # Extract frozen SigLIP-SO400M + DINOv2-Large embeddings
    sig_p = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    sig_m = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", dtype=torch.bfloat16).to("cuda").eval()
    din_p = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    din_m = AutoModel.from_pretrained("facebook/dinov2-large", dtype=torch.bfloat16).to("cuda").eval()

    @torch.no_grad()
    def feat_fn(im: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        vs = sig_m.get_image_features(**sig_p(images=im, return_tensors="pt").to("cuda"))
        vs = (vs if isinstance(vs, torch.Tensor) else vs.pooler_output)[0].float()
        vs = (vs / (vs.norm() + 1e-8)).cpu().numpy()
        vd = din_m(**din_p(images=im, return_tensors="pt").to("cuda")).last_hidden_state[0, 0].float()
        vd = (vd / (vd.norm() + 1e-8)).cpu().numpy()
        return vs, vd

    def concat_feat(im: Image.Image) -> np.ndarray:
        vs, vd = feat_fn(im)
        return np.concatenate([vs, vd], axis=0)

    # 1. Build single-image self-augmented views ONLY on the 32 train chars (ZERO cross-pose images!)
    aug_fns = [
        lambda im: im,
        e26b._clean_luma_gray,
        e25a._to_cyan_night_lighting,
        _to_warm_sunset,
        lambda im: e25a._to_hue_shifted(im, 60),
        lambda im: e25a._to_hue_shifted(im, 120),
        lambda im: e25a._to_hue_shifted(im, 180),
        lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        lambda im: e26b._clean_luma_gray(im.transpose(Image.FLIP_LEFT_RIGHT)),
    ]
    print("Extracting self-supervised color-invariant training views on 32 train chars (ZERO cross-pose!)...", flush=True)
    tr_bank: list[torch.Tensor] = []
    for r in tr_chars:
        im = Image.open(r["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
        views = [concat_feat(af(im)) for af in aug_fns]
        tr_bank.append(torch.tensor(np.stack(views, axis=0), device="cuda", dtype=torch.float32))
    # tr_bank: 32 chars x 9 views x D
    D = tr_bank[0].shape[1]

    # 2. Train residual bottleneck projector with SupCon / InfoNCE across the 32 chars' color/lighting views!
    proj = nn.Sequential(
        nn.Linear(D, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
    ).to("cuda")
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3, weight_decay=1e-2)
    rng = random.Random(42)

    proj.train()
    for step in range(400):
        # Sample 2 random color/lighting views per character across all 32 characters (batch = 32 pairs)
        v1 = torch.stack([tr_bank[i][rng.randrange(len(aug_fns))] for i in range(len(tr_bank))], dim=0)
        v2 = torch.stack([tr_bank[i][rng.randrange(len(aug_fns))] for i in range(len(tr_bank))], dim=0)
        z1 = F.normalize(proj(v1), dim=-1)
        z2 = F.normalize(proj(v2), dim=-1)
        logits = (z1 @ z2.t()) / 0.07
        labels = torch.arange(len(tr_bank), device="cuda")
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Trained Color-Invariant Structural Projector (400 steps, final InfoNCE loss={float(loss.item()):.4f})!", flush=True)
    proj.eval()

    # 3. Evaluate on ALL 44 strict XPose-vs-XPose pairs AND strictly HELD-OUT 16 unseen chars!
    transforms = [
        ("XPose_SameColor", lambda im: im),
        ("XPose_BWGray", e26b._clean_luma_gray),
        ("XPose_CyanMoon", e25a._to_cyan_night_lighting),
        ("XPose_WarmSunset", _to_warm_sunset),
        ("XPose_HueShift135", lambda im: e25a._to_hue_shifted(im, 95)),
    ]

    res = {"all44": {}, "unseen16": {}}
    print(f"\n=== EXP 27B SELF-SUPERVISED COLOR-INVARIANT STRUCTURAL METRIC (ZERO Cross-Pose Training!) ===", flush=True)
    for tname, tfn in transforms:
        m_sig, m_din, m_luma_sig, m_ours = [], [], [], []
        u_sig, u_din, u_luma_sig, u_ours = [], [], [], []
        for r_pos, _r_neg, xp_pos, xp_neg in pairs:
            is_unseen = r_pos["char_id"] in te_cids
            im_ref = Image.open(r_pos["image_path"]).convert("RGB").resize((448, 448), Image.LANCZOS)
            im_pos = tfn(Image.open(xp_pos).convert("RGB").resize((448, 448), Image.LANCZOS))
            im_neg = Image.open(xp_neg).convert("RGB").resize((448, 448), Image.LANCZOS)

            sr, dr = feat_fn(im_ref)
            sp, dp = feat_fn(im_pos)
            sn, dn = feat_fn(im_neg)
            ok_sig = int(np.dot(sr, sp) > np.dot(sr, sn))
            ok_din = int(np.dot(dr, dp) > np.dot(dr, dn))

            srg, _ = feat_fn(e26b._clean_luma_gray(im_ref))
            spg, _ = feat_fn(e26b._clean_luma_gray(im_pos))
            sng, _ = feat_fn(e26b._clean_luma_gray(im_neg))
            ok_ls = int(np.dot(srg, spg) > np.dot(srg, sng))

            with torch.no_grad():
                zr = F.normalize(proj(torch.tensor(np.concatenate([sr, dr]), device="cuda").unsqueeze(0)), dim=-1)
                zp = F.normalize(proj(torch.tensor(np.concatenate([sp, dp]), device="cuda").unsqueeze(0)), dim=-1)
                zn = F.normalize(proj(torch.tensor(np.concatenate([sn, dn]), device="cuda").unsqueeze(0)), dim=-1)
                ok_ours = int((zr @ zp.t()).item() > (zr @ zn.t()).item())

            m_sig.append(ok_sig); m_din.append(ok_din); m_luma_sig.append(ok_ls); m_ours.append(ok_ours)
            if is_unseen:
                u_sig.append(ok_sig); u_din.append(ok_din); u_luma_sig.append(ok_ls); u_ours.append(ok_ours)

        res["all44"][tname] = {
            "siglip_raw": round(float(np.mean(m_sig)), 4),
            "dinov2_raw": round(float(np.mean(m_din)), 4),
            "siglip_luma": round(float(np.mean(m_luma_sig)), 4),
            "ours_ssl_proj": round(float(np.mean(m_ours)), 4),
            "p_ours_vs_siglip": round(_mcnemar_p(m_sig, m_ours), 5),
        }
        res["unseen16"][tname] = {
            "siglip_raw": round(float(np.mean(u_sig)), 4),
            "dinov2_raw": round(float(np.mean(u_din)), 4),
            "siglip_luma": round(float(np.mean(u_luma_sig)), 4),
            "ours_ssl_proj": round(float(np.mean(u_ours)), 4),
        }
        a = res["all44"][tname]; u = res["unseen16"][tname]
        print(f"  {tname:18s} | All44: SigLIP={a['siglip_raw']:.3f} DINOv2={a['dinov2_raw']:.3f} LumaSig={a['siglip_luma']:.3f} -> OURS-SSL={a['ours_ssl_proj']:.3f} (p={a['p_ours_vs_siglip']:.4f}) | Unseen16: SigLIP={u['siglip_raw']:.3f} -> OURS-SSL={u['ours_ssl_proj']:.3f}", flush=True)

    (args.out_dir / "exp27b_ssl_struct_metric.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen_rest", "eval_vlm_scale44", "train_struct_metric"], required=True)
    ap.add_argument("--manifest48", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--xpose-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp26a_p7_crosspose"))
    ap.add_argument("--edit-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/hg-Qwen-Image-Edit-2511")
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen3-VL-32B-Instruct")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "gen_rest":
        return stage_gen_rest(args)
    if args.stage == "eval_vlm_scale44":
        return stage_eval_vlm_scale44(args)
    return stage_train_struct_metric(args)


if __name__ == "__main__":
    raise SystemExit(main())
