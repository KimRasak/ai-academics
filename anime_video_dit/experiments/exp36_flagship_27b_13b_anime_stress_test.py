#!/usr/bin/env python3
"""
Exp 36 (Flagship-Scale Video DiT Track — `Wan2.2-I2V-A14B` 27B MoE & `HunyuanVideo-I2V` 13B MM-DiT!):

Directly answers the core scaling question:
  "Were the 5B Video DiT anime failures merely due to small parameter/data scale, OR what genuine,
   unsolved 2D anime character failures PERSIST even on 13B–27B MoE flagship Video DiTs
   (MiniMax-H3 parameter & training scale)?"

Evaluates BOTH frontier open-weight flagship Video I2V DiTs on 100% framing-matched square (`512x512`)
81-frame (5s, Wan2.2-A14B) / 61-frame (HunyuanVideo-I2V) clips across 6 flagship 2D anime stress regimes
(plus 1-to-1 matched Photorealistic Real Human controls on S1, S2, S4, S5):
  - `S1_HeadTurn`:          3/4 profile head turn (direct scale comparison vs 5B)
  - `S2_DynamicWind`:       Strong wind + hand brushing bangs back (direct scale comparison vs 5B)
  - `S3_NonBioAffect`:      Hollow highlight-free eyes + sharp forehead shadow while keeping frozen smile!
  - `S4_180_TurnReturn`:    Long-horizon 5s self-occlusion & re-appearance (turn completely away -> turn back!)
  - `S5_DramaticRelightCel`:Sweeping golden sunset beam across face (2D hard cel shadow vs 3D airbrush gravity!)
  - `S6_FastActionSmear`:   High-velocity shock whip turn + both hands raised to cheeks!

Exports BOTH:
  - Browser-playable H.264 `.mp4` videos for EVERY generated clip
  - 6-frame visual inspection contact strips (`t=0..T-1`) + stacked Anime-vs-Real & Cross-Regime QC sheets
  - Full quantitative diagnostics (`motion`, `flat_delta`, `airbrush_delta`, `sharpR`, `inkR`, `boil`,
    `satOv`, `jmp`, AND `Luma-CLIP` start-to-end / occlusion-return structural identity!)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp31b_video_dit_anime_vs_real_diagnostic as e31b
import exp32b_wan22_causal_dit_surgery as e32b
import exp35b_export_showcase_mp4s as e35b


FLAGSHIP_REGIMES = [
    (
        "S1_HeadTurn",
        "The character smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent appearance.",
        True,  # run matched real control too
    ),
    (
        "S2_DynamicWind",
        "A sudden gust of wind blows the character's hair strongly across their face as they raise one hand to brush their bangs back with a surprised expression, dynamic lively motion.",
        True,
    ),
    (
        "S3_NonBioAffect",
        "The character's bright eye highlights suddenly vanish leaving hollow dark lifeless eyes while a sharp ominous dark shadow falls across their upper forehead, yet their mouth stays frozen in a calm eerie smile, 2D Japanese TV anime cel shading.",
        False,
    ),
    (
        "S4_180_TurnReturn",
        "The character turns completely away to show the back of their head, pauses briefly, then turns smoothly back around to face the camera with the exact same face and gentle smile, consistent character appearance.",
        True,
    ),
    (
        "S5_DramaticRelightCel",
        "A bright golden sunset light beam sweeps slowly across the character's face as they glance sideways toward the window, casting a crisp hard-edged shadow while preserving flat cel-shaded colors and clean ink outlines.",
        True,
    ),
    (
        "S6_FastActionSmear",
        "The character suddenly whips their head sideways in shock, hair flying rapidly outward as they bring both hands up near their cheeks with wide surprised eyes, expressive energetic animation.",
        False,
    ),
]


def _save_single_mp4(frames: list[Image.Image], out_mp4: Path, fps: int = 16) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp4.parent / f".tmp_{out_mp4.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        w, h = f.size
        if w % 2 or h % 2:
            f = f.crop((0, 0, w - (w % 2), h - (h % 2)))
        f.save(tmp / f"f_{i:03d}.png")
    e35b._encode_dir_to_mp4(tmp, out_mp4, fps=fps)


@torch.no_grad()
def stage_run(args) -> int:
    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    sel_pids = [0, 2, 4, 6]  # 4 diverse characters (navy girl, silver boy, black-hair girl, lavender girl)

    tasks = []
    for pid in sel_pids:
        p = pairs[pid]
        for rname, rprompt, run_real in FLAGSHIP_REGIMES:
            tasks.append((pid, "anime", Path(p["anime_path"]), p["anime_id"], rname, rprompt))
            if run_real:
                tasks.append((pid, "real", Path(p["real_path"]), p["real_id"], rname, rprompt))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    out_m = args.out_dir / args.model_tag
    meta_dir = out_m / "meta"
    strips_dir = out_m / "strips"
    mp4_dir = out_m / "mp4"
    meta_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir.mkdir(parents=True, exist_ok=True)

    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[1]}_{t[4]}.json").exists()]
    if not todo:
        print(f"[{args.model_tag} s{args.shard}] all tasks complete!", flush=True)
        return 0

    if args.model_tag == "wan22_a14b":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
        # Model CPU offload keeps only the active 14B expert (high-noise or low-noise) on GPU (~32GB VRAM peak!)
        pipe.enable_model_cpu_offload()
        H, W, N_FRAMES, STEPS = 512, 512, 81, 40
    elif args.model_tag == "hunyuan_i2v":
        from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo-I2V", subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo-I2V", transformer=transformer, torch_dtype=torch.float16
        )
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()
        H, W, N_FRAMES, STEPS = 512, 512, 61, 30
    else:
        raise ValueError(args.model_tag)

    for pid, dom, img_path, cid, rname, rprompt in todo:
        im = Image.open(img_path).convert("RGB").resize((W, H), Image.LANCZOS)
        seed = 8600 + pid * 31 + [r[0] for r in FLAGSHIP_REGIMES].index(rname)
        gen = torch.Generator(device="cuda").manual_seed(seed)

        if args.model_tag == "wan22_a14b":
            frames = pipe(
                image=im,
                prompt=rprompt,
                negative_prompt="blurry, low quality, distorted, extra limbs, watermark, text",
                height=H,
                width=W,
                num_frames=N_FRAMES,
                num_inference_steps=STEPS,
                guidance_scale=3.5,
                guidance_scale_2=3.5,
                generator=gen,
            ).frames[0]
        else:
            frames = pipe(
                image=im,
                prompt=rprompt,
                height=H,
                width=W,
                num_frames=N_FRAMES,
                num_inference_steps=STEPS,
                guidance_scale=6.0,
                generator=gen,
            ).frames[0]

        pil_frames = e31b._to_pil_frames(frames)
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["inp_flat_t0"] = inp_st["flat_t0"]
        st["inp_airbrush_t0"] = inp_st["airbrush_t0"]
        st["flat_delta_vs_inp"] = round(st["flat_end"] - inp_st["flat_t0"], 4)
        st["airbrush_delta_vs_inp"] = round(st["airbrush_end"] - inp_st["airbrush_t0"], 4)
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))

        key = f"{pid:02d}_{dom}_{rname}"
        _save_single_mp4(pil_frames, mp4_dir / f"{key}.mp4", fps=16)
        e31b._save_strip(
            pil_frames,
            strips_dir / f"{key}.png",
            f"{args.model_tag} | {key} ({cid}) | mot={st['motion_l1']:.3f} dFlat={st['flat_delta_vs_inp']:+.3f} dAir={st['airbrush_delta_vs_inp']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f} satOv={st['sat_overshoot']:+.2f}",
            n_show=6,
            thumb_h=180,
        )
        kf_dir = out_m / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            fi = round(k_i * (T - 1) / 5)
            pil_frames[fi].save(kf_dir / f"k{k_i}.png")

        rec = {"model": args.model_tag, "pair_idx": pid, "domain": dom, "char_id": cid, "regime": rname, "prompt": rprompt, "stats": st}
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[{args.model_tag} s{args.shard}] {key:30s} | mot={st['motion_l1']:.4f} dFlat={st['flat_delta_vs_inp']:+.3f} dAir={st['airbrush_delta_vs_inp']:+.3f} | "
            f"shR={st['sharp_ratio_vs_inp']:.3f} inkR={st['dark_ink_ratio']:.3f} boil={st['edge_boiling']:.4f} satOv={st['sat_overshoot']:+.3f}",
            flush=True,
        )
    return 0


@torch.no_grad()
def stage_eval(args) -> int:
    from transformers import CLIPModel, CLIPProcessor

    out_m = args.out_dir / args.model_tag
    meta_dir = out_m / "meta"
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(meta_dir.glob("*.json"))]

    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.bfloat16).to("cuda").eval()
    for r in recs:
        key = f"{r['pair_idx']:02d}_{r['domain']}_{r['regime']}"
        kf_dir = out_m / "keyframes" / key
        ims = [ImageOps.autocontrast(ImageOps.grayscale(Image.open(kf_dir / fn).convert("RGB")), cutoff=1).convert("RGB") for fn in ["input.png", "k0.png", "k1.png", "k2.png", "k3.png", "k4.png", "k5.png"]]
        pv = cp(images=ims, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)
        f = cm.get_image_features(pixel_values=pv)
        if not isinstance(f, torch.Tensor): f = f.pooler_output
        e = F.normalize(f.float(), dim=-1)
        sims = (e[1:] @ e[0]).cpu().numpy().tolist()  # k0..k5 vs input
        r["luma_id_mean"] = round(float(np.mean(sims[1:])), 4)
        r["luma_id_end"] = round(float(sims[-1]), 4)
        r["luma_id_min"] = round(float(np.min(sims[1:])), 4)

    print(f"\n=== EXP 36 FLAGSHIP STRESS-TEST SUMMARY: {args.model_tag} (n={len(recs)} clips) ===", flush=True)
    agg = {}
    for rname, _, _ in FLAGSHIP_REGIMES:
        agg[rname] = {}
        for dom in ["anime", "real"]:
            sub = [r for r in recs if r["regime"] == rname and r["domain"] == dom]
            if not sub: continue
            m = {
                "n": len(sub),
                "mot": round(float(np.mean([r["stats"]["motion_l1"] for r in sub])), 4),
                "dFlat": round(float(np.mean([r["stats"]["flat_delta_vs_inp"] for r in sub])), 4),
                "dAir": round(float(np.mean([r["stats"]["airbrush_delta_vs_inp"] for r in sub])), 4),
                "shR": round(float(np.mean([r["stats"]["sharp_ratio_vs_inp"] for r in sub])), 4),
                "inkR": round(float(np.mean([r["stats"]["dark_ink_ratio"] for r in sub])), 4),
                "boil": round(float(np.mean([r["stats"]["edge_boiling"] for r in sub])), 4),
                "satOv": round(float(np.mean([r["stats"]["sat_overshoot"] for r in sub])), 4),
                "luma_id_mean": round(float(np.mean([r["luma_id_mean"] for r in sub])), 4),
                "luma_id_end": round(float(np.mean([r["luma_id_end"] for r in sub])), 4),
            }
            agg[rname][dom] = m
            print(
                f"  {rname:22s} {dom:5s} (n={m['n']}) | mot={m['mot']:.4f} | dFlat={m['dFlat']:+.3f} dAir={m['dAir']:+.3f} | "
                f"shR={m['shR']:.3f} inkR={m['inkR']:.3f} boil={m['boil']:.4f} satOv={m['satOv']:+.3f} | LumaID(mean={m['luma_id_mean']:.3f}, end={m['luma_id_end']:.3f})",
                flush=True,
            )
        # Stack QC sheet per regime
        paths = []
        for pid in [0, 2, 4, 6]:
            for dom in ["anime", "real"]:
                sp = out_m / "strips" / f"{pid:02d}_{dom}_{rname}.png"
                if sp.exists(): paths.append(sp)
        e31b._stack_strips(paths, out_m / f"qc36_{args.model_tag}_{rname}.png")

    (out_m / f"summary_{args.model_tag}.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "eval"])
    ap.add_argument("--model-tag", required=True, choices=["wan22_a14b", "hunyuan_i2v"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp36_flagship_27b_13b"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_run(args) if args.stage == "run" else stage_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
