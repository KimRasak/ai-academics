#!/usr/bin/env python3
"""
Exp 37B (Cross-Architecture Flagship Verification of "Physical World-Model Literalism"):

Verifies across TWO additional completely independent 2025/2026 frontier Video DiT families:
  1. `hunyuan15_i2v`: `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v`
     (54-layer MM-DiT, Qwen2.5-VL + ByT5 dual text encoders, SigLIP vision encoder, 32ch 3D-VAE)
  2. `ltx13b_i2v`: `Lightricks/LTX-Video-0.9.7-dev`
     (13B 48-layer Spatiotemporal DiT, T5-XXL text encoder, 128ch 3D-VAE)

Tests the EXACT same 6 deconfounded 2D Anime Grammar regimes (`G1`–`G6`) PLUS:
  - `S1_HeadTurn` (3D physical kinematics positive control — verifies the model works cleanly!)
  - `S3_NonBioAffect` (affective phrasing of forehead shadow + hollow eyes)
across 4 diverse characters (`00, 02, 04, 06` = 32 clips per flagship model).
Exports playable H.264 `.mp4`, 6-frame strips, stacked QC sheets, and leakage/chroma diagnostics!
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp31b_video_dit_anime_vs_real_diagnostic as e31b
import exp32b_wan22_causal_dit_surgery as e32b
import exp36_flagship_27b_13b_anime_stress_test as e36
import exp37a_flagship_2d_grammar_benchmark as e37a


CROSS_ARCH_REGIMES = [
    (
        "S1_HeadTurn",
        "The character smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent appearance.",
        "color",
    ),
    (
        "S3_NonBioAffect",
        "The character's bright eye highlights suddenly vanish leaving hollow dark lifeless eyes while a sharp ominous dark shadow falls across their upper forehead, yet their mouth stays frozen in a calm eerie smile, 2D Japanese TV anime cel shading.",
        "color",
    ),
] + list(e37a.GRAMMAR_REGIMES)


@torch.no_grad()
def stage_run(args) -> int:
    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    sel_pids = [0, 2, 4, 6]
    manga_dir = Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp29_p7_retrieval_manga/native_bw_manga_imgs")
    avail_manga = sorted(manga_dir.glob("*_manga.png"))
    manga_by_pid = {}
    for idx_i, pid in enumerate(sel_pids):
        cid = pairs[pid]["anime_id"]
        exact = manga_dir / f"{cid}_manga.png"
        manga_by_pid[pid] = exact if exact.exists() else avail_manga[idx_i % len(avail_manga)]

    tasks = []
    for pid in sel_pids:
        p = pairs[pid]
        for rname, rprompt, src_type in CROSS_ARCH_REGIMES:
            img_path = Path(p["anime_path"]) if src_type == "color" else manga_by_pid[pid]
            cid = p["anime_id"] if src_type == "color" else img_path.stem
            tasks.append((pid, img_path, cid, rname, rprompt))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    out_m = args.out_dir / args.model_tag
    meta_dir, strips_dir, mp4_dir = out_m / "meta", out_m / "strips", out_m / "mp4"
    for d in [meta_dir, strips_dir, mp4_dir]:
        d.mkdir(parents=True, exist_ok=True)

    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[3]}.json").exists()]
    if not todo:
        print(f"[{args.model_tag} s{args.shard}] all tasks complete!", flush=True)
        return 0

    if args.model_tag == "hunyuan15_i2v":
        from diffusers import HunyuanVideo15ImageToVideoPipeline
        pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
            torch_dtype=torch.bfloat16,
        )
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()
        W, H, N_FRAMES, STEPS = 720, 720, 49, 30
    elif args.model_tag == "ltx13b_i2v":
        from diffusers import LTXConditionPipeline
        pipe = LTXConditionPipeline.from_pretrained(
            "Lightricks/LTX-Video-0.9.7-dev",
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        W, H, N_FRAMES, STEPS = 512, 512, 65, 40
    else:
        raise ValueError(args.model_tag)

    for pid, img_path, cid, rname, rprompt in todo:
        im = Image.open(img_path).convert("RGB").resize((W, H), Image.LANCZOS)
        seed = 9800 + pid * 41 + [r[0] for r in CROSS_ARCH_REGIMES].index(rname)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        neg = "blurry, low quality, distorted, extra limbs, watermark, text"

        if args.model_tag == "hunyuan15_i2v":
            frames = pipe(
                image=im,
                prompt=rprompt,
                negative_prompt=neg,
                num_frames=N_FRAMES,
                num_inference_steps=STEPS,
                generator=gen,
            ).frames[0]
        else:
            frames = pipe(
                image=im,
                prompt=rprompt,
                negative_prompt=neg,
                height=H,
                width=W,
                num_frames=N_FRAMES,
                num_inference_steps=STEPS,
                guidance_scale=3.0,
                generator=gen,
            ).frames[0]

        pil_frames = e31b._to_pil_frames(frames)
        # Ensure consistent size vs input for pixel diagnostics
        if pil_frames[0].size != im.size:
            im_cmp = im.resize(pil_frames[0].size, Image.LANCZOS)
        else:
            im_cmp = im
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im_cmp, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["flat_delta_vs_inp"] = round(st["flat_end"] - inp_st["flat_t0"], 4)
        st["airbrush_delta_vs_inp"] = round(st["airbrush_end"] - inp_st["airbrush_t0"], 4)
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))
        st.update(e37a._grammar_diagnostics(np_frames, inp_u8))

        key = f"{pid:02d}_{rname}"
        e36._save_single_mp4(pil_frames, mp4_dir / f"{key}.mp4", fps=16)
        e31b._save_strip(
            pil_frames,
            strips_dir / f"{key}.png",
            f"{args.model_tag} | {key} ({cid}) | dL_glob={st['global_luma_delta']:+.2f} dL_low={st['lower_face_luma_delta']:+.2f} shR={st['sharp_ratio_vs_inp']:.2f} chrPk={st['chroma_peak']:.2f}",
            n_show=6,
            thumb_h=180,
        )
        kf_dir = out_m / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im_cmp.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        rec = {"model": args.model_tag, "pair_idx": pid, "char_id": cid, "img_path": str(img_path), "regime": rname, "prompt": rprompt, "stats": st}
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[{args.model_tag} s{args.shard}] {key:28s} | mot={st['motion_l1']:.4f} dL_glob={st['global_luma_delta']:+.3f} "
            f"dL_hair={st['hair_luma_delta']:+.3f} dL_low={st['lower_face_luma_delta']:+.3f} | shR={st['sharp_ratio_vs_inp']:.3f} chrPk={st['chroma_peak']:.3f}",
            flush=True,
        )
    return 0


@torch.no_grad()
def stage_eval(args) -> int:
    from transformers import CLIPModel, CLIPProcessor

    out_m = args.out_dir / args.model_tag
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_m / "meta").glob("*.json"))]

    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.bfloat16).to("cuda").eval()
    for r in recs:
        key = f"{r['pair_idx']:02d}_{r['regime']}"
        kf_dir = out_m / "keyframes" / key
        ims = [ImageOps.autocontrast(ImageOps.grayscale(Image.open(kf_dir / fn).convert("RGB")), cutoff=1).convert("RGB") for fn in ["input.png", "k0.png", "k1.png", "k2.png", "k3.png", "k4.png", "k5.png"]]
        pv = cp(images=ims, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)
        f = cm.get_image_features(pixel_values=pv)
        if not isinstance(f, torch.Tensor): f = f.pooler_output
        e = F.normalize(f.float(), dim=-1)
        sims = (e[1:] @ e[0]).cpu().numpy().tolist()
        r["luma_id_mean"] = round(float(np.mean(sims[1:])), 4)
        r["luma_id_end"] = round(float(sims[-1]), 4)

    print(f"\n=== EXP 37B CROSS-ARCH FLAGSHIP GRAMMAR SUMMARY: {args.model_tag} (n={len(recs)} clips) ===", flush=True)
    agg = {}
    for rname, _, _ in CROSS_ARCH_REGIMES:
        sub = [r for r in recs if r["regime"] == rname]
        if not sub: continue
        m = {
            "n": len(sub),
            "mot": round(float(np.mean([r["stats"]["motion_l1"] for r in sub])), 4),
            "dL_glob": round(float(np.mean([r["stats"]["global_luma_delta"] for r in sub])), 4),
            "dL_hair": round(float(np.mean([r["stats"]["hair_luma_delta"] for r in sub])), 4),
            "dL_up": round(float(np.mean([r["stats"]["upper_face_luma_delta"] for r in sub])), 4),
            "dL_low": round(float(np.mean([r["stats"]["lower_face_luma_delta"] for r in sub])), 4),
            "shR": round(float(np.mean([r["stats"]["sharp_ratio_vs_inp"] for r in sub])), 4),
            "chr_in": round(float(np.mean([r["stats"]["chroma_in"] for r in sub])), 4),
            "chr_pk": round(float(np.mean([r["stats"]["chroma_peak"] for r in sub])), 4),
            "luma_id_end": round(float(np.mean([r["luma_id_end"] for r in sub])), 4),
        }
        agg[rname] = m
        print(
            f"  {rname:25s} | mot={m['mot']:.4f} | dL(glob={m['dL_glob']:+.3f} hair={m['dL_hair']:+.3f} up={m['dL_up']:+.3f} low={m['dL_low']:+.3f}) | "
            f"shR={m['shR']:.3f} chr({m['chr_in']:.3f}->{m['chr_pk']:.3f}) | LumaID_end={m['luma_id_end']:.3f}",
            flush=True,
        )
        paths = [out_m / "strips" / f"{pid:02d}_{rname}.png" for pid in [0, 2, 4, 6] if (out_m / "strips" / f"{pid:02d}_{rname}.png").exists()]
        e31b._stack_strips(paths, out_m / f"qc37b_{args.model_tag}_{rname}.png")

    (out_m / f"summary_{args.model_tag}.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "eval"])
    ap.add_argument("--model-tag", required=True, choices=["hunyuan15_i2v", "ltx13b_i2v"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp37b_cross_arch_grammar"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_run(args) if args.stage == "run" else stage_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
