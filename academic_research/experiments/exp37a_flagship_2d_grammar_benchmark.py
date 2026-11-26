#!/usr/bin/env python3
"""
Exp 37A (Flagship-Scale Video DiT Benchmark — "The 3D Physical World-Model Trap on 2D Non-Physical Anime Grammar"):

Having proven in Exp 36 that scaling Video DiTs to 27B MoE (`Wan2.2-I2V-A14B`) SOLVES standard 3D physical kinematics
(head turns, wind + hands, even 180-deg self-occlusion return!), we now run a DECONFOUNDED 6-regime flagship test of
core 2D Japanese Animation Grammar that intentionally violates 3D photorealistic world-model physics — with ZERO horror
vocabulary confounds:

  - `G1_HighlightOff_Only`:   Micro-local specular extinction ONLY (NO "dark/shadow/ominous" words; explicit bright light)
  - `G2_ForeheadShadow_Clean`:Local 2D graphic forehead gloom bar ONLY (explicit bright studio daylight + unchanged hair/smile)
  - `G3_AngerVein_Pop`:       2D non-biological red cross anger vein symbol popping on temple while keeping polite smile
  - `G4_SweatDrop_Slide`:     2D non-biological oversized stylized blue anime sweat drop sliding down temple
  - `G5_StarryEyes_Morph`:    2D iconic non-biological eye morph (pupils turn into golden 4-pointed stars in excitement)
  - `G6_BWManga_Monochrome`:  Native hand-inked B&W Japanese manga portrait turning head (strictly monochrome ink + screentone)

Evaluates BOTH `Wan2.2-I2V-A14B` (27B MoE, 81f) and `HunyuanVideo-I2V` (13B MM-DiT, 61f), exporting browser-playable
H.264 `.mp4` clips, 6-frame strips, stacked QC sheets, and regime-specific physical-leakage diagnostics:
  - `global_luma_drop` (whole-frame blackout / room dimming)
  - `lower_face_luma_drop` (illumination leakage into cheeks/mouth)
  - `chroma_bleed` (spontaneous colorization on pure monochrome B&W manga)
  - `shR`, `inkR`, `dAir`, `satOv`, and `Luma-CLIP ID`!
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


GRAMMAR_REGIMES = [
    (
        "G1_HighlightOff_Only",
        "The character stays still in the exact same bright even studio lighting with unchanged bright hair, clothes, and gentle smile, while only the small white specular highlight dots inside both eyes smoothly fade away leaving flat matte dull irises, clean 2D Japanese TV anime.",
        "color",
    ),
    (
        "G2_ForeheadShadow_Clean",
        "In bright even studio daylight, a stylized flat slate-blue 2D anime gloom shadow band with fine vertical ink lines appears strictly across the character's upper forehead and brow while the character keeps their exact gentle smile, bright hair color, bright cheeks, clothes, and bright white background completely unchanged.",
        "color",
    ),
    (
        "G3_AngerVein_Pop",
        "In bright flat 2D Japanese anime style, the character holds a polite closed-mouth smile as a classic stylized red cross-shaped anime anger vein symbol pops onto their upper forehead temple and pulses gently, everything else stays bright, clean, and consistent.",
        "color",
    ),
    (
        "G4_SweatDrop_Slide",
        "In bright flat 2D Japanese anime cel shading, the character gives an awkward nervous smile as a single large stylized translucent blue comic sweat drop appears beside their temple and slides gently downward, clean crisp lineart.",
        "color",
    ),
    (
        "G5_StarryEyes_Morph",
        "In bright flat 2D Japanese anime comedy style, the character's eyes widen with joyful excitement as their pupils morph into bright glowing golden four-pointed star shapes sparkling inside their irises, consistent bright hair and face.",
        "color",
    ),
    (
        "G6_BWManga_Monochrome",
        "The black-and-white Japanese manga character smoothly turns their head to a three-quarter side profile and smiles gently, strictly maintaining pure monochrome black ink lines, crisp white paper, and fine halftone screentone dots with zero color.",
        "manga",
    ),
]


def _grammar_diagnostics(np_frames: list[np.ndarray], inp_u8: np.ndarray) -> dict:
    """Measure global blackout, lower-face luminance leakage, and B&W chroma bleed."""
    f0 = inp_u8.astype(np.float32) / 255.0
    fe = np_frames[-1].astype(np.float32) / 255.0
    H, W, _ = f0.shape

    luma0 = 0.299 * f0[..., 0] + 0.587 * f0[..., 1] + 0.114 * f0[..., 2]
    lumae = 0.299 * fe[..., 0] + 0.587 * fe[..., 1] + 0.114 * fe[..., 2]

    # Regions: top hair (0..22%), upper face/eyes (28..52%), lower face/mouth (52..72%), torso/clothes (72..100%)
    global_luma_delta = float(lumae.mean() - luma0.mean())
    hair_luma_delta = float(lumae[: int(0.22 * H), int(0.2 * W) : int(0.8 * W)].mean() - luma0[: int(0.22 * H), int(0.2 * W) : int(0.8 * W)].mean())
    upper_face_luma_delta = float(lumae[int(0.28 * H) : int(0.52 * H), int(0.28 * W) : int(0.72 * W)].mean() - luma0[int(0.28 * H) : int(0.52 * H), int(0.28 * W) : int(0.72 * W)].mean())
    lower_face_luma_delta = float(lumae[int(0.52 * H) : int(0.72 * H), int(0.32 * W) : int(0.68 * W)].mean() - luma0[int(0.52 * H) : int(0.72 * H), int(0.32 * W) : int(0.68 * W)].mean())
    torso_luma_delta = float(lumae[int(0.72 * H) :, int(0.2 * W) : int(0.8 * W)].mean() - luma0[int(0.72 * H) :, int(0.2 * W) : int(0.8 * W)].mean())

    # Chroma (max-min RGB) for B&W manga colorization detection
    chroma0 = float((f0.max(axis=-1) - f0.min(axis=-1)).mean())
    chromae = float((fe.max(axis=-1) - fe.min(axis=-1)).mean())
    chroma_max_frame = max(float(((fr.astype(np.float32) / 255.0).max(axis=-1) - (fr.astype(np.float32) / 255.0).min(axis=-1)).mean()) for fr in np_frames[1:])

    return {
        "global_luma_delta": round(global_luma_delta, 4),
        "hair_luma_delta": round(hair_luma_delta, 4),
        "upper_face_luma_delta": round(upper_face_luma_delta, 4),
        "lower_face_luma_delta": round(lower_face_luma_delta, 4),
        "torso_luma_delta": round(torso_luma_delta, 4),
        "chroma_in": round(chroma0, 4),
        "chroma_end": round(chromae, 4),
        "chroma_peak": round(chroma_max_frame, 4),
    }


@torch.no_grad()
def stage_run(args) -> int:
    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    sel_pids = [0, 2, 4, 6]
    manga_dir = Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp29_p7_retrieval_manga/native_bw_manga_imgs")

    # Match 4 clean native hand-inked B&W manga portraits (prefer exact same char IDs if present, else deterministic first 4)
    manga_by_pid = {}
    avail_manga = sorted(manga_dir.glob("*_manga.png"))
    for idx_i, pid in enumerate(sel_pids):
        cid = pairs[pid]["anime_id"]
        exact = manga_dir / f"{cid}_manga.png"
        manga_by_pid[pid] = exact if exact.exists() else avail_manga[idx_i % len(avail_manga)]

    tasks = []
    for pid in sel_pids:
        p = pairs[pid]
        for rname, rprompt, src_type in GRAMMAR_REGIMES:
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

    if args.model_tag == "wan22_a14b":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        H, W, N_FRAMES, STEPS = 512, 512, 81, 40
    elif args.model_tag == "hunyuan_i2v":
        from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo-I2V", subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo-I2V", transformer=transformer, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        H, W, N_FRAMES, STEPS = 720, 720, 61, 30
    else:
        raise ValueError(args.model_tag)

    for pid, img_path, cid, rname, rprompt in todo:
        im = Image.open(img_path).convert("RGB").resize((W, H), Image.LANCZOS)
        seed = 9700 + pid * 37 + [r[0] for r in GRAMMAR_REGIMES].index(rname)
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
        st["flat_delta_vs_inp"] = round(st["flat_end"] - inp_st["flat_t0"], 4)
        st["airbrush_delta_vs_inp"] = round(st["airbrush_end"] - inp_st["airbrush_t0"], 4)
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))
        st.update(_grammar_diagnostics(np_frames, inp_u8))

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
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        rec = {"model": args.model_tag, "pair_idx": pid, "char_id": cid, "img_path": str(img_path), "regime": rname, "prompt": rprompt, "stats": st}
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[{args.model_tag} s{args.shard}] {key:28s} | dL_glob={st['global_luma_delta']:+.3f} dL_hair={st['hair_luma_delta']:+.3f} "
            f"dL_up={st['upper_face_luma_delta']:+.3f} dL_low={st['lower_face_luma_delta']:+.3f} | shR={st['sharp_ratio_vs_inp']:.3f} chrPk={st['chroma_peak']:.3f}",
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

    print(f"\n=== EXP 37A 2D ANIME GRAMMAR BENCHMARK SUMMARY: {args.model_tag} (n={len(recs)} clips) ===", flush=True)
    agg = {}
    for rname, _, _ in GRAMMAR_REGIMES:
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
        e31b._stack_strips(paths, out_m / f"qc37a_{args.model_tag}_{rname}.png")

    (out_m / f"summary_{args.model_tag}.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "eval"])
    ap.add_argument("--model-tag", required=True, choices=["wan22_a14b", "hunyuan_i2v"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp37a_2d_grammar"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_run(args) if args.stage == "run" else stage_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
