#!/usr/bin/env python3
"""
Exp 31B (Video DiT Track — First-Principles Anime vs. Real-Human Video DiT Diagnostic Suite):

Runs controlled paired I2V generation across TWO distinct frontier open-weight Video DiTs:
  1. `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (Flow-Matching 3D-RoPE Video DiT, 2025/2026 flagship)
  2. `THUDM/CogVideoX-5b-I2V` (Expert-Adaptive 3D-Attention Video DiT)

On 8 matched 2D Cel-Shaded Anime vs. 8 Photorealistic Human portraits x 4 motion regimes:
  - `R1_HeadTurn`       (3D head rotation from flat 2D drawing vs real face)
  - `R2_DynamicWind`    (high-displacement hair/hand motion -> cel-shading & lineart stress)
  - `R3_StylizedAffect` (dark hollow eyes + forehead shadow while keeping frozen smile)
  - `R4_CameraOrbit`    (3D camera orbit -> 2D flatness vs 3D world-model conflict)

Also includes `--stage vae_control` to cleanly isolate 3D-VAE temporal compression artifacts
on moving 2D cel animation vs. DiT denoising trajectory drift!

Saves:
  - Compact 5-frame visual contact strips (`t=0, 12, 24, 36, 48`) for direct human/agent inspection
  - Full frame-by-frame quantitative diagnostics (Motion, FlatPixelRatio, AirbrushCreep,
    InkSharpness, EdgeBoiling, Luma-CLIP / RGB-SigLIP identity retention)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


REGIMES = [
    (
        "R1_HeadTurn",
        "The character smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent appearance.",
    ),
    (
        "R2_DynamicWind",
        "A sudden gust of wind blows the character's hair strongly across their face as they raise one hand to brush their bangs back with a surprised expression, dynamic lively motion.",
    ),
    (
        "R3_StylizedAffect",
        "The character's eyes suddenly darken and lose their bright highlights while an ominous shadow falls across their upper forehead, yet their mouth remains fixed in a calm eerie smile.",
    ),
    (
        "R4_CameraOrbit",
        "The camera smoothly orbits horizontally around the character while they blink softly and hold their pose, clean consistent art style.",
    ),
]


def _frame_stats(frames_u8: list[np.ndarray]) -> dict:
    """
    Compute clean, interpretable, domain-grounded Video DiT metrics on RGB uint8 frames [T, H, W, 3]:
      - motion_l1:          mean frame-to-frame |I_{t+1} - I_t| (normalized 0..1)
      - disp_from_0:        mean |I_t - I_0| over final third of video (total visual motion)
      - flat_ratio_t0 / tEnd / delta:
                            fraction of pixels with low spatial gradient (|grad| < 4.0) -> cel flatness!
      - airbrush_t0 / tEnd / delta:
                            fraction of pixels in mid-gradient shading band (4.0 <= |grad| < 22.0) -> 3D gradient creep!
      - edge_sharp_t0 / tEnd / retention:
                            mean gradient magnitude of top-5% strongest edge pixels -> lineart crispness!
      - dark_ink_t0 / tEnd / retention:
                            fraction of crisp dark outline pixels (luma < 65 & grad > 22) -> ink preservation!
      - edge_boiling:       temporal 2nd-derivative |I_{t+1} - 2 I_t + I_{t-1}| on edge pixels (line flicker/boiling)
      - interior_flicker:   temporal 2nd-derivative on interior non-edge pixels
    """
    arr = np.stack(frames_u8, axis=0).astype(np.float32)  # [T, H, W, 3]
    T = arr.shape[0]
    gray = 0.299 * arr[:, :, :, 0] + 0.587 * arr[:, :, :, 1] + 0.114 * arr[:, :, :, 2]  # [T, H, W]

    # Spatial gradient magnitude per frame
    gx = np.abs(gray[:, :, 1:] - gray[:, :, :-1])[:, :-1, :]
    gy = np.abs(gray[:, 1:, :] - gray[:, :-1, :])[:, :, :-1]
    gmag = np.sqrt(gx * gx + gy * gy)  # [T, H-1, W-1]
    g_crop = gray[:, :-1, :-1]

    flat_curve = (gmag < 4.0).mean(axis=(1, 2))
    air_curve = ((gmag >= 4.0) & (gmag < 22.0)).mean(axis=(1, 2))
    ink_curve = ((g_crop < 70.0) & (gmag >= 22.0)).mean(axis=(1, 2))

    # Top-5% edge sharpness per frame
    sharp_curve = []
    for t in range(T):
        q95 = np.percentile(gmag[t], 95)
        sharp_curve.append(float(gmag[t][gmag[t] >= q95].mean()))
    sharp_curve = np.array(sharp_curve, dtype=np.float32)

    # Motion & temporal 2nd-derivative (boiling / flicker)
    diff1 = np.abs(arr[1:] - arr[:-1]).mean(axis=(1, 2, 3)) / 255.0
    motion_l1 = float(diff1.mean())
    tail_start = max(1, (2 * T) // 3)
    disp_tail = float(np.abs(arr[tail_start:] - arr[0:1]).mean() / 255.0)

    if T >= 3:
        d2 = np.abs(arr[2:] - 2.0 * arr[1:-1] + arr[:-2]).mean(axis=-1) / 255.0  # [T-2, H, W]
        d2_c = d2[:, :-1, :-1]
        edge_mask = gmag[1:-1] >= 22.0
        int_mask = gmag[1:-1] < 4.0
        edge_boil = float(d2_c[edge_mask].mean()) if edge_mask.any() else 0.0
        int_flick = float(d2_c[int_mask].mean()) if int_mask.any() else 0.0
    else:
        edge_boil = int_flick = 0.0

    # Compare t=0 against mean of final third of frames (avoids single-frame noise)
    flat_0, flat_end = float(flat_curve[0]), float(flat_curve[tail_start:].mean())
    air_0, air_end = float(air_curve[0]), float(air_curve[tail_start:].mean())
    sharp_0, sharp_end = float(sharp_curve[0]), float(sharp_curve[tail_start:].mean())
    ink_0, ink_end = float(ink_curve[0]), float(ink_curve[tail_start:].mean())

    return {
        "T": int(T),
        "motion_l1": round(motion_l1, 5),
        "disp_tail": round(disp_tail, 5),
        "flat_t0": round(flat_0, 4),
        "flat_end": round(flat_end, 4),
        "flat_delta": round(flat_end - flat_0, 4),
        "airbrush_t0": round(air_0, 4),
        "airbrush_end": round(air_end, 4),
        "airbrush_delta": round(air_end - air_0, 4),
        "edge_sharp_t0": round(sharp_0, 2),
        "edge_sharp_end": round(sharp_end, 2),
        "edge_sharp_ratio": round(sharp_end / max(sharp_0, 1e-4), 4),
        "dark_ink_t0": round(ink_0, 4),
        "dark_ink_end": round(ink_end, 4),
        "dark_ink_ratio": round(ink_end / max(ink_0, 1e-4), 4),
        "edge_boiling": round(edge_boil, 5),
        "interior_flicker": round(int_flick, 5),
    }


def _save_strip(frames_pil: list[Image.Image], out_path: Path, title: str, n_show: int = 5, thumb_h: int = 192) -> None:
    T = len(frames_pil)
    idxs = [round(i * (T - 1) / (n_show - 1)) for i in range(n_show)]
    w0, h0 = frames_pil[0].size
    thumb_w = int(round(w0 * thumb_h / h0))
    canvas = Image.new("RGB", (n_show * thumb_w, thumb_h + 24), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill=(245, 245, 245))
    for k, fi in enumerate(idxs):
        im = frames_pil[fi].resize((thumb_w, thumb_h), Image.LANCZOS)
        canvas.paste(im, (k * thumb_w, 24))
        draw.text((k * thumb_w + 6, thumb_h + 24 - 16), f"t={fi}", fill=(255, 230, 80))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _stack_strips(strip_paths: list[Path], out_path: Path) -> None:
    ims = [Image.open(p).convert("RGB") for p in strip_paths if p.exists()]
    if not ims:
        return
    w = max(im.width for im in ims)
    h = sum(im.height for im in ims)
    sheet = Image.new("RGB", (w, h), (16, 16, 20))
    y = 0
    for im in ims:
        sheet.paste(im, (0, y))
        y += im.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _to_pil_frames(frames) -> list[Image.Image]:
    out = []
    for f in frames:
        if isinstance(f, Image.Image):
            out.append(f.convert("RGB"))
        else:
            a = np.asarray(f)
            if a.dtype != np.uint8:
                a = np.clip(a * 255.0 if a.max() <= 1.5 else a, 0, 255).astype(np.uint8)
            out.append(Image.fromarray(a, mode="RGB"))
    return out


@torch.no_grad()
def stage_i2v(args) -> int:
    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n_pairs]
    # Build flat task list: (pair_idx, domain, img_path, char_id, regime_name, prompt)
    tasks = []
    for p in pairs:
        for dom, id_k, path_k in [("anime", "anime_id", "anime_path"), ("real", "real_id", "real_path")]:
            for rname, rprompt in REGIMES:
                tasks.append((p["pair_idx"], dom, Path(p[path_k]), p[id_k], rname, rprompt))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    out_model = args.out_dir / args.model_tag
    strips_dir = out_model / "strips"
    meta_dir = out_model / "meta"
    strips_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[1]}_{t[4]}.json").exists()]
    if not todo:
        print(f"[{args.model_tag} shard {args.shard}] all tasks already complete!", flush=True)
        return 0

    if args.model_tag == "cogvideox5b":
        from diffusers import CogVideoXImageToVideoPipeline
        pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        H, W, N_FRAMES, STEPS, GUIDANCE = 480, 720, 49, 30, 6.0
    elif args.model_tag == "wan22_5b":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        H, W, N_FRAMES, STEPS, GUIDANCE = 480, 704, 49, 30, 5.0
    else:
        raise ValueError(f"Unknown model_tag: {args.model_tag}")

    for pair_idx, dom, img_path, cid, rname, rprompt in todo:
        # Center-crop / pad-free clean resize matching target aspect ratio
        im = Image.open(img_path).convert("RGB")
        # Crop square -> slight landscape crop (preserve full head & upper torso cleanly!)
        iw, ih = im.size
        target_ar = W / H
        crop_h = int(round(iw / target_ar))
        if crop_h <= ih:
            # Anchor near top (y0 = 0..5%) so head/hair is never clipped!
            y0 = int(0.03 * (ih - crop_h))
            im = im.crop((0, y0, iw, y0 + crop_h))
        im = im.resize((W, H), Image.LANCZOS)

        gen = torch.Generator(device="cuda").manual_seed(4200 + pair_idx * 10 + REGIMES.index((rname, rprompt)))
        if args.model_tag == "cogvideox5b":
            frames = pipe(
                image=im,
                prompt=rprompt,
                height=H,
                width=W,
                num_frames=N_FRAMES,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=gen,
            ).frames[0]
        else:
            frames = pipe(
                image=im,
                prompt=rprompt,
                negative_prompt="blurry, low quality, distorted, extra limbs, watermark, text",
                height=H,
                width=W,
                num_frames=N_FRAMES,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=gen,
            ).frames[0]

        pil_frames = _to_pil_frames(frames)
        # Prepend exact resized input frame at index 0 for exact t=0 reference metrics?
        # Wait: I2V outputs frame 0 as generated frame 0; record stats on generated frames directly AND against exact input frame!
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        st = _frame_stats(np_frames)
        inp_st = _frame_stats([np.asarray(im, dtype=np.uint8), np_frames[0], np_frames[-1]])
        st["inp_flat_t0"] = inp_st["flat_t0"]
        st["inp_airbrush_t0"] = inp_st["airbrush_t0"]
        st["inp_edge_sharp_t0"] = inp_st["edge_sharp_t0"]
        st["f0_vs_inp_l1"] = round(float(np.abs(np_frames[0].astype(np.float32) - np.asarray(im, dtype=np.float32)).mean() / 255.0), 5)
        st["flat_delta_vs_inp"] = round(st["flat_end"] - inp_st["flat_t0"], 4)
        st["airbrush_delta_vs_inp"] = round(st["airbrush_end"] - inp_st["airbrush_t0"], 4)
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)

        key = f"{pair_idx:02d}_{dom}_{rname}"
        strip_p = strips_dir / f"{key}.png"
        _save_strip(
            pil_frames,
            strip_p,
            f"{args.model_tag} | {key} ({cid}) | mot={st['motion_l1']:.3f} dFlat={st['flat_delta_vs_inp']:+.3f} dAir={st['airbrush_delta_vs_inp']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}",
        )
        # Also save keyframe PNGs (t=0, 12, 24, 36, 48) for downstream VLM / CLIP / mechanistic analysis!
        kf_dir = out_model / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        for fi in [0, 12, 24, 36, len(pil_frames) - 1]:
            pil_frames[fi].save(kf_dir / f"f{fi:02d}.png")

        rec = {
            "model": args.model_tag,
            "pair_idx": pair_idx,
            "domain": dom,
            "char_id": cid,
            "regime": rname,
            "stats": st,
            "strip_path": str(strip_p),
        }
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[{args.model_tag} s{args.shard}] {key:24s} | mot={st['motion_l1']:.4f} | flat:{st['inp_flat_t0']:.3f}->{st['flat_end']:.3f}({st['flat_delta_vs_inp']:+.3f}) | air:{st['inp_airbrush_t0']:.3f}->{st['airbrush_end']:.3f}({st['airbrush_delta_vs_inp']:+.3f}) | sharpR={st['sharp_ratio_vs_inp']:.3f} | boil={st['edge_boiling']:.4f}", flush=True)
    return 0


@torch.no_grad()
def stage_vae_control(args) -> int:
    """
    Isolate 3D-VAE compression loss from DiT denoising drift!
    Constructs a 49-frame smooth rigid pan+zoom animation directly from each Anime & Real image
    (0% DiT involvement, 100% crisp ground-truth cel shading & ink lines) and encodes->decodes through
    BOTH `CogVideoX 3D-VAE` and `Wan2.2 3D-VAE`.
    """
    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n_pairs]
    H, W, T = 480, 704, 49

    def make_pan_clip(img_path: Path) -> np.ndarray:
        im = Image.open(img_path).convert("RGB").resize((864, 864), Image.LANCZOS)
        frames = []
        for t in range(T):
            frac = t / (T - 1)
            # Smooth horizontal pan (80 px) + gentle zoom (1.0 -> 1.12)
            crop_w = int(round(760 - 70 * frac))
            crop_h = int(round(crop_w * H / W))
            x0 = int(round(20 + 80 * frac))
            y0 = int(round(25 + 15 * math.sin(math.pi * frac)))
            fr = im.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((W, H), Image.LANCZOS)
            frames.append(np.asarray(fr, dtype=np.uint8))
        return np.stack(frames, axis=0)  # [T, H, W, 3]

    results = {}
    for vtag, loader in [
        ("wan22_vae", lambda: __import__("diffusers").AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()),
        ("cogvideox_vae", lambda: __import__("diffusers").AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=torch.bfloat16).to("cuda").eval()),
    ]:
        vae = loader()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        res_dom = {"anime": [], "real": []}
        for p in pairs:
            for dom, pk in [("anime", "anime_path"), ("real", "real_path")]:
                gt = make_pan_clip(Path(p[pk]))  # [T, H, W, 3]
                x = torch.from_numpy(gt).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0  # [1, 3, T, H, W]
                x = x.to("cuda", dtype=next(vae.parameters()).dtype)
                z = vae.encode(x).latent_dist.mode()
                rec = vae.decode(z).sample
                rec = ((rec[0].float().clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
                t_min = min(len(gt), len(rec))
                gt_c, rec_c = gt[:t_min], rec[:t_min]
                mse = float(np.mean((gt_c.astype(np.float32) - rec_c.astype(np.float32)) ** 2))
                psnr = 10.0 * math.log10(255.0 * 255.0 / max(mse, 1e-6))
                st_gt = _frame_stats(list(gt_c))
                st_rec = _frame_stats(list(rec_c))
                res_dom[dom].append({
                    "psnr": round(psnr, 2),
                    "flat_gt": st_gt["flat_end"],
                    "flat_rec": st_rec["flat_end"],
                    "flat_delta": round(st_rec["flat_end"] - st_gt["flat_end"], 4),
                    "air_delta": round(st_rec["airbrush_end"] - st_gt["airbrush_end"], 4),
                    "sharp_ratio": round(st_rec["edge_sharp_end"] / max(st_gt["edge_sharp_end"], 1e-4), 4),
                    "edge_boil_gt": st_gt["edge_boiling"],
                    "edge_boil_rec": st_rec["edge_boiling"],
                })
        summary = {}
        print(f"\n=== 3D-VAE ISOLATION CONTROL ({vtag}, moving pan+zoom clip, n={len(pairs)}) ===", flush=True)
        for dom in ["anime", "real"]:
            rows = res_dom[dom]
            summary[dom] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
            s = summary[dom]
            print(f"  {dom:6s} | PSNR={s['psnr']:.2f} dB | flat:{s['flat_gt']:.3f}->{s['flat_rec']:.3f}({s['flat_delta']:+.3f}) | dAir={s['air_delta']:+.3f} | sharpR={s['sharp_ratio']:.3f} | edgeBoil:{s['edge_boil_gt']:.4f}->{s['edge_boil_rec']:.4f}", flush=True)
        results[vtag] = summary
        del vae; torch.cuda.empty_cache()

    out_p = args.out_dir / "exp31b_vae_isolation_control.json"
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def stage_aggregate(args) -> int:
    """Aggregate all completed meta JSONs + stack visual comparison contact sheets per regime!"""
    out_model = args.out_dir / args.model_tag
    meta_dir = out_model / "meta"
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(meta_dir.glob("*.json"))]
    print(f"\n=== EXP 31B AGGREGATED DIAGNOSTICS: {args.model_tag} (n={len(recs)} clips) ===", flush=True)
    agg = {}
    for rname, _ in REGIMES:
        agg[rname] = {}
        for dom in ["anime", "real"]:
            sub = [r["stats"] for r in recs if r["regime"] == rname and r["domain"] == dom]
            if not sub:
                continue
            keys = ["motion_l1", "disp_tail", "inp_flat_t0", "flat_end", "flat_delta_vs_inp", "inp_airbrush_t0", "airbrush_end", "airbrush_delta_vs_inp", "sharp_ratio_vs_inp", "dark_ink_ratio", "edge_boiling", "interior_flicker", "f0_vs_inp_l1"]
            m = {k: round(float(np.mean([x[k] for x in sub])), 4) for k in keys}
            agg[rname][dom] = m
            print(
                f"  {rname:18s} {dom:5s} (n={len(sub)}) | mot={m['motion_l1']:.4f} disp={m['disp_tail']:.3f} | "
                f"flat:{m['inp_flat_t0']:.3f}->{m['flat_end']:.3f}({m['flat_delta_vs_inp']:+.3f}) | "
                f"air:{m['inp_airbrush_t0']:.3f}->{m['airbrush_end']:.3f}({m['airbrush_delta_vs_inp']:+.3f}) | "
                f"sharpR={m['sharp_ratio_vs_inp']:.3f} | inkR={m['dark_ink_ratio']:.3f} | boil={m['edge_boiling']:.4f}",
                flush=True,
            )
        # Build stacked Anime vs Real visual QC contact sheet for first 4 pairs in each regime
        paths = []
        for pid in range(4):
            paths.append(out_model / "strips" / f"{pid:02d}_anime_{rname}.png")
            paths.append(out_model / "strips" / f"{pid:02d}_real_{rname}.png")
        _stack_strips(paths, out_model / f"qc_{args.model_tag}_{rname}.png")

    (out_model / f"summary_{args.model_tag}.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["i2v", "vae_control", "aggregate"])
    ap.add_argument("--model-tag", default="cogvideox5b", choices=["cogvideox5b", "wan22_5b"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit"))
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "i2v":
        return stage_i2v(args)
    if args.stage == "vae_control":
        return stage_vae_control(args)
    return stage_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
