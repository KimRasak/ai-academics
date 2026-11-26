#!/usr/bin/env python3
"""
Export browser-compatible H.264 MP4 videos (`libx264`, `yuv420p`, 16 fps) for the HTML report showcase,
using the EXACT same seeds and pipelines from Exp 34A/34B/35 so every video matches our reported results bit-for-bit!
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp32a_wan22_dit_mechanistic_probe as e32a
import exp32b_wan22_causal_dit_surgery as e32b
import exp33_wan22_decoupled_flow_anchor_ablation as e33
import exp34a_cogvideox_cross_arch_probe_and_fix as e34a
import exp34b_wan22_early_winner_and_layer_localization as e34b
import exp35_wan22_real_safety_and_unseen_prompts as e35


def _get_ffmpeg() -> str:
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _encode_dir_to_mp4(tmp_dir: Path, out_mp4: Path, fps: int = 16) -> None:
    cmd = [
        _get_ffmpeg(), "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "f_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    for f in tmp_dir.glob("*.png"):
        f.unlink()
    tmp_dir.rmdir()


def _save_side_by_side_mp4(frames_l: list[Image.Image], frames_r: list[Image.Image], lbl_l: str, lbl_r: str, out_mp4: Path, fps: int = 16) -> None:
    """Create crisp side-by-side H.264 MP4 (`Base` on left, `Ours` on right) with clean header bar."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_mp4.parent / f".tmp_{out_mp4.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    w, h = frames_l[0].size
    bar = 28
    for i, (fl, fr) in enumerate(zip(frames_l, frames_r)):
        c = Image.new("RGB", (w * 2 + 4, h + bar), (15, 23, 42))
        d = ImageDraw.Draw(c)
        d.text((12, 6), lbl_l, fill=(248, 113, 113))
        d.text((w + 16, 6), lbl_r, fill=(52, 211, 153))
        c.paste(fl, (0, bar))
        c.paste(fr, (w + 4, bar))
        cw, ch = c.size
        if cw % 2 == 1 or ch % 2 == 1:
            c = c.crop((0, 0, cw - (cw % 2), ch - (ch % 2)))
        c.save(tmp_dir / f"f_{i:03d}.png")
    _encode_dir_to_mp4(tmp_dir, out_mp4, fps=fps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp35b_showcase_mp4s"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    reg_map = dict(e32a.REGIMES_32 + e35.UNSEEN_PROMPTS)

    # Exact curated showcase jobs: (id, model, pair_idx, dom, rname, seed_mode)
    jobs = [
        # Wan2.2 DynamicWind (4 unseen chars 04..07 from Exp 34B)
        ("wan_wind_p04", "wan", 4, "anime", "R2_DynamicWind", "exp34b"),
        ("wan_wind_p05", "wan", 5, "anime", "R2_DynamicWind", "exp34b"),
        ("wan_wind_p06", "wan", 6, "anime", "R2_DynamicWind", "exp34b"),
        ("wan_wind_p07", "wan", 7, "anime", "R2_DynamicWind", "exp34b"),
        # Wan2.2 Unseen Prompt TurnWaveWink (00, 01, 03 from Exp 35)
        ("wan_wave_p00", "wan", 0, "anime", "U3_TurnWaveWink", "exp35"),
        ("wan_wave_p01", "wan", 1, "anime", "U3_TurnWaveWink", "exp35"),
        ("wan_wave_p03", "wan", 3, "anime", "U3_TurnWaveWink", "exp35"),
        # Wan2.2 HeadTurn (00, 01 anime + 00 real safety control)
        ("wan_turn_p00", "wan", 0, "anime", "R1_HeadTurn", "exp34b"),
        ("wan_turn_p01", "wan", 1, "anime", "R1_HeadTurn", "exp34b"),
        ("wan_turn_real00", "wan", 0, "real", "R1_HeadTurn", "exp35"),
        # Honest boundary: HairTossLaugh p03
        ("wan_laugh_p03_limit", "wan", 3, "anime", "U2_HairTossLaugh", "exp35"),
        # Cross-Arch CogVideoX-5b-I2V (p00 3D hand removal + p03 blue foam removal!)
        ("cog_wind_p00", "cog", 0, "anime", "R2_DynamicWind", "exp34a"),
        ("cog_wind_p03", "cog", 3, "anime", "R2_DynamicWind", "exp34a"),
    ]

    my_jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    # Fast-path: encode any already-rendered .tmp_* frames without re-running diffusion!
    for j in my_jobs:
        jid = j[0]
        td = args.out_dir / f".tmp_{jid}"
        mp = args.out_dir / f"{jid}.mp4"
        if not mp.exists() and td.exists() and len(list(td.glob("f_*.png"))) >= 40:
            _encode_dir_to_mp4(td, mp, fps=16)
            print(f"[mp4 shard {args.shard}] encoded cached {jid}.mp4", flush=True)

    wan_jobs = [j for j in my_jobs if j[1] == "wan" and not (args.out_dir / f"{j[0]}.mp4").exists()]
    cog_jobs = [j for j in my_jobs if j[1] == "cog" and not (args.out_dir / f"{j[0]}.mp4").exists()]

    if wan_jobs:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", vae=vae, torch_dtype=torch.bfloat16).to("cuda")
        attn_ctrl = {"beta": 0.0, "l_min": 0, "l_max": 30, "T_lat": 13, "step_active": False}
        for li, blk in enumerate(pipe.transformer.blocks):
            blk.attn1.set_processor(e32b.AnchorBoostWanAttnProcessor(blk.attn1.processor, li, attn_ctrl))

        spec_base, spec_ours = e34b.SCALE16_METHODS[0], e34b.SCALE16_METHODS[1]
        for jid, _, pid, dom, rname, smode in wan_jobs:
            pk = "anime_path" if dom == "anime" else "real_path"
            im = Image.open(pairs[pid][pk]).convert("RGB").resize((512, 512), Image.LANCZOS)
            rprompt = reg_map[rname]
            if smode == "exp34b":
                seed = 5200 + pid * 10 + (0 if rname == "R1_HeadTurn" else 1)
            else:
                seed = 7300 + pid * 17 + abs(hash(rname)) % 97
            fl = e33._run_decoupled_wan22(pipe, im, rprompt, seed, spec_base, attn_ctrl)
            fr = e33._run_decoupled_wan22(pipe, im, rprompt, seed, spec_ours, attn_ctrl)
            _save_side_by_side_mp4(fl, fr, f"Base (Wan2.2-5B) | {jid}", "OURS (Zero-Train 1.0x Surgery)", args.out_dir / f"{jid}.mp4")
            print(f"[mp4 shard {args.shard}] saved {jid}.mp4", flush=True)
        del pipe, vae; torch.cuda.empty_cache()

    if cog_jobs:
        from diffusers import CogVideoXImageToVideoPipeline
        pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16).to("cuda")
        pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
        state = {"beta": 0.0, "step_active": False, "probe_active": False, "l_min": 0, "l_max": 42, "T_lat": 13, "probe_recs": {}}
        for li, blk in enumerate(pipe.transformer.transformer_blocks):
            blk.attn1.set_processor(e34a.CogProbeAndBoostAttnProcessor(blk.attn1.processor, li, state))

        for jid, _, pid, _dom, _rname, _smode in cog_jobs:
            im = Image.open(pairs[pid]["anime_path"]).convert("RGB")
            iw, ih = im.size
            crop_h = int(round(iw / (720.0 / 480.0)))
            y0 = int(0.03 * (ih - crop_h))
            im = im.crop((0, y0, iw, y0 + crop_h)).resize((720, 480), Image.LANCZOS)
            seed = 4200 + pid * 10 + 1
            fl, _ = e34a._run_cog_custom(pipe, im, e34a.WIND_PROMPT, seed, state, beta=0.0, s_min=0, s_max=0, use_orth_cfg=False)
            fr, _ = e34a._run_cog_custom(pipe, im, e34a.WIND_PROMPT, seed, state, beta=1.2, s_min=0, s_max=15, use_orth_cfg=False)
            # Resize slightly to 480x320 per side so side-by-side is crisp 964x348 and loads fast!
            fl = [f.resize((480, 320), Image.LANCZOS) for f in fl]
            fr = [f.resize((480, 320), Image.LANCZOS) for f in fr]
            _save_side_by_side_mp4(fl, fr, f"Base (CogVideoX-5B) | {jid}", "OURS (EarlyAnchor 1.0x)", args.out_dir / f"{jid}.mp4")
            print(f"[mp4 shard {args.shard}] saved {jid}.mp4", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
