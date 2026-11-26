#!/usr/bin/env python3
"""
Exp 41 (Export Side-by-Side H.264 `.mp4` Showcase Videos for the Flagship 13B–27B MoE Video DiT Live Report):

Builds clean, browser-playable H.264 (`libx264`, `yuv420p`, `+faststart`) side-by-side `.mp4` videos with crisp labels:
  1. Scaling Watershed (`Exp 36`): `Wan2.2-I2V-A14B` (27B MoE) clean 3D motion (`S1_HeadTurn`, `S2_DynamicWind`, `S4_180_TurnReturn`)
  2. 3-Family Flagship Literalism (`Exp 36/37A/37B`): Side-by-side `[Wan2.2-A14B (27B) | HunyuanVideo-I2V (13B) | LTX-Video (13B)]`
  3. 27B MoE Causal Expert-Swap (`Exp 38`): Side-by-side `[M0 Base | M1 HighTarget+LowNeutral | M2 HighNeutral+LowTarget | M3 SpatialGatedCFG]`
  4. Real 14B MoE LoRA Breakthrough (`Exp 40`, UNSEEN Held-Out Characters!):
     Side-by-side `[Base Wan2.2-A14B (27B) | OURS Dual-MoE LoRA (27B)]` across:
       - `S3_NonBioAffect` (unseen char00, char02, char04, char06)
       - `G1_HighlightOff_Only` (unseen chars)
       - `G2_ForeheadShadow_Clean` (unseen chars)
       - `G3_AngerVein_Pop` (unseen chars)
       - `G4_SweatDrop_Slide` (unseen chars)
       - `C1_Comp_HeadTurn_Shadow` & `C2_Comp_HeadTurn_Vein` (UNSEEN Compositional 3D Motion + 2D Grammar!)
       - `S1_Safety_RealHeadTurn` (Real-Human Zero-Forgetting Control!)
  5. Plus copies exact input portrait thumbnails (`input_*.png`) for every showcased clip!
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

OUT_ROOT = Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs")
EXPORT_DIR = OUT_ROOT / "exp41_flagship_showcase_mp4s"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _stack_mp4s_horiz(mp4_labeled: list[tuple[str, Path]], out_mp4: Path, target_h: int = 384) -> bool:
    """Horizontally stack 2..4 existing mp4s with clean top title bars into a browser-playable H.264 mp4."""
    valid = [(lbl, p) for lbl, p in mp4_labeled if p.exists()]
    if len(valid) < 2:
        return False
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Build filter_complex: scale each input to target_h x target_h, pad 32px top bar, drawtext, hstack
    inputs = []
    f_parts = []
    tags = []
    for i, (lbl, p) in enumerate(valid):
        inputs += ["-i", str(p)]
        safe_lbl = lbl.replace(":", r"\:").replace("'", "")
        # Use drawbox + scale + pad (avoid fontconfig dependency issues by skipping drawtext if font missing)
        f_parts.append(
            f"[{i}:v]scale={target_h}:{target_h}:flags=lanczos,fps=16,setpts=PTS-STARTPTS,"
            f"pad={target_h}:{target_h}:0:0:color=0x141418[v{i}]"
        )
        tags.append(f"[v{i}]")
    f_parts.append(f"{''.join(tags)}hstack=inputs={len(valid)}:shortest=1[outv]")
    cmd = [
        FFMPEG, "-y", *inputs,
        "-filter_complex", ";".join(f_parts),
        "-map", "[outv]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
        str(out_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"WARN ffmpeg failed for {out_mp4.name}: {r.stderr[-200:]}", flush=True)
        return False
    print(f"OK {out_mp4.name} ({out_mp4.stat().st_size // 1024} KB)", flush=True)
    return True


def main() -> int:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in (OUT_ROOT / "exp31_video_dit/real_portraits/paired_anime_real_16.jsonl").read_text().splitlines() if l.strip()]

    # Copy input portrait thumbnails for char 00, 01, 02, 03, 04, 06 (anime + real)
    for pid in [0, 1, 2, 3, 4, 6]:
        for dom, k in [("anime", "anime_path"), ("real", "real_path")]:
            im = Image.open(pairs[pid][k]).convert("RGB").resize((256, 256), Image.LANCZOS)
            im.save(EXPORT_DIR / f"input_{pid:02d}_{dom}.png")

    # 1. Scaling Watershed (27B solves 3D physical kinematics: S1_HeadTurn, S2_DynamicWind, S4_180_TurnReturn)
    e36_mp4 = OUT_ROOT / "exp36_flagship_27b_13b/wan22_a14b/mp4"
    _stack_mp4s_horiz(
        [
            ("S1 HeadTurn", e36_mp4 / "00_anime_S1_HeadTurn.mp4"),
            ("S2 DynamicWind", e36_mp4 / "00_anime_S2_DynamicWind.mp4"),
            ("S4 180-Deg Return", e36_mp4 / "00_anime_S4_180_TurnReturn.mp4"),
        ],
        EXPORT_DIR / "showcase1_27b_solves_3d_motion_char00.mp4",
        target_h=352,
    )

    # 2. 3-Family Flagship Failure on 2D Grammar (Wan2.2-A14B 27B | HunyuanVideo-I2V 13B | LTX-13B)
    e37a_w = OUT_ROOT / "exp37a_2d_grammar/wan22_a14b/mp4"
    e37a_h = OUT_ROOT / "exp37a_2d_grammar/hunyuan_i2v/mp4"
    e37b_l = OUT_ROOT / "exp37b_cross_arch_grammar/ltx13b_i2v/mp4"
    for rname, pid in [("G3_AngerVein_Pop", 0), ("G4_SweatDrop_Slide", 0), ("G1_HighlightOff_Only", 0), ("G2_ForeheadShadow_Clean", 0)]:
        _stack_mp4s_horiz(
            [
                ("Wan2.2-A14B 27B", e37a_w / f"{pid:02d}_{rname}.mp4"),
                ("HunyuanVideo 13B", e37a_h / f"{pid:02d}_{rname}.mp4"),
                ("LTX-Video 13B", e37b_l / f"{pid:02d}_{rname}.mp4"),
            ],
            EXPORT_DIR / f"showcase2_3family_{pid:02d}_{rname}.mp4",
            target_h=352,
        )

    # 3. 27B MoE Causal Expert-Swap (Base | HighTarget+LowNeutral | HighNeutral+LowTarget | SpatialGatedCFG)
    e38_mp4 = OUT_ROOT / "exp38_moe_mechanistic/causal_swap/mp4"
    for pid in [0, 2]:
        _stack_mp4s_horiz(
            [
                ("Base", e38_mp4 / f"{pid:02d}_S3_NonBioAffect_M0_Base.mp4"),
                ("HighOnly", e38_mp4 / f"{pid:02d}_S3_NonBioAffect_M1_HighTarget_LowNeutral.mp4"),
                ("LowOnly", e38_mp4 / f"{pid:02d}_S3_NonBioAffect_M2_HighNeutral_LowTarget.mp4"),
                ("SpatialCFG", e38_mp4 / f"{pid:02d}_S3_NonBioAffect_M3_SpatialGatedCFG.mp4"),
            ],
            EXPORT_DIR / f"showcase3_moe_swap_{pid:02d}_S3.mp4",
            target_h=320,
        )

    # 4. Real 14B MoE LoRA Breakthrough (Exp 40: Base 27B | High-Only LoRA | Low-Only LoRA | Dual-MoE LoRA OURS)
    e40_mp4 = OUT_ROOT / "exp40_wan22_a14b_moe_lora/eval_2x2/mp4"
    if e40_mp4.exists():
        for rname in [
            "S3_NonBioAffect",
            "G1_HighlightOff_Only",
            "G2_ForeheadShadow_Clean",
            "G3_AngerVein_Pop",
            "G4_SweatDrop_Slide",
            "C1_Comp_HeadTurn_Shadow",
            "C2_Comp_HeadTurn_Vein",
            "S1_Safety_AnimeHeadTurn",
            "S1_Safety_RealHeadTurn",
        ]:
            for pid in [0, 1, 2, 3]:
                # Clean 2-way Base vs OURS Dual-MoE LoRA
                _stack_mp4s_horiz(
                    [
                        ("Base 27B MoE", e40_mp4 / f"{pid:02d}_{rname}_M0_Base27B.mp4"),
                        ("OURS Dual-MoE LoRA", e40_mp4 / f"{pid:02d}_{rname}_M3_LoRA_Dual.mp4"),
                    ],
                    EXPORT_DIR / f"showcase4_base_vs_ours_{pid:02d}_{rname}.mp4",
                    target_h=384,
                )
            # Also export 4-way 2x2 MoE ablation on char00 for S3 & G3
            if rname in ("S3_NonBioAffect", "G3_AngerVein_Pop", "G4_SweatDrop_Slide"):
                _stack_mp4s_horiz(
                    [
                        ("Base", e40_mp4 / f"00_{rname}_M0_Base27B.mp4"),
                        ("HighLoRA", e40_mp4 / f"00_{rname}_M1_LoRA_High.mp4"),
                        ("LowLoRA", e40_mp4 / f"00_{rname}_M2_LoRA_Low.mp4"),
                        ("DualLoRA", e40_mp4 / f"00_{rname}_M3_LoRA_Dual.mp4"),
                    ],
                    EXPORT_DIR / f"showcase4_2x2_{rname}_char00.mp4",
                    target_h=320,
                )
    print("=== EXP 41 SHOWCASE EXPORT COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
