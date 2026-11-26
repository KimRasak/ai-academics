#!/usr/bin/env python3
"""
Exp 48 (Zero-Shot Temporal Extrapolation [49f train -> 81f 5s test] & Simultaneous Multi-Symbol Composition
on Flagship 27B MoE `Wan2.2-I2V-A14B` Clean v2):

Tests TWO frontier generalization axes NEVER seen in training:
  1. **Zero-Shot 81-Frame (5.0s) Temporal Horizon Extrapolation** (`1.65x` longer than 49f training clips!):
     - `L1_81f_S3_NonBioAffect`
     - `L2_81f_G3_AngerVein_Pop`
     - `L3_81f_G4_SweatDrop_Slide`
     - `L4_81f_C2_Comp_HeadTurn_Vein`
  2. **Simultaneous Multi-Symbol Composition** (combining TWO 2D grammar symbols simultaneously, with/without 3D motion):
     - `MS1_Shadow_Plus_SweatDrop` (49f): Forehead gloom shadow + sliding blue comic sweat drop simultaneously
     - `MS2_HeadTurn_Shadow_Vein`  (49f): 3D head turn + forehead gloom shadow + pulsing red anger vein ALL THREE simultaneously
Evaluates `M0_Base27B` vs `M5_DualLoRA_v2` on 2 UNSEEN held-out chars (`fc_00`, `fc_02`) = 24 clips (~8 min on 8xH100!).
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
import exp31b_video_dit_anime_vs_real_diagnostic as e31b
import exp32b_wan22_causal_dit_surgery as e32b
import exp36_flagship_27b_13b_anime_stress_test as e36
import exp37a_flagship_2d_grammar_benchmark as e37a


EXP48_REGIMES = [
    (
        "L1_81f_S3_NonBioAffect",
        81,
        "The character's bright eye highlights suddenly vanish leaving hollow dark lifeless eyes while a sharp ominous dark shadow falls across their upper forehead, yet their mouth stays frozen in a calm eerie smile, 2D Japanese TV anime cel shading.",
    ),
    (
        "L2_81f_G3_AngerVein_Pop",
        81,
        "In bright flat 2D Japanese anime style, the character holds a polite closed-mouth smile as a classic stylized red cross-shaped anime anger vein symbol pops onto their upper forehead temple and pulses gently, everything else stays bright, clean, and consistent.",
    ),
    (
        "L3_81f_G4_SweatDrop_Slide",
        81,
        "In bright flat 2D Japanese anime cel shading, the character gives an awkward nervous smile as a single large stylized translucent blue comic sweat drop appears beside their temple and slides gently downward, clean crisp lineart.",
    ),
    (
        "L4_81f_C2_Comp_HeadTurn_Vein",
        81,
        "The character smoothly turns their head from front view to a three-quarter side profile holding a polite closed-mouth smile as a classic stylized red cross-shaped anime anger vein symbol pops onto their upper forehead temple and pulses gently, bright even studio lighting, clean 2D Japanese TV anime.",
    ),
    (
        "MS1_Shadow_Plus_SweatDrop",
        49,
        "In bright flat 2D Japanese anime cel shading, a translucent dark indigo anime gloom shadow with fine vertical lines falls across the character's upper forehead while simultaneously a single stylized translucent blue comic sweat drop appears beside their temple and slides gently downward, holding an awkward polite smile with bright hair and white background unchanged.",
    ),
    (
        "MS2_HeadTurn_Shadow_Vein",
        49,
        "The character smoothly turns their head from front view to a three-quarter side profile holding a calm polite smile while simultaneously a translucent dark 2D anime gloom shadow falls across their upper forehead AND a classic stylized red cross-shaped anime anger vein symbol pops onto their temple, bright unchanged hair and crisp 2D Japanese TV anime cel shading.",
    ),
]


@torch.no_grad()
def stage_run(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from peft import PeftModel

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = []
    for pid in [0, 2]:
        p = pairs[pid]
        for ri, (rname, n_frames, rprompt) in enumerate(EXP48_REGIMES):
            for mode in ["M0_Base27B", "M5_DualLoRA_v2"]:
                tasks.append((pid, Path(p["anime_path"]), p["anime_id"], rname, n_frames, rprompt, mode, 9750 + pid * 47 + ri))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    meta_dir, strips_dir, mp4_dir, kf_root = args.out_dir / "meta", args.out_dir / "strips", args.out_dir / "mp4", args.out_dir / "keyframes"
    for d in [meta_dir, strips_dir, mp4_dir, kf_root]:
        d.mkdir(parents=True, exist_ok=True)
    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[3]}_{t[6]}.json").exists()]
    if not todo:
        print(f"[exp48 s{args.shard}] all complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_dir / "lora_high_r16")
    pipe.transformer_2 = PeftModel.from_pretrained(pipe.transformer_2, args.lora_dir / "lora_low_r16")
    pipe.enable_model_cpu_offload()

    for pid, img_path, cid, rname, n_frames, rprompt, mode, seed in todo:
        if mode == "M0_Base27B":
            pipe.transformer.disable_adapter_layers()
            pipe.transformer_2.disable_adapter_layers()
        else:
            pipe.transformer.enable_adapter_layers()
            pipe.transformer_2.enable_adapter_layers()

        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=im, prompt=rprompt,
            negative_prompt="blurry, low quality, distorted, extra limbs, watermark, text",
            height=512, width=512, num_frames=n_frames, num_inference_steps=36,
            guidance_scale=3.5, guidance_scale_2=3.5, generator=gen,
        ).frames[0]

        pil_frames = e31b._to_pil_frames(frames)
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))
        st.update(e37a._grammar_diagnostics(np_frames, inp_u8))

        key = f"{pid:02d}_{rname}_{mode}"
        e36._save_single_mp4(pil_frames, mp4_dir / f"{key}.mp4", fps=16)
        e31b._save_strip(
            pil_frames, strips_dir / f"{key}.png",
            f"Exp48 {mode} ({n_frames}f) | {pid:02d}_{rname} | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=170,
        )
        kf_dir = kf_root / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        (meta_dir / f"{key}.json").write_text(json.dumps({
            "pair_idx": pid, "char_id": cid, "regime": rname, "num_frames": n_frames, "mode": mode, "seed": seed, "stats": st,
        }, indent=2), encoding="utf-8")
        print(f"[exp48 s{args.shard}] {key:46s} ({n_frames}f) | dL_g={st['global_luma_delta']:+.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


def stage_stack(args) -> int:
    strips_dir = args.out_dir / "strips"
    for pid in [0, 2]:
        for rname, _, _ in EXP48_REGIMES:
            ps = [strips_dir / f"{pid:02d}_{rname}_{m}.png" for m in ["M0_Base27B", "M5_DualLoRA_v2"]]
            ps = [p for p in ps if p.exists()]
            if len(ps) == 2:
                e31b._stack_strips(ps, args.out_dir / f"qc48_{pid:02d}_{rname}.png")
    print(f"Stacked qc48_*.png in {args.out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "stack"])
    ap.add_argument("--lora-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp44_high_lora_v2"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp48_long81f_and_multisymbol"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_run(args) if args.stage == "run" else stage_stack(args)


if __name__ == "__main__":
    raise SystemExit(main())
