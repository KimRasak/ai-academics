#!/usr/bin/env python3
"""
Exp 41B (Zero-Shot Cross-Style & Cross-Generator OOD Generalization of 14B MoE LoRA on `Wan2.2-I2V-A14B`!):

Directly answers Reviewer-2's #1 generalization question:
  "Your 14B MoE LoRA was trained ONLY on modern TV-anime portraits (`Animagine-XL-4.0`, `fc_08..fc_39`).
   Does it generalize ZERO-SHOT to completely different anime art styles (`Retro 90s`, `Chibi/SD`, `Watercolor`,
   `Monochrome B&W Manga`) AND a completely different image generator (`Illustrious-XL-v2.0`)?"

Evaluates `M0_Base27B` vs `M3_LoRA_Dual (OURS)` on 6 radically OOD portraits x 4 core 2D Anime Grammar regimes
(`S3_NonBioAffect`, `G1_HighlightOff_Only`, `G3_AngerVein_Pop`, `G4_SweatDrop_Slide` = 48 clips, 24 blind pairs),
exporting playable H.264 `.mp4`, stacked Base-vs-OURS strips, and keyframes for Blind Order-Balanced `Qwen3-VL-32B` Judging!
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
import exp36_flagship_27b_13b_anime_stress_test as e36
import exp37a_flagship_2d_grammar_benchmark as e37a


OOD_PORTRAITS = [
    ("retro90s", "/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp14b_multistyle/images/retro90s_c0__cheerful.png"),
    ("chibi_sd", "/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp14b_multistyle/images/chibi_sd_c0__cheerful.png"),
    ("watercolor", "/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp14b_multistyle/images/watercolor_c0__cheerful.png"),
    ("bw_manga", "/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp14b_multistyle/images/bw_manga_c0__cheerful.png"),
    ("illustrious_blonde", "/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp19b_xgen/illustrious_v2/illustrious_v2_c2_blonde_girl_native_cheerful.png"),
    ("illustrious_black", "/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp19b_xgen/illustrious_v2/illustrious_v2_c3_black_girl_native_cheerful.png"),
]

OOD_REGIMES = [
    (
        "S3_NonBioAffect",
        "The character's bright eye highlights suddenly vanish leaving hollow dark lifeless eyes while a sharp ominous dark shadow falls across their upper forehead, yet their mouth stays frozen in a calm eerie smile, 2D Japanese TV anime cel shading.",
    ),
    (
        "G1_HighlightOff_Only",
        "The character stays still in the exact same bright even studio lighting with unchanged bright hair, clothes, and gentle smile, while only the small white specular highlight dots inside both eyes smoothly fade away leaving flat matte dull irises, clean 2D Japanese TV anime.",
    ),
    (
        "G3_AngerVein_Pop",
        "In bright flat 2D Japanese anime style, the character holds a polite closed-mouth smile as a classic stylized red cross-shaped anime anger vein symbol pops onto their upper forehead temple and pulses gently, everything else stays bright, clean, and consistent.",
    ),
    (
        "G4_SweatDrop_Slide",
        "In bright flat 2D Japanese anime cel shading, the character gives an awkward nervous smile as a single large stylized translucent blue comic sweat drop appears beside their temple and slides gently downward, clean crisp lineart.",
    ),
]


@torch.no_grad()
def stage_run(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from peft import PeftModel

    tasks = []
    for idx_p, (style_tag, img_p) in enumerate(OOD_PORTRAITS):
        for rname, rprompt in OOD_REGIMES:
            for mode in ["M0_Base27B", "M3_LoRA_Dual"]:
                tasks.append((idx_p, style_tag, Path(img_p), rname, rprompt, mode))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    meta_dir, strips_dir, mp4_dir = args.out_dir / "meta", args.out_dir / "strips", args.out_dir / "mp4"
    for d in [meta_dir, strips_dir, mp4_dir]:
        d.mkdir(parents=True, exist_ok=True)
    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[3]}_{t[5]}.json").exists()]
    if not todo:
        print(f"[ood s{args.shard}] all complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_root / "lora_high_r16")
    pipe.transformer_2 = PeftModel.from_pretrained(pipe.transformer_2, args.lora_root / "lora_low_r16")
    pipe.enable_model_cpu_offload()

    for idx_p, style_tag, img_p, rname, rprompt, mode in todo:
        if mode == "M0_Base27B":
            pipe.transformer.disable_adapter_layers()
            pipe.transformer_2.disable_adapter_layers()
        else:
            pipe.transformer.enable_adapter_layers()
            pipe.transformer_2.enable_adapter_layers()

        im = Image.open(img_p).convert("RGB").resize((512, 512), Image.LANCZOS)
        seed = 9600 + idx_p * 53 + [r[0] for r in OOD_REGIMES].index(rname)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=im,
            prompt=rprompt,
            negative_prompt="blurry, low quality, distorted, extra limbs, watermark, text",
            height=512,
            width=512,
            num_frames=49,
            num_inference_steps=36,
            guidance_scale=3.5,
            guidance_scale_2=3.5,
            generator=gen,
        ).frames[0]

        pil_frames = e31b._to_pil_frames(frames)
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e37a._grammar_diagnostics(np_frames, inp_u8))

        key = f"{idx_p:02d}_{rname}_{mode}"
        e36._save_single_mp4(pil_frames, mp4_dir / f"{key}.mp4", fps=16)
        e31b._save_strip(
            pil_frames,
            strips_dir / f"{key}.png",
            f"{mode:14s} | {style_tag:18s} | {rname} | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6,
            thumb_h=170,
        )
        kf_dir = args.out_dir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for ki in range(6):
            pil_frames[round(ki * (T - 1) / 5)].save(kf_dir / f"k{ki}.png")

        rec = {"pair_idx": idx_p, "style": style_tag, "regime": rname, "mode": mode, "stats": st}
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[ood s{args.shard}] {key:38s} ({style_tag}) | dL_glob={st['global_luma_delta']:+.3f} dL_hair={st['hair_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.3f}", flush=True)
    return 0


def stage_summarize(args) -> int:
    for rname, _ in OOD_REGIMES:
        paths = []
        for idx_p, _ in enumerate(OOD_PORTRAITS):
            for mode in ["M0_Base27B", "M3_LoRA_Dual"]:
                p = args.out_dir / "strips" / f"{idx_p:02d}_{rname}_{mode}.png"
                if p.exists():
                    paths.append(p)
        if paths:
            e31b._stack_strips(paths, args.out_dir / f"qc41b_ood_{rname}.png")
    print("OOD visual sheets stacked!", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "summarize"])
    ap.add_argument("--lora-root", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp40_wan22_a14b_moe_lora"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp41b_cross_style_ood"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_run(args) if args.stage == "run" else stage_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
