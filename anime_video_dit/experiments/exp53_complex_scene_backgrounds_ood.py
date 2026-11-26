#!/usr/bin/env python3
"""
Exp 53: Zero-Shot Complex Textured-Background & Multi-Lighting Anime Scene OOD Stress Test (`Wan2.2-27B`)

Directly answers Reviewer-2's robustness question:
  "Your synthetic training clips used clean white studio backgrounds. Does your High-Noise MoE LoRA
   preserve rich, complex, multi-lighting real-world anime backgrounds (classroom, sunset rooftop,
   neon night street, sakura park, cozy library, rainy station) WITHOUT smudging or corrupting them?"

Stage 1 (`--stage gen_scenes`, ~15s on 1 GPU):
  Generates 6 diverse complex-scene anime portraits using public open-weight `Animagine-XL-4.0`
  (cross-generator OOD + complex textured backgrounds + varied warm/cool/neon/overcast lighting!).
Stage 2 (`--stage eval`, 6 scenes x 4 regimes x 3 modes = 72 clips -> ~9 min on 8xH100!):
  Evaluates `M0_Base27B`, `M4_HighLoRA_v2` (High-Noise ONLY, Low-Noise 100% frozen!), and `M5_DualLoRA_v2`
  across `S3_NonBioAffect`, `G2_ForeheadShadow_Clean`, `G3_AngerVein_Pop`, `G4_SweatDrop_Slide`!
  NOTE: Prompts are adapted naturally to preserve the complex background ("keeping the detailed scene
  background, lighting, hair, clothes, and smile unchanged") — ZERO mention of white studio background!
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

SCENES = [
    (
        "s0_classroom",
        "masterpiece, best quality, 2D Japanese TV anime screenshot, 1girl, upper body portrait, brown bob hair, navy school blazer, gentle calm smile, bright open eyes, inside a detailed sunlit Japanese high school classroom with green chalkboard, wooden desks, and blue sky windows behind her, crisp cel shading",
        8101,
    ),
    (
        "s1_sunset_rooftop",
        "masterpiece, best quality, 2D Japanese TV anime screenshot, 1boy, upper body portrait, messy dark blue hair, school uniform, gentle calm smile, bright open eyes, standing on a golden hour orange sunset school rooftop with chain-link fence, glowing orange clouds, and warm rim light, crisp cel shading",
        8102,
    ),
    (
        "s2_neon_night",
        "masterpiece, best quality, 2D Japanese TV anime screenshot, 1girl, upper body portrait, silver twin-tails, casual denim jacket, polite smile, bright open eyes, standing on a vibrant Tokyo night street with colorful pink and cyan neon signs and blurred bokeh city lights behind her, crisp anime cel shading",
        8103,
    ),
    (
        "s3_sakura_park",
        "masterpiece, best quality, 2D Japanese TV anime screenshot, 1girl, upper body portrait, emerald green hair, cream cardigan, gentle smile, bright open eyes, outdoor park under blooming pink cherry blossom trees with sunlit green leaves and floating sakura petals, crisp cel shading",
        8104,
    ),
    (
        "s4_cozy_library",
        "masterpiece, best quality, 2D Japanese TV anime screenshot, 1boy, upper body portrait, auburn hair, knit sweater, calm polite smile, bright open eyes, inside a warm wooden library packed with colorful bookshelves and glowing golden desk lamps behind him, crisp 2D anime cel shading",
        8105,
    ),
    (
        "s5_rainy_station",
        "masterpiece, best quality, 2D Japanese TV anime screenshot, 1girl, upper body portrait, lavender short hair, yellow raincoat, awkward polite smile, bright open eyes, outdoor Japanese train station platform in soft overcast blue rain with vending machines, roof canopy, and tracks behind her, crisp cel shading",
        8106,
    ),
]

SCENE_REGIMES = [
    (
        "S3_NonBioAffect",
        "The character's bright eye highlights suddenly vanish leaving hollow matte lifeless eyes while a sharp ominous dark 2D anime gloom shadow falls strictly across their upper forehead, yet their mouth stays frozen in a calm smile and the entire detailed scene background, lighting, hair, and clothes remain unchanged, 2D Japanese TV anime cel shading.",
    ),
    (
        "G2_ForeheadShadow_Clean",
        "A stylized translucent slate-blue 2D anime gloom shadow band with fine vertical ink lines appears strictly across the character's upper forehead and brow while the character keeps their exact gentle smile, hair, clothes, and the entire detailed scene background and lighting completely unchanged, 2D Japanese TV anime.",
    ),
    (
        "G3_AngerVein_Pop",
        "In 2D Japanese TV anime style, the character holds a polite closed-mouth smile as a classic stylized red cross-shaped anime anger vein symbol pops onto their upper forehead temple and pulses gently, while the character and detailed scene background stay completely consistent.",
    ),
    (
        "G4_SweatDrop_Slide",
        "In 2D Japanese TV anime cel shading, the character gives an awkward nervous smile as a single large stylized translucent blue comic sweat drop appears beside their temple and slides gently downward, clean crisp lineart, detailed scene background completely unchanged.",
    ),
]


@torch.no_grad()
def stage_gen_scenes(args) -> int:
    from diffusers import StableDiffusionXLPipeline

    sc_dir = args.out_dir / "scenes"
    sc_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in SCENES if not (sc_dir / f"{s[0]}.png").exists()]
    if todo:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "/gluster_hdd_cv/user/jinzili/weights/animagine-xl-4.0",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cuda")
        neg = "lowres, bad anatomy, blurry, worst quality, low quality, text, watermark, 3d, photorealistic, monochrome, closed eyes"
        for tag, prompt, seed in todo:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            im = pipe(prompt=prompt, negative_prompt=neg, height=1024, width=1024, num_inference_steps=28, guidance_scale=5.5, generator=gen).images[0]
            im.resize((512, 512), Image.LANCZOS).save(sc_dir / f"{tag}.png")
            print(f"[exp53 gen] Saved {tag}.png", flush=True)

    # Save 2x3 contact sheet for visual QC
    ims = [Image.open(sc_dir / f"{s[0]}.png").convert("RGB").resize((256, 256), Image.LANCZOS) for s in SCENES]
    sheet = Image.new("RGB", (768, 512), (20, 20, 20))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % 3) * 256, (i // 3) * 256))
    sheet.save(args.out_dir / "qc53_input_scenes_grid.png")
    print(f"[exp53 gen] Saved contact sheet -> {args.out_dir / 'qc53_input_scenes_grid.png'}", flush=True)
    return 0


@torch.no_grad()
def stage_eval(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from peft import PeftModel

    sc_dir = args.out_dir / "scenes"
    for sub in ("meta", "strips", "mp4", "keyframes"):
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    modes = ["M0_Base27B", "M4_HighLoRA_v2", "M5_DualLoRA_v2"]
    tasks = []
    for idx, (tag, _, _) in enumerate(SCENES):
        img_path = sc_dir / f"{tag}.png"
        for ri, (rname, rprompt) in enumerate(SCENE_REGIMES):
            seed = 9800 + idx * 59 + ri
            for m in modes:
                tasks.append((idx, tag, img_path, rname, rprompt, m, seed))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (args.out_dir / "meta" / f"{t[0]:02d}_{t[3]}_{t[5]}.json").exists()]
    if not todo:
        print(f"[exp53 eval s{args.shard}] all complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_high_dir)
    pipe.transformer_2 = PeftModel.from_pretrained(pipe.transformer_2, args.lora_low_dir)
    pipe.enable_model_cpu_offload()

    for idx, tag, img_path, rname, rprompt, m, seed in todo:
        if m == "M0_Base27B":
            pipe.transformer.disable_adapter_layers()
            pipe.transformer_2.disable_adapter_layers()
        elif m == "M4_HighLoRA_v2":
            pipe.transformer.enable_adapter_layers()
            pipe.transformer_2.disable_adapter_layers()
        else:
            pipe.transformer.enable_adapter_layers()
            pipe.transformer_2.enable_adapter_layers()

        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=im, prompt=rprompt,
            negative_prompt="blurry, low quality, distorted, extra limbs, watermark, text",
            height=512, width=512, num_frames=49, num_inference_steps=36,
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

        # Extra background-border preservation metric (outer 12% border L1 error vs input!)
        h, w = inp_u8.shape[:2]
        bmask = np.zeros((h, w), dtype=bool)
        bw, bh = int(0.12 * w), int(0.12 * h)
        bmask[:bh, :] = True; bmask[:, :bw] = True; bmask[:, -bw:] = True
        end_f = np_frames[-1].astype(np.float32) / 255.0
        inp_f = inp_u8.astype(np.float32) / 255.0
        st["bg_border_l1"] = round(float(np.abs(end_f[bmask] - inp_f[bmask]).mean()), 4)

        key = f"{idx:02d}_{rname}_{m}"
        e36._save_single_mp4(pil_frames, args.out_dir / "mp4" / f"{key}.mp4", fps=16)
        e31b._save_strip(
            pil_frames, args.out_dir / "strips" / f"{key}.png",
            f"{m} | {tag}_{rname} | dL(g={st['global_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) bgL1={st['bg_border_l1']:.3f} shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=160,
        )
        kf_dir = args.out_dir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        rec = {"pair_idx": idx, "idx": idx, "domain": tag, "char_id": tag, "regime": rname, "mode": m, "seed": seed, "stats": st}
        (args.out_dir / "meta" / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[exp53 s{args.shard}] {key:42s} ({tag:16s}) | bgL1={st['bg_border_l1']:.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


def stage_summary(args) -> int:
    modes = ["M0_Base27B", "M4_HighLoRA_v2", "M5_DualLoRA_v2"]
    summary = {}
    for rname, _ in SCENE_REGIMES:
        summary[rname] = {}
        for m in modes:
            recs = []
            for idx in range(len(SCENES)):
                p = args.out_dir / "meta" / f"{idx:02d}_{rname}_{m}.json"
                if p.exists():
                    recs.append(json.loads(p.read_text(encoding="utf-8"))["stats"])
            if recs:
                summary[rname][m] = {
                    "n": len(recs),
                    "dL_glob": round(float(np.mean([r["global_luma_delta"] for r in recs])), 4),
                    "dL_up": round(float(np.mean([r["upper_face_luma_delta"] for r in recs])), 4),
                    "dL_low": round(float(np.mean([r["lower_face_luma_delta"] for r in recs])), 4),
                    "bg_border_l1": round(float(np.mean([r["bg_border_l1"] for r in recs])), 4),
                    "shR": round(float(np.mean([r["sharp_ratio_vs_inp"] for r in recs])), 4),
                }
        ps = []
        for idx in range(len(SCENES)):
            for m in modes:
                p = args.out_dir / "strips" / f"{idx:02d}_{rname}_{m}.png"
                if p.exists():
                    ps.append(p)
        if ps:
            e31b._stack_strips(ps, args.out_dir / f"qc53_scenes_{rname}.png")
    (args.out_dir / "summary_exp53_scenes.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[exp53] Saved complex-scene summary & QC sheets -> {args.out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen_scenes", "eval", "summary"])
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp53_complex_scene_ood"))
    ap.add_argument("--lora-high-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp44_high_lora_v2/lora_high_r16"))
    ap.add_argument("--lora-low-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp44_high_lora_v2/lora_low_r16"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "gen_scenes":
        return stage_gen_scenes(args)
    if args.stage == "eval":
        return stage_eval(args)
    return stage_summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
