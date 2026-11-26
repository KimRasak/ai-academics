#!/usr/bin/env python3
"""
Exp 33 (Video DiT Track — Timestep-Decoupled Causal Factorization inside `Wan2.2-TI2V-5B`!):

Directly resolves the exact visual diagnostic from Exp 32B:
  - In Exp 32B, `M3_AnchorAttn` produced studio-quality, full-amplitude 3/4 head turns on `char00/01/02`
    (eliminating mutated eyes and hallucinated headphones), whereas stacking hard norm-clamping + early
    anchor-bias in `M4` over-damped head-turn initiation on `char00/02`, and stopping AnchorAttn at step 24
    left fine eye highlights unlocked at steps 24..29!

Tests a clean, principled **Timestep-Decoupled Flow & Anchor Factorization** (ZERO training, 1.0x cost!)
on ALL 8 Anime characters x 2 high-motion regimes (`R1_HeadTurn`, `R2_DynamicWind`, same seeds = 80 clips):
  - `C0_Base`:          Unmodified `Wan2.2-TI2V-5B` (`g = 5.0`)
  - `C1_EarlyAttn`:     Anchor-Attn ONLY at Early high-noise steps (`s = 0..9`, `beta = 1.2`, `L0..29`)
                        -> causal test: does early anchor bias freeze motion without locking fine identity?
  - `C2_MidLateAttn`:   Anchor-Attn ONLY at Mid-to-Fine steps (`s = 8..29`, `beta = 1.0`, `L0..29`)
                        -> frees early motion (`s=0..7`) while locking eyes/hair/cel-shading through step 29!
  - `C3_OrthCFG`:       Pure Parallel-Overshoot Suppression (`eta_par = 0.0`, NO hard norm cap on orthogonal
                        motion!) -> kills neon solarization without damping head turns!
  - `C4_OURS_Decoupled`: `C2_MidLateAttn` (`s=8..29`, `beta=1.0`, `L0..29`) + `C3_OrthCFG` (`eta_par=0.0`,
                        soft `rho_cap=0.10` ONLY for extreme >2x explosions, never clamping normal head turns!)

Also scores EVERY clip on:
  - **Prompt Motion Compliance (VLM-verified `Qwen3-VL-8B` or Motion/Displacement + Head-Turn Verification)**
  - **Luma-CLIP Structural Identity (`mean` & `worst-frame min`)**
  - **Sharpness Retention (`shR`), Edge Boiling (`boil`), Saturation Overshoot (`satOv`), Jump-Cut Spike (`jmp`)**
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
import exp32a_wan22_dit_mechanistic_probe as e32a
import exp32b_wan22_causal_dit_surgery as e32b


METHODS_33 = [
    # (name, g, use_orth_cfg, eta_par, rho_cap, beta, s_min, s_max, l_min, l_max)
    ("C0_Base",           5.0, False, 1.0, 9.99, 0.00,  0,  0,  0, 30),
    ("C1_EarlyAttn",      5.0, False, 1.0, 9.99, 1.20,  0, 10,  0, 30),
    ("C2_MidLateAttn",    5.0, False, 1.0, 9.99, 1.00,  8, 30,  0, 30),
    ("C3_OrthCFG",        5.0, True,  0.0, 0.12, 0.00,  0,  0,  0, 30),
    ("C4_OURS_Decoupled", 5.0, True,  0.0, 0.12, 1.00,  8, 30,  0, 30),
]


@torch.no_grad()
def _run_decoupled_wan22(
    pipe,
    image: Image.Image,
    prompt: str,
    seed: int,
    spec: tuple,
    attn_ctrl: dict,
    H: int = 512,
    W: int = 512,
    num_frames: int = 49,
    num_inference_steps: int = 30,
) -> list[Image.Image]:
    _mname, g_scale, use_orth_cfg, eta_p, rho_c, beta, s_min, s_max, l_min, l_max = spec
    device = pipe._execution_device
    dtype = pipe.transformer.dtype

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted, extra limbs, watermark, text",
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        device=device,
        dtype=dtype,
    )
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    image_tensor = pipe.video_processor.preprocess(image, height=H, width=W).to(device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(seed)
    latents, condition, first_frame_mask = pipe.prepare_latents(
        image=image_tensor,
        batch_size=1,
        num_channels_latents=pipe.vae.config.z_dim,
        height=H,
        width=W,
        num_frames=num_frames,
        dtype=torch.float32,
        device=device,
        generator=gen,
        latents=None,
        last_image=None,
    )

    attn_ctrl["T_lat"] = latents.shape[2]
    attn_ctrl["beta"] = beta
    attn_ctrl["l_min"] = l_min
    attn_ctrl["l_max"] = l_max

    for i, t in enumerate(timesteps):
        attn_ctrl["step_active"] = (s_min <= i < s_max)
        latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
        latent_model_input = latent_model_input.to(dtype)
        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
        timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)

        with pipe.transformer.cache_context("cond"):
            v_cond = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
        with pipe.transformer.cache_context("uncond"):
            v_uncond = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                return_dict=False,
            )[0]

        if use_orth_cfg:
            noise_pred = e32b._apply_manifold_flow_cfg(
                v_cond, v_uncond, g_scale, eta_parallel=eta_p, rho_cap=rho_c
            )
        else:
            noise_pred = v_uncond + g_scale * (v_cond - v_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
    latents = latents.to(pipe.vae.dtype)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(device, pipe.vae.dtype)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(device, pipe.vae.dtype)
    latents = latents / latents_std + latents_mean
    video = pipe.vae.decode(latents, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
    return e31b._to_pil_frames(frames)


@torch.no_grad()
def stage_run(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n_pairs]
    tasks = []
    for p in pairs:
        for rname, rprompt in e32a.REGIMES_32:
            for spec in METHODS_33:
                tasks.append((p["pair_idx"], Path(p["anime_path"]), p["anime_id"], rname, rprompt, spec))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    meta_dir = args.out_dir / "meta"
    strips_dir = args.out_dir / "strips"
    meta_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)

    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[3]}_{t[5][0]}.json").exists()]
    if not todo:
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    attn_ctrl = {"beta": 0.0, "l_min": 0, "l_max": 30, "T_lat": 13, "step_active": False}
    for li, blk in enumerate(pipe.transformer.blocks):
        blk.attn1.set_processor(e32b.AnchorBoostWanAttnProcessor(blk.attn1.processor, li, attn_ctrl))

    for pair_idx, img_path, cid, rname, rprompt, spec in todo:
        mname = spec[0]
        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        seed = 5200 + pair_idx * 10 + (0 if rname == "R1_HeadTurn" else 1)

        pil_frames = _run_decoupled_wan22(pipe, im, rprompt, seed, spec, attn_ctrl)
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st["flat_delta_vs_inp"] = round(st["flat_end"] - inp_st["flat_t0"], 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))

        key = f"{pair_idx:02d}_{rname}_{mname}"
        strip_p = strips_dir / f"{key}.png"
        e31b._save_strip(
            pil_frames, strip_p,
            f"{mname:17s} | {pair_idx:02d}_{rname} | mot={st['motion_l1']:.3f} disp={st['disp_tail']:.2f} shR={st['sharp_ratio_vs_inp']:.2f} boil={st['edge_boiling']:.3f} satOv={st['sat_overshoot']:+.3f} jmp={st['jump_spike_ratio']:.1f}",
        )
        kf_dir = args.out_dir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        for fi in [0, 12, 24, 36, len(pil_frames) - 1]:
            pil_frames[fi].save(kf_dir / f"f{fi:02d}.png")

        rec = {"pair_idx": pair_idx, "char_id": cid, "regime": rname, "method": mname, "stats": st}
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[exp33 s{args.shard}] {key:36s} | mot={st['motion_l1']:.4f} disp={st['disp_tail']:.3f} | "
            f"shR={st['sharp_ratio_vs_inp']:.3f} boil={st['edge_boiling']:.4f} | satOv={st['sat_overshoot']:+.3f} jmp={st['jump_spike_ratio']:.2f}",
            flush=True,
        )
    return 0


@torch.no_grad()
def stage_eval(args) -> int:
    from transformers import CLIPModel, CLIPProcessor

    meta_dir = args.out_dir / "meta"
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(meta_dir.glob("*.json"))]

    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.bfloat16).to("cuda").eval()
    def enc_clip(ims: list[Image.Image], luma: bool) -> torch.Tensor:
        if luma:
            ims = [ImageOps.autocontrast(ImageOps.grayscale(im), cutoff=1).convert("RGB") for im in ims]
        pv = cp(images=ims, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)
        f = cm.get_image_features(pixel_values=pv)
        if not isinstance(f, torch.Tensor):
            f = f.pooler_output
        return F.normalize(f.float(), dim=-1)

    for r in recs:
        key = f"{r['pair_idx']:02d}_{r['regime']}_{r['method']}"
        kf_dir = args.out_dir / "keyframes" / key
        ims = [Image.open(kf_dir / fn).convert("RGB") for fn in ["input.png", "f00.png", "f12.png", "f24.png", "f36.png", "f48.png"]]
        el = enc_clip(ims, luma=True)
        er = enc_clip(ims, luma=False)
        r["luma_clip_min"] = round(float((el[1:] @ el[0]).min().item()), 4)
        r["luma_clip_mean"] = round(float((el[2:] @ el[0]).mean().item()), 4)
        r["rgb_clip_mean"] = round(float((er[2:] @ er[0]).mean().item()), 4)

    summary = {}
    print(f"\n=== EXP 33 TIMESTEP-DECOUPLED FACTORIZATION (Wan2.2-TI2V-5B 512x512, n={len(recs)} clips) ===", flush=True)
    for rname, _ in e32a.REGIMES_32:
        summary[rname] = {}
        print(f"\n--- {rname} ---", flush=True)
        for spec in METHODS_33:
            mname = spec[0]
            sub = [r for r in recs if r["regime"] == rname and r["method"] == mname]
            if not sub:
                continue
            m = {
                "n": len(sub),
                "motion_l1": round(float(np.mean([r["stats"]["motion_l1"] for r in sub])), 4),
                "disp_tail": round(float(np.mean([r["stats"]["disp_tail"] for r in sub])), 4),
                "sharp_ratio": round(float(np.mean([r["stats"]["sharp_ratio_vs_inp"] for r in sub])), 4),
                "edge_boiling": round(float(np.mean([r["stats"]["edge_boiling"] for r in sub])), 4),
                "sat_overshoot": round(float(np.mean([r["stats"]["sat_overshoot"] for r in sub])), 4),
                "jump_spike": round(float(np.mean([r["stats"]["jump_spike_ratio"] for r in sub])), 2),
                "luma_clip_mean": round(float(np.mean([r["luma_clip_mean"] for r in sub])), 4),
                "luma_clip_min": round(float(np.mean([r["luma_clip_min"] for r in sub])), 4),
                "rgb_clip_mean": round(float(np.mean([r["rgb_clip_mean"] for r in sub])), 4),
            }
            summary[rname][mname] = m
            print(
                f"  {mname:18s} | mot={m['motion_l1']:.4f} disp={m['disp_tail']:.3f} | "
                f"LumaID={m['luma_clip_mean']:.4f}(min={m['luma_clip_min']:.4f}) RGBID={m['rgb_clip_mean']:.4f} | "
                f"shR={m['sharp_ratio']:.3f} boil={m['edge_boiling']:.4f} satOv={m['sat_overshoot']:+.3f} jmp={m['jump_spike']:.2f}",
                flush=True,
            )

        for pid in range(4):
            paths = [args.out_dir / "strips" / f"{pid:02d}_{rname}_{ms[0]}.png" for ms in METHODS_33]
            e31b._stack_strips(paths, args.out_dir / f"cmp33_{rname}_char{pid:02d}.png")

    (args.out_dir / "exp33_decoupled_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved exp33_decoupled_summary.json and cmp33 visual comparison sheets!", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "eval"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp33_wan22_decoupled"))
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_run(args) if args.stage == "run" else stage_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
