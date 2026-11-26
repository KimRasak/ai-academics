#!/usr/bin/env python3
"""
Exp 32B (Video DiT Track — Causal Zero-Training Video DiT Surgery inside `Wan2.2-TI2V-5B`!):

Directly tests causal sufficiency of the TWO coupled internal Video DiT mechanisms discovered in Exp 32A:
  - Mechanism #1: OOD CFG Velocity Overshoot (+40% to +81% higher `||v_cond - v_uncond||` on 2D Anime
    -> drives latent solarization, abstract color-blob meltdown, and 2x-3x violent over-rotation).
  - Mechanism #2: Frame-0 Anchor Attention Starvation (-26% to -30% drop in `Attn(f_t -> f_0)` across
    Early/Mid blocks L0..L19, collapsing to 4.4%-4.8% on far frames -> identity swaps & jump-cuts).

Compares 5 controlled inference methods at IDENTICAL seeds (1.0x inference cost, ZERO training,
ZERO external ControlNets/IP-Adapters) on 8 Anime characters x 2 high-motion regimes (`R1_HeadTurn`, `R2_DynamicWind`):
  - `M0_Base`:        Unmodified `Wan2.2-TI2V-5B` (`g = 5.0`)
  - `M1_LowCFG`:      Naive lower guidance (`g = 2.5`) -- fair baseline checking motion vs stability trade-off!
  - `M2_FlowCap`:     Mechanism #1 Fix ONLY: Orthogonal-Projected + Real-Manifold Relative-Norm Capped CFG
                      (damps parallel saturation overshoot + caps `||dv_eff|| / ||v_uncond|| <= rho_cap`,
                      calibrated to Real-Human relative velocity budget from Exp 32A!)
  - `M3_AnchorAttn`:  Mechanism #2 Fix ONLY: Distance-Scaled Frame-0 Anchor Logit Restoration inside
                      `WanAttnProcessor` (`attn1` blocks L0..L19, restoring `Attn(f_t -> f_0)` on far frames!)
  - `M4_OURS_Dual`:   Combined Mechanism #1 + #2 (`M2_FlowCap` + `M3_AnchorAttn`)!

Generates side-by-side 5-row stacked visual comparison strips (`M0`..`M4` for every anime character & regime)
+ full quantitative Pareto table (Motion vs. Luma-CLIP Structural Identity, Edge Sharpness, Saturation Explosion, Edge Boiling).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp31b_video_dit_anime_vs_real_diagnostic as e31b
import exp32a_wan22_dit_mechanistic_probe as e32a


class AnchorBoostWanAttnProcessor:
    """
    Zero-overhead Flash/SDPA-compatible Frame-0 Anchor-Attention Restoration for `Wan2.2` 3D Self-Attention (`attn1`).
    Adds a smooth temporal-distance-scaled additive attention bias ONLY from generated-frame queries (`t >= 1`)
    to Frame-0 clean anchor keys (`k_t == 0`) via `F.scaled_dot_product_attention(..., attn_mask=bias)`!
    """

    def __init__(self, orig_proc, layer_idx: int, ctrl: dict):
        self.orig_proc = orig_proc
        self.layer_idx = layer_idx
        self.ctrl = ctrl
        self._cached_mask_key = None
        self._cached_mask = None

    def _get_bias_mask(self, seq_len: int, T_lat: int, beta: float, device, dtype):
        key = (seq_len, T_lat, round(beta, 4), device, dtype)
        if self._cached_mask_key == key:
            return self._cached_mask
        N_sp = seq_len // T_lat
        # Bias shape [1, 1, seq_len, seq_len]: broadcast cleanly in SDPA!
        # Wait: even more memory-efficient -> only allocate [1, 1, T_lat*N_sp, T_lat*N_sp] (~22MB for 3328x3328 bf16!)
        bias = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
        for t in range(1, T_lat):
            # Smooth temporal-distance schedule: small at t=1, full beta at far frames t>=4
            w_t = beta * (1.0 - math.exp(-t / 3.0))
            bias[:, :, t * N_sp : (t + 1) * N_sp, 0:N_sp] = w_t
        self._cached_mask_key = key
        self._cached_mask = bias
        return bias

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        beta = float(self.ctrl.get("beta", 0.0))
        l_min = int(self.ctrl.get("l_min", 0))
        l_max = int(self.ctrl.get("l_max", 20))
        active_step = bool(self.ctrl.get("step_active", True))

        if (
            encoder_hidden_states is not None
            or beta <= 0.0
            or not active_step
            or not (l_min <= self.layer_idx < l_max)
        ):
            return self.orig_proc(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)

        T_lat = int(self.ctrl["T_lat"])
        seq_len = hidden_states.shape[1]
        if seq_len % T_lat != 0:
            return self.orig_proc(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)

        from diffusers.models.transformers.transformer_wan import _get_qkv_projections

        query, key, value = _get_qkv_projections(attn, hidden_states, None)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = e32a._apply_rotary_emb(query, *rotary_emb)
            key = e32a._apply_rotary_emb(key, *rotary_emb)

        # Transpose to [B, H, L, D] for native PyTorch SDPA with additive float bias
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        bias = self._get_bias_mask(seq_len, T_lat, beta, q.device, q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).flatten(2, 3).type_as(hidden_states)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


def _apply_manifold_flow_cfg(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    guidance_scale: float,
    eta_parallel: float = 0.15,
    rho_cap: float = 0.068,
) -> torch.Tensor:
    """
    Manifold-Calibrated Projected & Relative-Norm-Capped Flow CFG:
      1. Decompose `dv = v_cond - v_uncond` per latent frame `t >= 1` into parallel (to `v_uncond`,
         which drives latent color/saturation solarization!) and orthogonal (semantic motion) components.
      2. Damp parallel component by `eta_parallel` (preserving clean motion while killing neon oversaturation).
      3. Softly cap each latent frame's relative update norm `||dv_proj|| / ||v_uncond|| <= rho_cap`
         (calibrated to Real-Human relative velocity budget from Exp 32A: ~0.06–0.07!).
    """
    dv = (v_cond - v_uncond).float()
    vu = v_uncond.float()
    out_dv = dv.clone()

    # Apply per latent frame t = 1..T_lat-1 (leave t=0 anchor untouched)
    T_lat = dv.shape[2]
    for t in range(1, T_lat):
        d_t = dv[:, :, t]
        u_t = vu[:, :, t]
        u_norm = torch.linalg.vector_norm(u_t) + 1e-6
        u_hat = u_t / u_norm
        proj_coeff = torch.sum(d_t * u_hat)
        d_par = proj_coeff * u_hat
        d_orth = d_t - d_par
        d_clean = d_orth + eta_parallel * d_par
        # Soft relative norm cap matching Real-Human velocity budget
        d_norm = torch.linalg.vector_norm(d_clean) + 1e-6
        max_norm = rho_cap * u_norm
        if d_norm > max_norm:
            d_clean = d_clean * (max_norm / d_norm)
        out_dv[:, :, t] = d_clean

    return (vu + guidance_scale * out_dv).to(v_cond.dtype)


@torch.no_grad()
def _run_custom_wan22_i2v(
    pipe,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    guidance_scale: float,
    use_flow_cap: bool,
    eta_parallel: float,
    rho_cap: float,
    attn_ctrl: dict,
    attn_beta: float,
    attn_step_max: int,
    H: int = 512,
    W: int = 512,
    num_frames: int = 49,
    num_inference_steps: int = 30,
) -> list[Image.Image]:
    """Clean, exact replication of `WanImageToVideoPipeline.__call__` with custom Flow-CFG & Anchor-Attn control."""
    device = pipe._execution_device
    dtype = pipe.transformer.dtype

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        device=device,
        dtype=dtype,
    )

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    image_tensor = pipe.video_processor.preprocess(image, height=H, width=W).to(device, dtype=torch.float32)
    num_channels_latents = pipe.vae.config.z_dim
    gen = torch.Generator(device=device).manual_seed(seed)

    latents, condition, first_frame_mask = pipe.prepare_latents(
        image=image_tensor,
        batch_size=1,
        num_channels_latents=num_channels_latents,
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
    attn_ctrl["beta"] = attn_beta

    for i, t in enumerate(timesteps):
        attn_ctrl["step_active"] = (i < attn_step_max)
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

        if use_flow_cap:
            noise_pred = _apply_manifold_flow_cfg(
                v_cond, v_uncond, guidance_scale, eta_parallel=eta_parallel, rho_cap=rho_cap
            )
        else:
            noise_pred = v_uncond + guidance_scale * (v_cond - v_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
    latents = latents.to(pipe.vae.dtype)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(device, pipe.vae.dtype)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(device, pipe.vae.dtype)
    latents = latents / latents_std + latents_mean
    video = pipe.vae.decode(latents, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
    return e31b._to_pil_frames(frames)


METHODS = [
    # (name, guidance, use_flow_cap, eta_par, rho_cap, attn_beta, attn_step_max)
    ("M0_Base",       5.0, False, 1.00, 9.99,  0.00, 0),
    ("M1_LowCFG",     2.5, False, 1.00, 9.99,  0.00, 0),
    ("M2_FlowCap",    5.0, True,  0.15, 0.068, 0.00, 0),
    ("M3_AnchorAttn", 5.0, False, 1.00, 9.99,  1.20, 24),
    ("M4_OURS_Dual",  5.0, True,  0.15, 0.068, 1.20, 24),
]


def _extra_artifact_stats(frames_u8: list[np.ndarray], inp_u8: np.ndarray) -> dict:
    """Measure extreme saturation / color solarization overshoot (captures neon yellow/cyan explosions!)."""
    def sat_p95(im_u8):
        a = im_u8.astype(np.float32) / 255.0
        mx = a.max(axis=2); mn = a.min(axis=2)
        s = np.where(mx > 0.02, (mx - mn) / np.maximum(mx, 0.02), 0.0)
        return float(np.percentile(s, 95)), float(s.mean())
    s95_0, sm_0 = sat_p95(inp_u8)
    s95_max = max(sat_p95(f)[0] for f in frames_u8[4:])
    sm_max = max(sat_p95(f)[1] for f in frames_u8[4:])
    # Max frame-to-frame jump (catches discontinuous jump-cuts like 02_anime R1 t=36!)
    arr = np.stack(frames_u8, axis=0).astype(np.float32) / 255.0
    step_l1 = np.abs(arr[1:] - arr[:-1]).mean(axis=(1, 2, 3))
    jump_ratio = float(step_l1.max() / max(float(step_l1.mean()), 1e-5))
    return {
        "sat_overshoot": round(sm_max - sm_0, 4),
        "sat95_overshoot": round(s95_max - s95_0, 4),
        "max_step_l1": round(float(step_l1.max()), 4),
        "jump_spike_ratio": round(jump_ratio, 2),
    }


@torch.no_grad()
def stage_run(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n_pairs]
    tasks = []
    for p in pairs:
        for rname, rprompt in e32a.REGIMES_32:
            for m_spec in METHODS:
                tasks.append((p["pair_idx"], Path(p["anime_path"]), p["anime_id"], rname, rprompt, m_spec))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    meta_dir = args.out_dir / "meta"
    strips_dir = args.out_dir / "strips"
    meta_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)

    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[3]}_{t[5][0]}.json").exists()]
    if not todo:
        print(f"[exp32b shard {args.shard}] all done!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    attn_ctrl = {"beta": 0.0, "l_min": 0, "l_max": 20, "T_lat": 13, "step_active": True}
    for li, blk in enumerate(pipe.transformer.blocks):
        blk.attn1.set_processor(AnchorBoostWanAttnProcessor(blk.attn1.processor, li, attn_ctrl))

    for pair_idx, img_path, cid, rname, rprompt, m_spec in todo:
        mname, g_scale, use_cap, eta_p, rho_c, beta, step_max = m_spec
        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        seed = 5200 + pair_idx * 10 + (0 if rname == "R1_HeadTurn" else 1)

        pil_frames = _run_custom_wan22_i2v(
            pipe, im, rprompt,
            negative_prompt="blurry, low quality, distorted, extra limbs, watermark, text",
            seed=seed,
            guidance_scale=g_scale,
            use_flow_cap=use_cap,
            eta_parallel=eta_p,
            rho_cap=rho_c,
            attn_ctrl=attn_ctrl,
            attn_beta=beta,
            attn_step_max=step_max,
        )
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st["flat_delta_vs_inp"] = round(st["flat_end"] - inp_st["flat_t0"], 4)
        st.update(_extra_artifact_stats(np_frames, inp_u8))

        key = f"{pair_idx:02d}_{rname}_{mname}"
        strip_p = strips_dir / f"{key}.png"
        e31b._save_strip(
            pil_frames, strip_p,
            f"{mname:13s} | {pair_idx:02d}_{rname} ({cid}) | mot={st['motion_l1']:.3f} shR={st['sharp_ratio_vs_inp']:.2f} boil={st['edge_boiling']:.3f} satOv={st['sat_overshoot']:+.3f} jmp={st['jump_spike_ratio']:.1f}",
        )
        kf_dir = args.out_dir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        for fi in [0, 12, 24, 36, len(pil_frames) - 1]:
            pil_frames[fi].save(kf_dir / f"f{fi:02d}.png")

        rec = {"pair_idx": pair_idx, "char_id": cid, "regime": rname, "method": mname, "stats": st}
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[exp32b s{args.shard}] {key:32s} | mot={st['motion_l1']:.4f} disp={st['disp_tail']:.3f} | "
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
    print(f"\n=== EXP 32B CAUSAL VIDEO DiT SURGERY COMPARISON (Wan2.2-TI2V-5B 512x512, n={len(recs)} clips) ===", flush=True)
    for rname, _ in e32a.REGIMES_32:
        summary[rname] = {}
        print(f"\n--- {rname} ---", flush=True)
        for m_spec in METHODS:
            mname = m_spec[0]
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
                f"  {mname:14s} | mot={m['motion_l1']:.4f} disp={m['disp_tail']:.3f} | "
                f"LumaID={m['luma_clip_mean']:.4f}(min={m['luma_clip_min']:.4f}) RGBID={m['rgb_clip_mean']:.4f} | "
                f"shR={m['sharp_ratio']:.3f} boil={m['edge_boiling']:.4f} satOv={m['sat_overshoot']:+.3f} jmp={m['jump_spike']:.2f}",
                flush=True,
            )

        # Build side-by-side 5-method comparison sheets for characters 00, 01, 02, 03!
        for pid in range(4):
            paths = [args.out_dir / "strips" / f"{pid:02d}_{rname}_{ms[0]}.png" for ms in METHODS]
            e31b._stack_strips(paths, args.out_dir / f"cmp_{rname}_char{pid:02d}.png")

    (args.out_dir / "exp32b_surgery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved exp32b_surgery_summary.json and side-by-side char00..03 comparison sheets!", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "eval"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp32b_wan22_surgery"))
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_run(args) if args.stage == "run" else stage_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
