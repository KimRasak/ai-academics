#!/usr/bin/env python3
"""
Exp 34A (Video DiT Track — Cross-Architecture Verification on `THUDM/CogVideoX-5b-I2V`!):

Verifies whether BOTH internal Video DiT mechanisms discovered on `Wan2.2-TI2V-5B` (Flow-Matching + 3D-RoPE)
reproduce on a COMPLETELY DIFFERENT Video DiT architecture (`CogVideoX-5b-I2V`: Expert-Adaptive 3D Attention + DDIM):

  Part 1 (`--stage probe`, n=6 matched Anime vs. Real pairs on `R2_DynamicWind`):
    - Measures step-by-step CFG velocity/noise diff norm `||eps_cond - eps_uncond||` & relative norm ratio
      on Anime vs. Real!
    - Measures exact 3D Self-Attention Frame-0 Anchor Mass `Attn(f_t -> f_0)` inside `CogVideoXAttnProcessor2_0`
      across ALL 42 transformer blocks!

  Part 2 (`--stage surgery`, n=8 Anime characters on `R2_DynamicWind`, where `CogVideoX` infamously
    erupts photorealistic 3D flesh hands onto 2D anime faces!):
    - Compares `G0_Base` vs `G1_EarlyAnchor` (`s=0..15`, boosting Frame-0 visual keys `[text_len : text_len+N_sp]`)
      vs `G2_OrthCFG` (damping parallel CFG overshoot `eta_par=0.0`) vs `G3_OURS_EarlyAnchor_OrthCFG`!
    - Generates stacked 4-method side-by-side visual sheets to check directly whether photorealistic 3D hand
      intrusion & dark-ink collapse vanish!
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
import exp32b_wan22_causal_dit_surgery as e32b


WIND_PROMPT = (
    "A sudden gust of wind blows the character's hair strongly across their face as they raise one hand "
    "to brush their bangs back with a surprised expression, dynamic lively motion."
)


class CogProbeAndBoostAttnProcessor:
    """
    Drop-in replacement for `CogVideoXAttnProcessor2_0` supporting:
      1. Non-invasive exact Frame-0 Anchor Attention Mass probing (`probe_active=True`)
      2. Distance-scaled Frame-0 Anchor Logit Restoration (`beta > 0` & `step_active=True`)
    """

    def __init__(self, orig_proc, layer_idx: int, state: dict):
        self.orig_proc = orig_proc
        self.layer_idx = layer_idx
        self.state = state
        self._mask_key = None
        self._mask = None

    def _get_bias(self, seq_len: int, text_len: int, T_lat: int, beta: float, device, dtype):
        key = (seq_len, text_len, T_lat, round(beta, 4), device, dtype)
        if self._mask_key == key:
            return self._mask
        vis_len = seq_len - text_len
        N_sp = vis_len // T_lat
        bias = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
        f0_start, f0_end = text_len, text_len + N_sp
        for t in range(1, T_lat):
            w_t = beta * (1.0 - math.exp(-t / 3.0))
            q_start = text_len + t * N_sp
            q_end = q_start + N_sp
            bias[:, :, q_start:q_end, f0_start:f0_end] = w_t
        self._mask_key = key
        self._mask = bias
        return bias

    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None, image_rotary_emb=None):
        beta = float(self.state.get("beta", 0.0))
        step_active = bool(self.state.get("step_active", False))
        probe_active = bool(self.state.get("probe_active", False))
        l_min = int(self.state.get("l_min", 0))
        l_max = int(self.state.get("l_max", 42))
        do_boost = (beta > 0.0 and step_active and (l_min <= self.layer_idx < l_max))

        if not do_boost and not probe_active:
            return self.orig_proc(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        from diffusers.models.embeddings import apply_rotary_emb

        text_seq_length = encoder_hidden_states.size(1)
        hs = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = hs.shape

        query = attn.to_q(hs)
        key = attn.to_k(hs)
        value = attn.to_v(hs)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        T_lat = int(self.state["T_lat"])
        vis_len = sequence_length - text_seq_length
        N_sp = vis_len // T_lat

        if probe_active and (vis_len % T_lat == 0):
            with torch.no_grad():
                # Take conditional sample (index 1 in [uncond, cond] batch of size 2!)
                b_idx = 1 if batch_size >= 2 else 0
                q_v = query[b_idx, :, text_seq_length:].float()  # [H, T*N, D]
                k_all = key[b_idx].float()                        # [H, text+T*N, D]
                stride = max(1, N_sp // 32)
                q_sub = q_v.view(attn.heads, T_lat, N_sp, head_dim)[:, :, ::stride, :]
                n_sub = q_sub.shape[2]
                scale = 1.0 / math.sqrt(head_dim)
                logits = torch.bmm(q_sub.reshape(attn.heads, -1, head_dim), k_all.transpose(1, 2)) * scale
                probs = torch.softmax(logits, dim=-1)  # [H, T*n_sub, text + T*N]
                # Fraction of visual-token attention going to Frame 0 visual keys (normalized over visual keys)
                p_vis = probs[:, :, text_seq_length:]
                p_f0 = p_vis[:, :, :N_sp].sum(dim=-1) / (p_vis.sum(dim=-1) + 1e-6)  # [H, T*n_sub]
                p_f0_t = p_f0.view(attn.heads, T_lat, n_sub).mean(dim=(0, 2))[1:].cpu().numpy()
                st_i = str(self.state["step_idx"])
                self.state["probe_recs"].setdefault(st_i, {})[str(self.layer_idx)] = {
                    "f0_mean": round(float(p_f0_t.mean()), 5),
                    "f0_by_t": [round(float(x), 4) for x in p_f0_t.tolist()],
                }

        attn_bias = self._get_bias(sequence_length, text_seq_length, T_lat, beta, query.device, query.dtype) if do_boost else None
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(hs.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        encoder_hidden_states_out, hidden_states_out = out.split([text_seq_length, vis_len], dim=1)
        return hidden_states_out, encoder_hidden_states_out


@torch.no_grad()
def _run_cog_custom(
    pipe,
    image: Image.Image,
    prompt: str,
    seed: int,
    state: dict,
    beta: float,
    s_min: int,
    s_max: int,
    use_orth_cfg: bool,
    collect_vel: bool = False,
    probe_steps: set[int] | None = None,
    H: int = 480,
    W: int = 720,
    num_frames: int = 49,
    num_inference_steps: int = 30,
    guidance_scale: float = 6.0,
) -> tuple[list[Image.Image], list[dict]]:
    device = pipe._execution_device
    dtype = pipe.transformer.dtype

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt="",
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=226,
        device=device,
        dtype=dtype,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    image_tensor = pipe.video_processor.preprocess(image, height=H, width=W).to(device, dtype=dtype)
    latent_channels = pipe.transformer.config.in_channels // 2
    gen = torch.Generator(device=device).manual_seed(seed)

    latents, image_latents = pipe.prepare_latents(
        image=image_tensor,
        batch_size=1,
        num_channels_latents=latent_channels,
        num_frames=num_frames,
        height=H,
        width=W,
        dtype=dtype,
        device=device,
        generator=gen,
        latents=None,
    )

    image_rotary_emb = (
        pipe._prepare_rotary_positional_embeddings(H, W, latents.size(1), device)
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )
    ofs_emb = None if pipe.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

    state["T_lat"] = latents.shape[1]
    state["beta"] = beta
    state["probe_recs"] = {}
    vel_recs = []

    old_pred = None
    for i, t in enumerate(timesteps):
        state["step_idx"] = i
        state["step_active"] = (s_min <= i < s_max)
        state["probe_active"] = bool(probe_steps and (i in probe_steps))

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        latent_image_input = torch.cat([image_latents] * 2)
        latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2).to(dtype)
        timestep = t.expand(latent_model_input.shape[0])

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            ofs=ofs_emb,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0].float()

        eps_u, eps_c = noise_pred.chunk(2)  # [1, T_lat, C, H_l, W_l]
        if collect_vel:
            # Measure on frames t=1..T_lat-1
            du = eps_u[0, 1:]
            dc = eps_c[0, 1:]
            dv = dc - du
            vel_recs.append({
                "step": i,
                "u_norm": round(float(torch.linalg.vector_norm(du).item()), 2),
                "cfg_diff_norm": round(float(torch.linalg.vector_norm(dv).item()), 2),
                "cfg_rel_ratio": round(float(torch.linalg.vector_norm(dv).item() / max(float(torch.linalg.vector_norm(du).item()), 1e-5)), 4),
            })

        # CogVideoX dynamic CFG schedule (exact default!)
        g_t = 1.0 + guidance_scale * ((1.0 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2.0)
        if use_orth_cfg:
            # Orthogonal-Projected CFG per latent frame t >= 1 (zero parallel saturation overshoot!)
            dv = eps_c - eps_u
            out_dv = dv.clone()
            for ti in range(1, dv.shape[1]):
                d_t = dv[:, ti]
                u_t = eps_u[:, ti]
                u_hat = u_t / (torch.linalg.vector_norm(u_t) + 1e-6)
                d_par = torch.sum(d_t * u_hat) * u_hat
                d_orth = d_t - d_par
                out_dv[:, ti] = d_orth
            noise_pred = eps_u + g_t * out_dv
        else:
            noise_pred = eps_u + g_t * (eps_c - eps_u)

        step_out = pipe.scheduler.step(noise_pred, t, latents.float(), return_dict=False)
        latents = step_out[0].to(dtype)

    video = pipe.decode_latents(latents)
    frames = pipe.video_processor.postprocess_video(video=video, output_type="pil")[0]
    return e31b._to_pil_frames(frames), vel_recs


@torch.no_grad()
def stage_probe(args) -> int:
    from diffusers import CogVideoXImageToVideoPipeline

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n_pairs]
    pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16).to("cuda")
    pipe.vae.enable_slicing(); pipe.vae.enable_tiling()

    state = {"beta": 0.0, "step_active": False, "probe_active": False, "l_min": 0, "l_max": 42, "T_lat": 13, "probe_recs": {}}
    for li, blk in enumerate(pipe.transformer.transformer_blocks):
        blk.attn1.set_processor(CogProbeAndBoostAttnProcessor(blk.attn1.processor, li, state))

    recs = {"anime": [], "real": []}
    for p in pairs:
        for dom, pk in [("anime", "anime_path"), ("real", "real_path")]:
            im = Image.open(p[pk]).convert("RGB").resize((720, 480), Image.LANCZOS)
            frames, vels = _run_cog_custom(
                pipe, im, WIND_PROMPT, seed=6200 + p["pair_idx"], state=state,
                beta=0.0, s_min=0, s_max=0, use_orth_cfg=False, collect_vel=True, probe_steps={5, 10, 20}
            )
            np_f = [np.asarray(f, dtype=np.uint8) for f in frames]
            st = e31b._frame_stats(np_f)
            f0_s10 = float(np.mean([v["f0_mean"] for v in state["probe_recs"].get("10", {}).values()]))
            f0_s20 = float(np.mean([v["f0_mean"] for v in state["probe_recs"].get("20", {}).values()]))
            cfg_e = float(np.mean([x["cfg_diff_norm"] for x in vels[:10]]))
            cfg_m = float(np.mean([x["cfg_diff_norm"] for x in vels[10:20]]))
            rel_e = float(np.mean([x["cfg_rel_ratio"] for x in vels[:10]]))
            recs[dom].append({"pair_idx": p["pair_idx"], "cfg_e": cfg_e, "cfg_m": cfg_m, "rel_e": rel_e, "f0_s10": f0_s10, "f0_s20": f0_s20, "mot": st["motion_l1"], "boil": st["edge_boiling"]})
            print(f"[CogProbe] p{p['pair_idx']:02d} {dom:5s} | cfgE={cfg_e:.1f} cfgM={cfg_m:.1f} relE={rel_e:.3f} | f0Attn(s10={f0_s10:.4f}, s20={f0_s20:.4f}) | mot={st['motion_l1']:.4f}", flush=True)

    summary = {}
    print("\n=== EXP 34A CROSS-ARCHITECTURE PROBE SUMMARY: CogVideoX-5b-I2V (R2_DynamicWind) ===", flush=True)
    for dom in ["anime", "real"]:
        summary[dom] = {k: round(float(np.mean([r[k] for r in recs[dom]])), 4) for k in ["cfg_e", "cfg_m", "rel_e", "f0_s10", "f0_s20", "mot", "boil"]}
        s = summary[dom]
        print(f"  {dom:5s} (n={len(recs[dom])}) | cfgE={s['cfg_e']:.1f} cfgM={s['cfg_m']:.1f} relE={s['rel_e']:.4f} | f0Attn(s10={s['f0_s10']:.4f}, s20={s['f0_s20']:.4f}) | mot={s['mot']:.4f} boil={s['boil']:.4f}", flush=True)
    (args.out_dir / "exp34a_cog_probe_summary.json").write_text(json.dumps({"summary": summary, "recs": recs}, indent=2), encoding="utf-8")
    return 0


COG_METHODS = [
    ("G0_Base",       0.0,  0,  0, False),
    ("G1_EarlyAnchor", 1.2, 0, 15, False),
    ("G2_OrthCFG",    0.0,  0,  0, True),
    ("G3_OURS_Both",  1.2,  0, 15, True),
]


@torch.no_grad()
def stage_surgery(args) -> int:
    from diffusers import CogVideoXImageToVideoPipeline
    from transformers import CLIPModel, CLIPProcessor

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n_pairs]
    tasks = [(p["pair_idx"], Path(p["anime_path"]), p["anime_id"], spec) for p in pairs for spec in COG_METHODS]
    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]

    meta_dir = args.out_dir / "surgery_meta"
    strips_dir = args.out_dir / "surgery_strips"
    meta_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)

    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[3][0]}.json").exists()]
    if todo:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16).to("cuda")
        pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
        state = {"beta": 0.0, "step_active": False, "probe_active": False, "l_min": 0, "l_max": 42, "T_lat": 13, "probe_recs": {}}
        for li, blk in enumerate(pipe.transformer.transformer_blocks):
            blk.attn1.set_processor(CogProbeAndBoostAttnProcessor(blk.attn1.processor, li, state))

        for pair_idx, img_path, cid, spec in todo:
            mname, beta, s_min, s_max, use_orth = spec
            im = Image.open(img_path).convert("RGB")
            iw, ih = im.size
            crop_h = int(round(iw / (720.0 / 480.0)))
            y0 = int(0.03 * (ih - crop_h))
            im = im.crop((0, y0, iw, y0 + crop_h)).resize((720, 480), Image.LANCZOS)

            frames, _ = _run_cog_custom(
                pipe, im, WIND_PROMPT, seed=4200 + pair_idx * 10 + 1,
                state=state, beta=beta, s_min=s_min, s_max=s_max, use_orth_cfg=use_orth,
            )
            np_f = [np.asarray(f, dtype=np.uint8) for f in frames]
            inp_u8 = np.asarray(im, dtype=np.uint8)
            st = e31b._frame_stats(np_f)
            inp_st = e31b._frame_stats([inp_u8, np_f[0], np_f[-1]])
            st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
            st.update(e32b._extra_artifact_stats(np_f, inp_u8))

            key = f"{pair_idx:02d}_{mname}"
            e31b._save_strip(
                frames, strips_dir / f"{key}.png",
                f"CogX {mname:14s} | p{pair_idx:02d} ({cid}) | mot={st['motion_l1']:.3f} shR={st['sharp_ratio_vs_inp']:.2f} inkR={st['dark_ink_ratio']:.2f} boil={st['edge_boiling']:.3f}",
            )
            kf_dir = args.out_dir / "surgery_kf" / key
            kf_dir.mkdir(parents=True, exist_ok=True)
            im.save(kf_dir / "input.png")
            for fi in [0, 12, 24, 36, len(frames) - 1]:
                frames[fi].save(kf_dir / f"f{fi:02d}.png")
            (meta_dir / f"{key}.json").write_text(json.dumps({"pair_idx": pair_idx, "char_id": cid, "method": mname, "stats": st}, indent=2), encoding="utf-8")
            print(f"[CogSurg s{args.shard}] {key:22s} | mot={st['motion_l1']:.4f} shR={st['sharp_ratio_vs_inp']:.3f} inkR={st['dark_ink_ratio']:.3f} boil={st['edge_boiling']:.4f}", flush=True)
        del pipe; torch.cuda.empty_cache()

    if args.stage == "surgery_eval":
        cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.bfloat16).to("cuda").eval()
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(meta_dir.glob("*.json"))]
        for r in recs:
            key = f"{r['pair_idx']:02d}_{r['method']}"
            kf_dir = args.out_dir / "surgery_kf" / key
            ims = [ImageOps.autocontrast(ImageOps.grayscale(Image.open(kf_dir / fn).convert("RGB")), cutoff=1).convert("RGB") for fn in ["input.png", "f12.png", "f24.png", "f36.png", "f48.png"]]
            pv = cp(images=ims, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)
            f = cm.get_image_features(pixel_values=pv)
            if not isinstance(f, torch.Tensor): f = f.pooler_output
            e = F.normalize(f.float(), dim=-1)
            r["luma_clip_mean"] = round(float((e[1:] @ e[0]).mean().item()), 4)
        print("\n=== EXP 34A CROSS-ARCH SURGERY SUMMARY: CogVideoX-5b-I2V (R2_DynamicWind) ===", flush=True)
        out_s = {}
        for mname, *_ in COG_METHODS:
            sub = [r for r in recs if r["method"] == mname]
            if not sub: continue
            out_s[mname] = {
                "n": len(sub),
                "mot": round(float(np.mean([r["stats"]["motion_l1"] for r in sub])), 4),
                "luma_id": round(float(np.mean([r["luma_clip_mean"] for r in sub])), 4),
                "shR": round(float(np.mean([r["stats"]["sharp_ratio_vs_inp"] for r in sub])), 4),
                "inkR": round(float(np.mean([r["stats"]["dark_ink_ratio"] for r in sub])), 4),
                "boil": round(float(np.mean([r["stats"]["edge_boiling"] for r in sub])), 4),
            }
            s = out_s[mname]
            print(f"  {mname:16s} | mot={s['mot']:.4f} | LumaID={s['luma_id']:.4f} | shR={s['shR']:.3f} inkR={s['inkR']:.3f} boil={s['boil']:.4f}", flush=True)
        for pid in range(4):
            e31b._stack_strips([strips_dir / f"{pid:02d}_{m[0]}.png" for m in COG_METHODS], args.out_dir / f"cmp_cog_wind_char{pid:02d}.png")
        (args.out_dir / "exp34a_cog_surgery_summary.json").write_text(json.dumps(out_s, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["probe", "surgery", "surgery_eval"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp34a_cogvideox_xarch"))
    ap.add_argument("--n-pairs", type=int, default=6)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "probe":
        return stage_probe(args)
    return stage_surgery(args)


if __name__ == "__main__":
    raise SystemExit(main())
