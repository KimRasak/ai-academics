#!/usr/bin/env python3
"""
Exp 32A (Video DiT Track — Opening the Black Box of `Wan2.2-TI2V-5B` on Framing-Matched 2D Anime vs. Real Humans!):

100% Framing-Deconfounded (`512x512` square — ZERO portrait crop mismatch between Anime and Real!)
mechanistic instrumentation of `Wan2.2-TI2V-5B`'s internal 3D Flow-Matching DiT dynamics across
8 matched 2D Anime vs. 8 matched Real Human portraits x 2 high-motion regimes (`R1_HeadTurn`, `R2_DynamicWind`):

Measures INSIDE the Video DiT at every denoising step s = 0..29 and across all 30 DiT blocks:
  1. **Flow-Matching Velocity & CFG Geometry (every step s = 0..29)**:
     - `||v_uncond||`, `||v_cond||`, **`||v_cond - v_uncond||` (CFG velocity push norm)**,
       `cos(v_cond, v_uncond)`, and per-frame latent velocity norm across latent frames `f_1..f_12`.
  2. **Exact 3D Self-Attention Frame-0 Anchor Mass (`Attn(f_t -> f_0)`)**:
     - Sampled cleanly via custom non-invasive `WanAttnProcessor` at steps `s in {0, 5, 10, 15, 20, 25, 29}`
       across ALL 30 DiT blocks (`blocks 0..29`), measuring for every generated latent frame `f_1..f_12`:
         * `mass_to_f0(layer, step, f_t)`: exact softmax attention mass allocated to clean Frame-0 anchor tokens!
         * `mass_to_self(layer, step, f_t)`: attention mass to same-frame tokens!
         * `mass_to_other(layer, step, f_t)`: attention mass to other generated frames!
         * Split separately for `cond` pass vs `uncond` pass!
  3. **Outcome Quality & Structural Identity (`t = 0, 12, 24, 36, 48`)**:
     - Motion (`motion_l1`, `disp_tail`), Edge Sharpness (`sharpR`), Dark Ink (`inkR`),
       Edge Boiling (`boil`), plus **color-invariant `Luma-CLIP` & `RGB-SigLIP` identity curves**!
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
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp31b_video_dit_anime_vs_real_diagnostic as e31b


REGIMES_32 = [
    (
        "R1_HeadTurn",
        "The character smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent appearance.",
    ),
    (
        "R2_DynamicWind",
        "A sudden gust of wind blows the character's hair strongly across their face as they raise one hand to brush their bangs back with a surprised expression, dynamic lively motion.",
    ),
]


def _apply_rotary_emb(hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class ProbeWanAttnProcessor:
    """
    Exact replica of `WanAttnProcessor` that ALSO records Frame-0 Anchor Attention Mass
    (`mass_to_f0`, `mass_to_self`, `mass_to_other` per latent frame `t=1..T-1`) whenever `probe_state["active"]` is True!
    Zero modification to the forward pass output (exact bit-for-bit identical output via SDPA).
    """

    def __init__(self, orig_proc, layer_idx: int, probe_state: dict):
        self.orig_proc = orig_proc
        self.layer_idx = layer_idx
        self.probe_state = probe_state

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        # Always run exact original processor for zero-perturbation output
        out = self.orig_proc(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)

        # Only probe 3D self-attention (encoder_hidden_states is None) on active diagnostic steps
        if encoder_hidden_states is None and self.probe_state.get("active", False):
            T_lat = self.probe_state["T_lat"]
            seq_len = hidden_states.shape[1]
            if seq_len % T_lat == 0:
                N_sp = seq_len // T_lat
                stride = max(1, N_sp // 64)
                from diffusers.models.transformers.transformer_wan import _get_qkv_projections
                with torch.no_grad():
                    q, k, _v = _get_qkv_projections(attn, hidden_states, None)
                    q = attn.norm_q(q).unflatten(2, (attn.heads, -1))
                    k = attn.norm_k(k).unflatten(2, (attn.heads, -1))
                    if rotary_emb is not None:
                        q = _apply_rotary_emb(q, *rotary_emb)
                        k = _apply_rotary_emb(k, *rotary_emb)
                    q_h = q[0].transpose(0, 1).float()  # [H, T*N, D]
                    k_h = k[0].transpose(0, 1).float()  # [H, T*N, D]
                    scale = 1.0 / math.sqrt(q_h.shape[-1])
                    q_sub = q_h.view(attn.heads, T_lat, N_sp, -1)[:, :, ::stride, :]
                    n_sub = q_sub.shape[2]
                    logits = torch.bmm(q_sub.reshape(attn.heads, -1, q_h.shape[-1]), k_h.transpose(1, 2)) * scale
                    probs = torch.softmax(logits, dim=-1)
                    p_mat = probs.view(attn.heads, T_lat, n_sub, T_lat, N_sp).sum(dim=-1).mean(dim=(0, 2))  # [T_q, T_k]
                    p_np = p_mat.cpu().numpy()
                    # Record mass_to_f0 for generated frames t=1..T-1, plus mass_to_self!
                    f0_vec = p_np[1:, 0].tolist()
                    self_vec = [float(p_np[t, t]) for t in range(1, T_lat)]
                    tag = self.probe_state["pass_tag"]  # "cond" or "uncond"
                    step_idx = self.probe_state["step_idx"]
                    self.probe_state["attn_records"].setdefault(tag, {}).setdefault(str(step_idx), {})[str(self.layer_idx)] = {
                        "f0_mean": round(float(np.mean(f0_vec)), 5),
                        "self_mean": round(float(np.mean(self_vec)), 5),
                        "f0_by_t": [round(float(x), 4) for x in f0_vec],
                    }
        return out


@torch.no_grad()
def stage_probe(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n_pairs]
    tasks = []
    for p in pairs:
        for dom, id_k, path_k in [("anime", "anime_id", "anime_path"), ("real", "real_id", "real_path")]:
            for rname, rprompt in REGIMES_32:
                tasks.append((p["pair_idx"], dom, Path(p[path_k]), p[id_k], rname, rprompt))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    meta_dir = args.out_dir / "meta"
    strips_dir = args.out_dir / "strips"
    meta_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)

    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[1]}_{t[4]}.json").exists()]
    if not todo:
        print(f"[exp32a shard {args.shard}] all done!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # Install non-invasive ProbeWanAttnProcessor on ALL 30 DiT blocks' 3D self-attention (`attn1`)
    probe_state: dict = {"active": False, "attn_records": {}}
    for li, blk in enumerate(pipe.transformer.blocks):
        blk.attn1.set_processor(ProbeWanAttnProcessor(blk.attn1.processor, li, probe_state))

    # Wrap transformer forward to log exact velocity norms & CFG geometry at EVERY step!
    orig_fwd = pipe.transformer.forward
    vel_state: dict = {"step_calls": [], "step_idx": 0}
    PROBE_STEPS = {0, 5, 10, 15, 20, 25, 29}

    def wrapped_fwd(*f_args, **f_kwargs):
        call_idx = len(vel_state["step_calls"])
        pass_tag = "cond" if call_idx % 2 == 0 else "uncond"
        step_idx = call_idx // 2
        probe_state["pass_tag"] = pass_tag
        probe_state["step_idx"] = step_idx
        probe_state["active"] = (step_idx in PROBE_STEPS)
        out = orig_fwd(*f_args, **f_kwargs)
        v = out[0] if isinstance(out, tuple) else out  # [1, C, T_lat, H_lat, W_lat]
        vel_state["step_calls"].append(v.detach().float().cpu())
        return out

    pipe.transformer.forward = wrapped_fwd

    H, W, N_FRAMES, STEPS, GUIDANCE = 512, 512, 49, 30, 5.0
    # Wan2.2 VAE spatial compression = 16, patch_size = 2 -> N_sp = (512/16/2)^2 = 16*16 = 256!
    # Temporal compression = 4 -> T_lat = (49-1)//4 + 1 = 13!
    probe_state["T_lat"] = 13
    probe_state["N_sp"] = 256

    for pair_idx, dom, img_path, cid, rname, rprompt in todo:
        # 100% Square-to-Square resize (ZERO aspect cropping mismatch between Anime and Real!)
        im = Image.open(img_path).convert("RGB").resize((W, H), Image.LANCZOS)
        vel_state["step_calls"] = []
        probe_state["attn_records"] = {}

        gen = torch.Generator(device="cuda").manual_seed(5200 + pair_idx * 10 + (0 if rname == "R1_HeadTurn" else 1))
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

        pil_frames = e31b._to_pil_frames(frames)
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([np.asarray(im, dtype=np.uint8), np_frames[0], np_frames[-1]])
        st["inp_flat_t0"] = inp_st["flat_t0"]
        st["inp_edge_sharp_t0"] = inp_st["edge_sharp_t0"]
        st["flat_delta_vs_inp"] = round(st["flat_end"] - inp_st["flat_t0"], 4)
        st["airbrush_delta_vs_inp"] = round(st["airbrush_end"] - inp_st["airbrush_t0"], 4)
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)

        # Compute step-by-step velocity & CFG diagnostics on generated frames t=1..T_lat-1!
        calls = vel_state["step_calls"]
        step_vel = []
        for s in range(len(calls) // 2):
            vc = calls[2 * s][0, :, 1:]    # [C, 12, H_l, W_l]
            vu = calls[2 * s + 1][0, :, 1:]
            dv = vc - vu
            vc_n = float(torch.linalg.vector_norm(vc).item())
            vu_n = float(torch.linalg.vector_norm(vu).item())
            dv_n = float(torch.linalg.vector_norm(dv).item())
            cos_cu = float(F.cosine_similarity(vc.flatten().unsqueeze(0), vu.flatten().unsqueeze(0)).item())
            # Per-frame CFG velocity norm ratio (tail vs early frames)
            dv_by_t = [round(float(torch.linalg.vector_norm(dv[:, ti]).item()), 2) for ti in range(dv.shape[1])]
            step_vel.append({
                "step": s,
                "v_cond_norm": round(vc_n, 2),
                "v_uncond_norm": round(vu_n, 2),
                "cfg_diff_norm": round(dv_n, 2),
                "cfg_rel_ratio": round(dv_n / max(vu_n, 1e-5), 4),
                "cos_cond_uncond": round(cos_cu, 4),
                "cfg_diff_by_t": dv_by_t,
            })

        key = f"{pair_idx:02d}_{dom}_{rname}"
        strip_p = strips_dir / f"{key}.png"
        mean_dv = float(np.mean([x["cfg_diff_norm"] for x in step_vel]))
        # Mean Frame-0 attention across all 30 layers at step 10 (mid-high noise) & step 20 (fine detail)
        att_c = probe_state["attn_records"].get("cond", {})
        f0_s10 = float(np.mean([v["f0_mean"] for v in att_c.get("10", {}).values()])) if "10" in att_c else 0.0
        f0_s20 = float(np.mean([v["f0_mean"] for v in att_c.get("20", {}).values()])) if "20" in att_c else 0.0

        e31b._save_strip(
            pil_frames,
            strip_p,
            f"wan22_512 | {key} | mot={st['motion_l1']:.3f} shR={st['sharp_ratio_vs_inp']:.2f} cfgN={mean_dv:.1f} f0A10={f0_s10:.3f} f0A20={f0_s20:.3f}",
        )
        # Save keyframes for Luma-CLIP structural identity eval
        kf_dir = args.out_dir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        for fi in [0, 12, 24, 36, len(pil_frames) - 1]:
            pil_frames[fi].save(kf_dir / f"f{fi:02d}.png")

        rec = {
            "pair_idx": pair_idx,
            "domain": dom,
            "char_id": cid,
            "regime": rname,
            "stats": st,
            "step_vel": step_vel,
            "attn_records": probe_state["attn_records"],
            "summary_f0_s10": round(f0_s10, 5),
            "summary_f0_s20": round(f0_s20, 5),
            "summary_mean_cfg_norm": round(mean_dv, 2),
        }
        (meta_dir / f"{key}.json").write_text(json.dumps(rec), encoding="utf-8")
        print(
            f"[exp32a s{args.shard}] {key:24s} | mot={st['motion_l1']:.4f} shR={st['sharp_ratio_vs_inp']:.3f} | "
            f"cfgN={mean_dv:.1f} cos={np.mean([x['cos_cond_uncond'] for x in step_vel]):.3f} | f0Attn(s10={f0_s10:.4f}, s20={f0_s20:.4f})",
            flush=True,
        )
    return 0


@torch.no_grad()
def stage_analyze(args) -> int:
    """Aggregate all 32 instrumented runs + compute Luma-CLIP & RGB-SigLIP per-frame identity curves!"""
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    meta_dir = args.out_dir / "meta"
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(meta_dir.glob("*.json"))]

    # Load CLIP-ViT-L/14 for Luma-CLIP structural identity tracking across t=0,12,24,36,48!
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.bfloat16).to("cuda").eval()
    def enc_luma_clip(ims: list[Image.Image]) -> torch.Tensor:
        g_ims = [ImageOps.autocontrast(ImageOps.grayscale(im), cutoff=1).convert("RGB") for im in ims]
        pv = cp(images=g_ims, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)
        f = cm.get_image_features(pixel_values=pv)
        if not isinstance(f, torch.Tensor):
            f = f.pooler_output
        return F.normalize(f.float(), dim=-1)

    for r in recs:
        key = f"{r['pair_idx']:02d}_{r['domain']}_{r['regime']}"
        kf_dir = args.out_dir / "keyframes" / key
        ims = [Image.open(kf_dir / fn).convert("RGB") for fn in ["input.png", "f00.png", "f12.png", "f24.png", "f36.png", "f48.png"]]
        embs = enc_luma_clip(ims)  # [6, D]
        sims = (embs[1:] @ embs[0]).cpu().numpy().tolist()  # sim(f0, inp), sim(f12, inp), ..., sim(f48, inp)
        r["luma_clip_id"] = [round(float(x), 4) for x in sims]

    print(f"\n=== EXP 32A MECHANISTIC ANALYSIS (Wan2.2-TI2V-5B 512x512 Square Framing-Matched, n={len(recs)}) ===", flush=True)
    summary = {}
    for rname, _ in REGIMES_32:
        summary[rname] = {}
        for dom in ["anime", "real"]:
            sub = [r for r in recs if r["regime"] == rname and r["domain"] == dom]
            if not sub:
                continue
            mot = float(np.mean([r["stats"]["motion_l1"] for r in sub]))
            disp = float(np.mean([r["stats"]["disp_tail"] for r in sub]))
            shR = float(np.mean([r["stats"]["sharp_ratio_vs_inp"] for r in sub]))
            boil = float(np.mean([r["stats"]["edge_boiling"] for r in sub]))
            id_curve = np.mean([r["luma_clip_id"] for r in sub], axis=0)
            # Step-wise CFG diff norm (early 0..9, mid 10..19, late 20..29) & relative ratio
            cfg_e = float(np.mean([[x["cfg_diff_norm"] for x in r["step_vel"][:10]] for r in sub]))
            cfg_m = float(np.mean([[x["cfg_diff_norm"] for x in r["step_vel"][10:20]] for r in sub]))
            cfg_l = float(np.mean([[x["cfg_diff_norm"] for x in r["step_vel"][20:]] for r in sub]))
            rel_e = float(np.mean([[x["cfg_rel_ratio"] for x in r["step_vel"][:10]] for r in sub]))
            rel_m = float(np.mean([[x["cfg_rel_ratio"] for x in r["step_vel"][10:20]] for r in sub]))
            cos_all = float(np.mean([[x["cos_cond_uncond"] for x in r["step_vel"]] for r in sub]))
            vu_e = float(np.mean([[x["v_uncond_norm"] for x in r["step_vel"][:10]] for r in sub]))
            # Frame-0 Anchor Attention by depth band (Early L0-9, Mid L10-19, Late L20-29) for cond vs uncond at steps 5, 10, 20
            def band_f0(tag, st_str, l_lo, l_hi):
                vals = []
                for r in sub:
                    d = r["attn_records"].get(tag, {}).get(st_str, {})
                    for li in range(l_lo, l_hi):
                        if str(li) in d:
                            vals.append(d[str(li)]["f0_mean"])
                return round(float(np.mean(vals)), 5) if vals else 0.0

            def t_decay_f0(tag, st_str):
                # Mean f0 attention across all 30 layers for each latent frame t=1..12
                mats = []
                for r in sub:
                    d = r["attn_records"].get(tag, {}).get(st_str, {})
                    if d:
                        mats.append(np.mean([v["f0_by_t"] for v in d.values()], axis=0))
                return [round(float(x), 4) for x in np.mean(mats, axis=0)] if mats else []

            summary[rname][dom] = {
                "n": len(sub),
                "motion_l1": round(mot, 4),
                "disp_tail": round(disp, 4),
                "sharp_ratio": round(shR, 4),
                "edge_boiling": round(boil, 4),
                "luma_clip_id_0_12_24_36_48": [round(float(x), 4) for x in id_curve],
                "cfg_norm_early_mid_late": [round(cfg_e, 2), round(cfg_m, 2), round(cfg_l, 2)],
                "cfg_rel_early_mid": [round(rel_e, 4), round(rel_m, 4)],
                "v_uncond_norm_early": round(vu_e, 2),
                "cos_cond_uncond": round(cos_all, 4),
                "f0_attn_cond_s10_E_M_L": [band_f0("cond", "10", 0, 10), band_f0("cond", "10", 10, 20), band_f0("cond", "10", 20, 30)],
                "f0_attn_uncond_s10_E_M_L": [band_f0("uncond", "10", 0, 10), band_f0("uncond", "10", 10, 20), band_f0("uncond", "10", 20, 30)],
                "f0_attn_cond_s20_E_M_L": [band_f0("cond", "20", 0, 10), band_f0("cond", "20", 10, 20), band_f0("cond", "20", 20, 30)],
                "f0_attn_by_t_cond_s10": t_decay_f0("cond", "10"),
                "f0_attn_by_t_cond_s20": t_decay_f0("cond", "20"),
            }
            s = summary[rname][dom]
            print(
                f"  {rname:15s} {dom:5s} | mot={s['motion_l1']:.4f} shR={s['sharp_ratio']:.3f} boil={s['edge_boiling']:.4f} | "
                f"LumaCLIP_ID={s['luma_clip_id_0_12_24_36_48']} | "
                f"CFG_N(E/M/L)={s['cfg_norm_early_mid_late']} relE={rel_e:.3f} cos={cos_all:.3f} | "
                f"f0Attn_s10(E/M/L)={s['f0_attn_cond_s10_E_M_L']} uncond={s['f0_attn_uncond_s10_E_M_L']}",
                flush=True,
            )

        # Stack first 4 pairs visually
        paths = []
        for pid in range(4):
            paths.append(args.out_dir / "strips" / f"{pid:02d}_anime_{rname}.png")
            paths.append(args.out_dir / "strips" / f"{pid:02d}_real_{rname}.png")
        e31b._stack_strips(paths, args.out_dir / f"qc_wan22_512_{rname}.png")

    out_p = args.out_dir / "exp32a_mechanistic_summary.json"
    out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved mechanistic summary to {out_p}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["probe", "analyze"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp32a_wan22_mechanistic"))
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_probe(args) if args.stage == "probe" else stage_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
