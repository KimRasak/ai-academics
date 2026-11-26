#!/usr/bin/env python3
"""
Exp 38 (Mechanistic Opening of Flagship 27B MoE `Wan2.2-I2V-A14B` — High-Noise vs. Low-Noise Expert Causality):

Opens BOTH 14B Experts of `Wan2.2-I2V-A14B` (`transformer`: High-Noise Expert for t >= 900, steps 0..10;
`transformer_2`: Low-Noise Expert for t < 900, steps 11..39) to localize WHERE and HOW "Physical World-Model Literalism"
occurs on 2D Non-Physical Anime Grammar (`S3_NonBioAffect`, `G1_HighlightOff_Only`, `G3_AngerVein_Pop`, `G6_BWManga_Monochrome`
vs. `S1_HeadTurn` positive control):

  Stage 1 (`--stage probe`):
    - Measures per-step (s=0..39) spatial CFG velocity `||v_cond - v_uncond||` split across 4 anatomical zones:
      [hair (0..22%), upper_face/eyes (22..52%), lower_face/mouth (52..72%), torso/bg (72..100%)].
    - Measures Text->Video Cross-Attention (`attn2`) spatial leakage of trigger tokens ("shadow", "dark", "highlight", "vein")
      across BOTH High-Noise (`transformer`) and Low-Noise (`transformer_2`) experts!
    - Decodes the `x0` latent snapshot right at the MoE handoff (`s=10`) to visually verify whether blackout/literalism
      already forms inside the 14B High-Noise Expert before the Low-Noise Expert even runs!

  Stage 2 (`--stage causal_swap`):
    - Direct 2-way Causal MoE Expert-Swap on `S3_NonBioAffect` & `G1_HighlightOff_Only` & `G3_AngerVein_Pop`:
        * `M0_Base`: standard target prompt in both experts
        * `M1_HighTarget_LowNeutral`: target prompt in High-Noise Expert (s=0..10), neutral bright smile in Low-Noise (s=11..39)
        * `M2_HighNeutral_LowTarget`: neutral bright smile in High-Noise Expert (s=0..10), target prompt in Low-Noise (s=11..39)
        * `M3_SpatialGatedCFG`: zero-training inference surgery — blends target velocity strictly inside soft upper-face/eye mask
          while keeping neutral velocity on hair, lower face/smile, torso, and background!
    - Plus `M4_ChromaLock` zero-training monochrome chroma projection on `G6_BWManga_Monochrome`!
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
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp31b_video_dit_anime_vs_real_diagnostic as e31b
import exp32b_wan22_causal_dit_surgery as e32b
import exp36_flagship_27b_13b_anime_stress_test as e36
import exp37a_flagship_2d_grammar_benchmark as e37a


NEUTRAL_PROMPT = (
    "The character stays gently breathing in the exact same bright even studio lighting with unchanged "
    "bright hair, clothes, and gentle smile, clean 2D Japanese TV anime cel shading."
)
NEG_PROMPT = "blurry, low quality, distorted, extra limbs, watermark, text"

PROBE_REGIMES = [
    (
        "S1_HeadTurn",
        "The character smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent appearance.",
        ["turns", "head", "profile"],
        "color",
    ),
    (
        "S3_NonBioAffect",
        "The character's bright eye highlights suddenly vanish leaving hollow dark lifeless eyes while a sharp ominous dark shadow falls across their upper forehead, yet their mouth stays frozen in a calm eerie smile, 2D Japanese TV anime cel shading.",
        ["shadow", "dark", "hollow", "lifeless", "ominous"],
        "color",
    ),
    (
        "G1_HighlightOff_Only",
        "The character stays still in the exact same bright even studio lighting with unchanged bright hair, clothes, and gentle smile, while only the small white specular highlight dots inside both eyes smoothly fade away leaving flat matte dull irises, clean 2D Japanese TV anime.",
        ["highlight", "fade", "matte", "dull", "eyes"],
        "color",
    ),
    (
        "G3_AngerVein_Pop",
        "In bright flat 2D Japanese anime style, the character holds a polite closed-mouth smile as a classic stylized red cross-shaped anime anger vein symbol pops onto their upper forehead temple and pulses gently, everything else stays bright, clean, and consistent.",
        ["red", "cross", "anger", "vein", "symbol"],
        "color",
    ),
    (
        "G6_BWManga_Monochrome",
        "The black-and-white Japanese manga character smoothly turns their head to a three-quarter side profile and smiles gently, strictly maintaining pure monochrome black ink lines, crisp white paper, and fine halftone screentone dots with zero color.",
        ["monochrome", "black", "ink", "screentone"],
        "manga",
    ),
]


def _zone_norms(v: torch.Tensor) -> dict:
    """Compute per-zone L2 norm of latent velocity [B, C, T, H, W] on generated frames t>=1."""
    vg = v[0, :, 1:].float()  # [C, T-1, H, W]
    H, W = vg.shape[-2], vg.shape[-1]
    h1 = max(1, int(0.22 * H))
    h2 = max(h1 + 1, int(0.52 * H))
    h3 = max(h2 + 1, int(0.72 * H))
    # Mean per-pixel L2 magnitude in each zone
    pm = vg.norm(dim=0)  # [T-1, H, W]
    hair = float(pm[:, :h1, :].mean().item())
    up = float(pm[:, h1:h2, :].mean().item())
    low = float(pm[:, h2:h3, :].mean().item())
    torso = float(pm[:, h3:, :].mean().item())
    total = float(pm.mean().item())
    leak_ratio = (hair + low + torso) / max(3.0 * up, 1e-6)
    return {
        "total": round(total, 4),
        "hair": round(hair, 4),
        "upper_face": round(up, 4),
        "lower_face": round(low, 4),
        "torso": round(torso, 4),
        "leak_ratio": round(leak_ratio, 4),
    }


def _make_soft_upper_face_mask(H: int, W: int, device, dtype) -> torch.Tensor:
    """Smooth spatial mask [1, 1, 1, H, W] strictly covering upper forehead + eyes/temples (y in [0.22, 0.56], x in [0.20, 0.80])."""
    yy = torch.linspace(0.0, 1.0, H, device=device, dtype=torch.float32)[:, None]
    xx = torch.linspace(0.0, 1.0, W, device=device, dtype=torch.float32)[None, :]
    # Smooth super-Gaussian box around upper face/forehead/eyes
    my = torch.exp(-(((yy - 0.39) / 0.17) ** 4))
    mx = torch.exp(-(((xx - 0.50) / 0.30) ** 4))
    m = (my * mx).clamp(0.0, 1.0)
    return m[None, None, None, :, :].to(dtype=dtype)


class CrossAttnProbeProcessor:
    """Non-invasive wrapper around WanAttnProcessor for cross-attention (`attn2`) spatial localization."""

    def __init__(self, orig_proc, expert_tag: str, layer_idx: int, state: dict):
        self.orig_proc = orig_proc
        self.expert_tag = expert_tag
        self.layer_idx = layer_idx
        self.state = state

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        out = self.orig_proc(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
        if encoder_hidden_states is not None and self.state.get("active", False) and self.state.get("pass_tag") == "cond":
            t_idxs = self.state.get("trigger_idxs", [])
            if t_idxs:
                from diffusers.models.transformers.transformer_wan import _get_qkv_projections
                with torch.no_grad():
                    q, k, _v = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
                    q = attn.norm_q(q).unflatten(2, (attn.heads, -1))
                    k = attn.norm_k(k).unflatten(2, (attn.heads, -1))
                    # Average across temporal frames for query to keep compute tiny
                    T_lat = self.state["T_lat"]
                    H_p, W_p = self.state["H_p"], self.state["W_p"]
                    q_h = q[0].transpose(0, 1).float()  # [H, T*H_p*W_p, D]
                    k_h = k[0].transpose(0, 1).float()  # [H, L_txt, D]
                    q_last = q_h.view(attn.heads, T_lat, H_p * W_p, -1)[:, -1]  # last latent frame [H, HW, D]
                    scale = 1.0 / math.sqrt(q_h.shape[-1])
                    logits = torch.bmm(q_last, k_h.transpose(1, 2)) * scale  # [H, HW, L_txt]
                    probs = torch.softmax(logits, dim=-1).mean(dim=0)  # [HW, L_txt]
                    trig_map = probs[:, t_idxs].sum(dim=-1).view(H_p, W_p)  # [H_p, W_p]
                    h1 = max(1, int(0.22 * H_p))
                    h2 = max(h1 + 1, int(0.52 * H_p))
                    h3 = max(h2 + 1, int(0.72 * H_p))
                    tot = float(trig_map.sum().item()) + 1e-8
                    rec = {
                        "hair_frac": round(float(trig_map[:h1, :].sum().item()) / tot, 4),
                        "upper_face_frac": round(float(trig_map[h1:h2, :].sum().item()) / tot, 4),
                        "lower_face_frac": round(float(trig_map[h2:h3, :].sum().item()) / tot, 4),
                        "torso_frac": round(float(trig_map[h3:, :].sum().item()) / tot, 4),
                        "mean_mass": round(float(trig_map.mean().item()), 5),
                    }
                    self.state["xattn"].setdefault(f"{self.expert_tag}_s{self.state['step_idx']}", {})[str(self.layer_idx)] = rec
        return out


def _find_trigger_token_idxs(tokenizer, prompt: str, keywords: list[str]) -> list[int]:
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    ids = toks["input_ids"][0].tolist()
    pieces = tokenizer.convert_ids_to_tokens(ids)
    idxs = []
    for i, p in enumerate(pieces):
        pl = p.lower().replace("▁", "")
        if any(kw in pl for kw in keywords) and len(pl) >= 2:
            idxs.append(i)
    return idxs


def _decode_wan_latents(pipe, latents: torch.Tensor) -> list[Image.Image]:
    lat = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(lat.device, lat.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        lat.device, lat.dtype
    )
    lat = lat / latents_std + latents_mean
    video = pipe.vae.decode(lat, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
    return e31b._to_pil_frames(frames)


@torch.no_grad()
def _run_custom_wan22_a14b(
    pipe,
    im: Image.Image,
    prompt_high: str,
    prompt_low: str,
    seed: int,
    height: int = 512,
    width: int = 512,
    num_frames: int = 49,
    num_steps: int = 36,
    guidance_scale: float = 3.5,
    mode: str = "standard",
    probe_state: dict | None = None,
    probe_steps: set[int] | None = None,
    neutral_prompt: str = NEUTRAL_PROMPT,
) -> tuple[list[Image.Image], dict]:
    """
    Exact custom denoising loop for `Wan2.2-I2V-A14B` (27B MoE) supporting:
      - `standard`: full mechanistic probing of High-Noise & Low-Noise experts + MoE-handoff x0 snapshot
      - `spatial_gated_cfg`: zero-training upper-face spatial-masked velocity blending (Target inside mask, Neutral outside)
      - `chroma_lock`: zero-training monochrome chroma projection on decoded/step latents
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 1. Encode prompts on GPU cleanly (CPU offload hooks move text_encoder automatically)
    pe_high, neg_pe = pipe.encode_prompt(
        prompt=prompt_high, negative_prompt=NEG_PROMPT, do_classifier_free_guidance=True,
        num_videos_per_prompt=1, device=device, dtype=dtype,
    )
    if prompt_low == prompt_high:
        pe_low = pe_high
    else:
        pe_low, _ = pipe.encode_prompt(
            prompt=prompt_low, negative_prompt=NEG_PROMPT, do_classifier_free_guidance=True,
            num_videos_per_prompt=1, device=device, dtype=dtype,
        )
    if mode == "spatial_gated_cfg":
        pe_neut, _ = pipe.encode_prompt(
            prompt=neutral_prompt, negative_prompt=NEG_PROMPT, do_classifier_free_guidance=True,
            num_videos_per_prompt=1, device=device, dtype=dtype,
        )
    else:
        pe_neut = None

    # 2. Scheduler & latents
    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    boundary_timestep = pipe.config.boundary_ratio * pipe.scheduler.config.num_train_timesteps  # 900.0

    image_tensor = pipe.video_processor.preprocess(im, height=height, width=width).to(device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(seed)
    num_channels_latents = pipe.vae.config.z_dim
    latents, condition = pipe.prepare_latents(
        image_tensor, 1, num_channels_latents, height, width, num_frames,
        torch.float32, device, gen, None, None,
    )
    T_lat, H_lat, W_lat = latents.shape[2], latents.shape[3], latents.shape[4]
    if probe_state is not None:
        probe_state["T_lat"] = T_lat
        probe_state["H_p"] = H_lat // 2
        probe_state["W_p"] = W_lat // 2

    spatial_mask = _make_soft_upper_face_mask(H_lat, W_lat, device, torch.float32) if mode == "spatial_gated_cfg" else None
    step_logs = []
    handoff_pil = None

    for i, t in enumerate(timesteps):
        is_high = bool(t >= boundary_timestep)
        current_model = pipe.transformer if is_high else pipe.transformer_2
        cur_pe = pe_high if is_high else pe_low
        latent_model_input = torch.cat([latents, condition], dim=1).to(dtype)
        timestep = t.expand(latents.shape[0])

        if probe_state is not None:
            probe_state["step_idx"] = i
            probe_state["active"] = bool(probe_steps and i in probe_steps)

        if probe_state is not None:
            probe_state["pass_tag"] = "cond"
        v_cond = current_model(
            hidden_states=latent_model_input, timestep=timestep,
            encoder_hidden_states=cur_pe, return_dict=False,
        )[0].float()

        if probe_state is not None:
            probe_state["pass_tag"] = "uncond"
            probe_state["active"] = False
        v_uncond = current_model(
            hidden_states=latent_model_input, timestep=timestep,
            encoder_hidden_states=neg_pe, return_dict=False,
        )[0].float()

        dv = v_cond - v_uncond
        if mode == "spatial_gated_cfg":
            v_neut = current_model(
                hidden_states=latent_model_input, timestep=timestep,
                encoder_hidden_states=pe_neut, return_dict=False,
            )[0].float()
            v_pred_target = v_uncond + guidance_scale * dv
            v_pred_neut = v_uncond + guidance_scale * (v_neut - v_uncond)
            # Blend target velocity strictly inside upper-face/eye mask; preserve neutral outside!
            v_pred = spatial_mask * v_pred_target + (1.0 - spatial_mask) * v_pred_neut
        else:
            v_pred = v_uncond + guidance_scale * dv

        z_cfg = _zone_norms(dv)
        step_logs.append({"step": i, "t": round(float(t.item()), 1), "expert": "high" if is_high else "low", "cfg_zones": z_cfg})

        # Capture decoded x0 preview right at the last High-Noise step (MoE handoff!)
        next_is_low = (i + 1 < len(timesteps)) and (float(timesteps[i + 1].item()) < boundary_timestep)
        if probe_state is not None and is_high and next_is_low and probe_state.get("save_handoff", False):
            sigma = float(t.item()) / 1000.0
            x0_est = latents - sigma * v_pred
            handoff_frames = _decode_wan_latents(pipe, x0_est)
            handoff_pil = handoff_frames[-1]

        latents = pipe.scheduler.step(v_pred, t, latents, return_dict=False)[0]

    pil_frames = _decode_wan_latents(pipe, latents)
    if mode == "chroma_lock":
        # Pure monochrome luminance-preserving projection for B&W manga
        out_mono = []
        for f in pil_frames:
            g = f.convert("L").convert("RGB")
            out_mono.append(g)
        pil_frames = out_mono

    return pil_frames, {"steps": step_logs, "handoff_pil": handoff_pil}


@torch.no_grad()
def stage_probe(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    sel_pids = [0, 2, 4, 6]
    manga_dir = Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp29_p7_retrieval_manga/native_bw_manga_imgs")
    avail_manga = sorted(manga_dir.glob("*_manga.png"))
    manga_by_pid = {pid: (manga_dir / f"{pairs[pid]['anime_id']}_manga.png" if (manga_dir / f"{pairs[pid]['anime_id']}_manga.png").exists() else avail_manga[i]) for i, pid in enumerate(sel_pids)}

    tasks = []
    for pid in sel_pids:
        p = pairs[pid]
        for rname, rprompt, kws, stype in PROBE_REGIMES:
            img_path = Path(p["anime_path"]) if stype == "color" else manga_by_pid[pid]
            cid = p["anime_id"] if stype == "color" else img_path.stem
            tasks.append((pid, img_path, cid, rname, rprompt, kws))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    out_p = args.out_dir / "probe"
    meta_dir, ho_dir = out_p / "meta", out_p / "handoff"
    meta_dir.mkdir(parents=True, exist_ok=True)
    ho_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[3]}.json").exists()]
    if not todo:
        print(f"[probe s{args.shard}] done!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    probe_state = {"active": False, "xattn": {}, "save_handoff": True}
    probe_layers = [0, 10, 20, 30, 39]
    for li in probe_layers:
        b1 = pipe.transformer.blocks[li].attn2
        b1.set_processor(CrossAttnProbeProcessor(b1.processor, "high", li, probe_state))
        b2 = pipe.transformer_2.blocks[li].attn2
        b2.set_processor(CrossAttnProbeProcessor(b2.processor, "low", li, probe_state))

    probe_steps = {2, 6, 9, 14, 24, 34}
    for pid, img_path, cid, rname, rprompt, kws in todo:
        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        seed = 8800 + pid * 31 + [r[0] for r in PROBE_REGIMES].index(rname)
        probe_state["xattn"] = {}
        probe_state["trigger_idxs"] = _find_trigger_token_idxs(pipe.tokenizer, rprompt, kws)

        pil_frames, info = _run_custom_wan22_a14b(
            pipe, im, rprompt, rprompt, seed=seed, num_frames=49, num_steps=36,
            mode="standard", probe_state=probe_state, probe_steps=probe_steps,
        )
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        st.update(e37a._grammar_diagnostics(np_frames, inp_u8))

        key = f"{pid:02d}_{rname}"
        if info["handoff_pil"] is not None:
            # Side-by-side: Input | MoE Handoff (s=9, End of High-Noise Expert) | Final (s=35, End of Low-Noise Expert)
            W, H = 256, 256
            canvas = Image.new("RGB", (W * 3, H + 28), (18, 18, 22))
            for idx_c, (lbl, im_c) in enumerate([("Input t=0", im), ("End of High-Noise Expert (x0)", info["handoff_pil"]), ("Final Output (End Low-Noise)", pil_frames[-1])]):
                canvas.paste(im_c.resize((W, H), Image.LANCZOS), (idx_c * W, 28))
            dr = ImageDraw.Draw(canvas)
            dr.text((8, 6), f"{key} ({cid}) | Left: Input | Mid: High-Noise Expert x0 | Right: Final Output", fill=(240, 240, 240))
            canvas.save(ho_dir / f"handoff_{key}.png")

        # Summarize High vs Low CFG velocity zones
        hi_s = [s["cfg_zones"] for s in info["steps"] if s["expert"] == "high"]
        lo_s = [s["cfg_zones"] for s in info["steps"] if s["expert"] == "low"]
        summary = {
            "high_nsteps": len(hi_s),
            "low_nsteps": len(lo_s),
            "high_cfg": {k: round(float(np.mean([x[k] for x in hi_s])), 4) for k in hi_s[0]} if hi_s else {},
            "low_cfg": {k: round(float(np.mean([x[k] for x in lo_s])), 4) for k in lo_s[0]} if lo_s else {},
        }
        rec = {
            "pair_idx": pid, "char_id": cid, "regime": rname, "trigger_idxs": probe_state["trigger_idxs"],
            "stats": st, "moe_summary": summary, "steps": info["steps"], "xattn": probe_state["xattn"],
        }
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[probe s{args.shard}] {key:25s} | Hi(n={summary['high_nsteps']}): up={summary['high_cfg'].get('upper_face',0):.3f} hair={summary['high_cfg'].get('hair',0):.3f} low={summary['high_cfg'].get('lower_face',0):.3f} LR={summary['high_cfg'].get('leak_ratio',0):.2f} | "
            f"Lo(n={summary['low_nsteps']}): up={summary['low_cfg'].get('upper_face',0):.3f} hair={summary['low_cfg'].get('hair',0):.3f} LR={summary['low_cfg'].get('leak_ratio',0):.2f}",
            flush=True,
        )
    return 0


@torch.no_grad()
def stage_causal_swap(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    sel_pids = [0, 2, 4, 6]
    manga_dir = Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp29_p7_retrieval_manga/native_bw_manga_imgs")
    avail_manga = sorted(manga_dir.glob("*_manga.png"))
    manga_by_pid = {pid: (manga_dir / f"{pairs[pid]['anime_id']}_manga.png" if (manga_dir / f"{pairs[pid]['anime_id']}_manga.png").exists() else avail_manga[i]) for i, pid in enumerate(sel_pids)}

    swap_regimes = {r[0]: (r[1], r[3]) for r in PROBE_REGIMES if r[0] in ("S3_NonBioAffect", "G1_HighlightOff_Only", "G3_AngerVein_Pop", "G6_BWManga_Monochrome")}
    tasks = []
    for pid in sel_pids:
        p = pairs[pid]
        for rname, (rprompt, stype) in swap_regimes.items():
            img_path = Path(p["anime_path"]) if stype == "color" else manga_by_pid[pid]
            cid = p["anime_id"] if stype == "color" else img_path.stem
            if rname == "G6_BWManga_Monochrome":
                methods = ["M0_Base", "M4_ChromaLock"]
            else:
                methods = ["M0_Base", "M1_HighTarget_LowNeutral", "M2_HighNeutral_LowTarget", "M3_SpatialGatedCFG"]
            for meth in methods:
                tasks.append((pid, img_path, cid, rname, rprompt, meth))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    out_c = args.out_dir / "causal_swap"
    meta_dir, strips_dir, mp4_dir = out_c / "meta", out_c / "strips", out_c / "mp4"
    for d in [meta_dir, strips_dir, mp4_dir]:
        d.mkdir(parents=True, exist_ok=True)
    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[3]}_{t[5]}.json").exists()]
    if not todo:
        print(f"[swap s{args.shard}] done!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    for pid, img_path, cid, rname, rprompt, meth in todo:
        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        seed = 8900 + pid * 31 + list(swap_regimes.keys()).index(rname)

        if meth == "M0_Base":
            ph, pl, mode = rprompt, rprompt, "standard"
        elif meth == "M1_HighTarget_LowNeutral":
            ph, pl, mode = rprompt, NEUTRAL_PROMPT, "standard"
        elif meth == "M2_HighNeutral_LowTarget":
            ph, pl, mode = NEUTRAL_PROMPT, rprompt, "standard"
        elif meth == "M3_SpatialGatedCFG":
            ph, pl, mode = rprompt, rprompt, "spatial_gated_cfg"
        elif meth == "M4_ChromaLock":
            ph, pl, mode = rprompt, rprompt, "chroma_lock"
        else:
            raise ValueError(meth)

        pil_frames, _ = _run_custom_wan22_a14b(
            pipe, im, ph, pl, seed=seed, num_frames=49, num_steps=36, mode=mode,
        )
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e37a._grammar_diagnostics(np_frames, inp_u8))

        key = f"{pid:02d}_{rname}_{meth}"
        e36._save_single_mp4(pil_frames, mp4_dir / f"{key}.mp4", fps=16)
        e31b._save_strip(
            pil_frames,
            strips_dir / f"{key}.png",
            f"{meth:24s} | {pid:02d}_{rname} | dL_glob={st['global_luma_delta']:+.2f} dL_hair={st['hair_luma_delta']:+.2f} dL_up={st['upper_face_luma_delta']:+.2f} dL_low={st['lower_face_luma_delta']:+.2f} shR={st['sharp_ratio_vs_inp']:.2f} chr={st['chroma_peak']:.3f}",
            n_show=6,
            thumb_h=170,
        )
        rec = {"pair_idx": pid, "char_id": cid, "regime": rname, "method": meth, "stats": st}
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[swap s{args.shard}] {key:42s} | dL(glob={st['global_luma_delta']:+.3f} hair={st['hair_luma_delta']:+.3f} up={st['upper_face_luma_delta']:+.3f} low={st['lower_face_luma_delta']:+.3f}) | shR={st['sharp_ratio_vs_inp']:.3f} chr={st['chroma_peak']:.3f}",
            flush=True,
        )
    return 0


def stage_summarize(args) -> int:
    # Summarize probe
    out_p = args.out_dir / "probe"
    if (out_p / "meta").exists():
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_p / "meta").glob("*.json"))]
        print(f"\n=== EXP 38 PROBE SUMMARY (n={len(recs)} runs, Wan2.2-I2V-A14B 27B MoE) ===", flush=True)
        for rname, _, _, _ in PROBE_REGIMES:
            sub = [r for r in recs if r["regime"] == rname]
            if not sub: continue
            hi_up = np.mean([r["moe_summary"]["high_cfg"]["upper_face"] for r in sub])
            hi_hr = np.mean([r["moe_summary"]["high_cfg"]["hair"] for r in sub])
            hi_lo = np.mean([r["moe_summary"]["high_cfg"]["lower_face"] for r in sub])
            hi_lr = np.mean([r["moe_summary"]["high_cfg"]["leak_ratio"] for r in sub])
            lo_up = np.mean([r["moe_summary"]["low_cfg"]["upper_face"] for r in sub])
            lo_hr = np.mean([r["moe_summary"]["low_cfg"]["hair"] for r in sub])
            lo_lr = np.mean([r["moe_summary"]["low_cfg"]["leak_ratio"] for r in sub])
            # Cross-attention upper_face fraction (High vs Low)
            hi_xa = [v["upper_face_frac"] for r in sub for k, d in r["xattn"].items() if k.startswith("high_") for v in d.values()]
            lo_xa = [v["upper_face_frac"] for r in sub for k, d in r["xattn"].items() if k.startswith("low_") for v in d.values()]
            print(
                f"  {rname:22s} | HIGH(s=0..9): up={hi_up:.3f} hair={hi_hr:.3f} low={hi_lo:.3f} LR={hi_lr:.2f} XAttn_up={np.mean(hi_xa) if hi_xa else 0:.3f} | "
                f"LOW(s=10..35): up={lo_up:.3f} hair={lo_hr:.3f} LR={lo_lr:.2f} XAttn_up={np.mean(lo_xa) if lo_xa else 0:.3f}",
                flush=True,
            )
            ho_paths = [out_p / "handoff" / f"handoff_{pid:02d}_{rname}.png" for pid in [0, 2, 4, 6] if (out_p / "handoff" / f"handoff_{pid:02d}_{rname}.png").exists()]
            if ho_paths:
                e31b._stack_strips(ho_paths, out_p / f"qc38_handoff_{rname}.png")

    # Summarize causal_swap
    out_c = args.out_dir / "causal_swap"
    if (out_c / "meta").exists():
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_c / "meta").glob("*.json"))]
        print(f"\n=== EXP 38 CAUSAL MoE EXPERT-SWAP & SPATIAL-GATED CFG SUMMARY (n={len(recs)} clips) ===", flush=True)
        for rname in ["S3_NonBioAffect", "G1_HighlightOff_Only", "G3_AngerVein_Pop", "G6_BWManga_Monochrome"]:
            sub_r = [r for r in recs if r["regime"] == rname]
            if not sub_r: continue
            print(f"  -- {rname} --", flush=True)
            for meth in sorted(set(r["method"] for r in sub_r)):
                sub = [r for r in sub_r if r["method"] == meth]
                print(
                    f"    {meth:26s} | dL(glob={np.mean([r['stats']['global_luma_delta'] for r in sub]):+.3f} "
                    f"hair={np.mean([r['stats']['hair_luma_delta'] for r in sub]):+.3f} "
                    f"up={np.mean([r['stats']['upper_face_luma_delta'] for r in sub]):+.3f} "
                    f"low={np.mean([r['stats']['lower_face_luma_delta'] for r in sub]):+.3f}) | "
                    f"shR={np.mean([r['stats']['sharp_ratio_vs_inp'] for r in sub]):.3f} chrPk={np.mean([r['stats']['chroma_peak'] for r in sub]):.3f}",
                    flush=True,
                )
            for pid in [0, 2, 4, 6]:
                paths = [out_c / "strips" / f"{pid:02d}_{rname}_{m}.png" for m in ["M0_Base", "M1_HighTarget_LowNeutral", "M2_HighNeutral_LowTarget", "M3_SpatialGatedCFG", "M4_ChromaLock"] if (out_c / "strips" / f"{pid:02d}_{rname}_{m}.png").exists()]
                if paths:
                    e31b._stack_strips(paths, out_c / f"cmp38_{pid:02d}_{rname}.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["probe", "causal_swap", "summarize"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp38_moe_mechanistic"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "probe":
        return stage_probe(args)
    if args.stage == "causal_swap":
        return stage_causal_swap(args)
    return stage_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
