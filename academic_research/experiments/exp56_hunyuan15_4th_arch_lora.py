#!/usr/bin/env python3
"""
Exp 56 (4th & Newest 2026 Flagship Architecture Weight-Level Cure + Sigma-Space & Depth Localization):
  Architecture: `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v`
    - 54-layer Dual-Stream MM-DiT (`hidden=2048`, `16×128`, `in_channels=65 = 32 xt + 32 cond + 1 mask`)
    - Dual text encoders: `Qwen2.5-VL` (`text_embed_dim=3584`) + `ByT5` (`text_embed_2_dim=1472`)
    - Vision encoder: `SigLIP` (`image_embed_dim=1152`)
    - 32-channel `4×16×16` 3D-VAE (`scaling_factor=1.03682`)
    - FlowMatch `shift=7.0` (`FlowMatchEulerDiscreteScheduler`, `use_meanflow=False`)

Completes the 4-FOR-4 100% FLAGSHIP MATRIX (`Wan2.2-27B MoE` shift=3.0, `LTX-13B` linear_quadratic,
`HunyuanVideo-I2V 13B` shift=17.0, `HunyuanVideo-1.5 54L` shift=7.0):
  1. `encode_cache`: Encodes the exact 176 Clean v2 clips (`33f × 512×512`) across 8 GPUs.
  2. `sign_check`: Mandatory 1-step base velocity cosine check (`cos > +0.90`).
  3. `train`: 4-GPU DDP `r=16` LoRA (`28.3M` params = `0.34%` of 8.3B DiT) on `shift=7.0` flow matching.
  4. `eval`: 8-GPU inference across `fc_00..03` × `6` regimes (`S3, G1, G2, G3, G4, S1_Real`) ×
     4 noise-space modes (`H0_Base`, `H1_Sigma_lt800`, `H2_Sigma_ge800`, `H3_Full`) PLUS
     3 zero-retraining depth slices (`HD1_Early_0_17`, `HD2_Mid_18_35`, `HD3_Late_36_53`) on
     4 grammar regimes (`S3, G2, G3, G4`)!
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp31b_video_dit_anime_vs_real_diagnostic as e31b
import exp32b_wan22_causal_dit_surgery as e32b
import exp36_flagship_27b_13b_anime_stress_test as e36
import exp37a_flagship_2d_grammar_benchmark as e37a
import exp39_build_2d_grammar_training_cache as e39
import exp39v2_clean_fg_specular_cache as e39v2
import exp40_train_wan22_a14b_moe_lora as e40

MODEL_ID = "/gluster_hdd_cv/user/jinzili/research_2026_anime/hf_cache/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-720p_i2v/snapshots/dcf088ef4e420dd54bc25930ee810bb6613dfcb5"

HY15_EVAL_REGIMES = [
    r for r in e40.EVAL_REGIMES
    if r[0] in (
        "S3_NonBioAffect",
        "G1_HighlightOff_Only",
        "G2_ForeheadShadow_Clean",
        "G3_AngerVein_Pop",
        "G4_SweatDrop_Slide",
        "S1_Safety_RealHeadTurn",
    )
]

DEPTH_REGIMES = ("S3_NonBioAffect", "G2_ForeheadShadow_Clean", "G3_AngerVein_Pop", "G4_SweatDrop_Slide")


def _pils_to_vae_tensor(frames: list[Image.Image], size: int = 512) -> torch.Tensor:
    arr = np.stack([np.asarray(f.resize((size, size), Image.LANCZOS), dtype=np.float32) for f in frames], axis=0)
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)
    return t / 127.5 - 1.0


@torch.no_grad()
def stage_encode_cache(args) -> int:
    from diffusers import HunyuanVideo15ImageToVideoPipeline

    by_char = e39._load_train_char_map(args.scale48_manifest)
    cids = sorted(by_char.keys())
    tasks = []
    for ci, cid in enumerate(cids):
        vmap = by_char[cid]
        for vi, var in enumerate(["eye_highlight_off", "forehead_shadow_on", "dead_eyes_and_shadow", "anger_vein_on", "sweat_drop_on"]):
            if var not in vmap or "orig" not in vmap:
                continue
            prompts = e39.PROMPT_TEMPLATES[var]
            prompt = prompts[(ci + vi) % len(prompts)]
            tasks.append(("grammar", cid, var, vmap["orig"], vmap[var], vmap.get("eye_highlight_off"), prompt))

    anc_dir = Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp39_2d_grammar_train/anchors_frames")
    if anc_dir.exists():
        for d in sorted(anc_dir.iterdir()):
            mp = d / "meta.json"
            if mp.exists():
                m = json.loads(mp.read_text(encoding="utf-8"))
                tasks.append(("anchor", m["char_id"], m["variant"], d, None, None, m["prompt"]))

    cache_dir = args.out_dir / "latent_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (cache_dir / f"{t[1]}__{t[2]}.pt").exists()]
    if not todo:
        print(f"[hy15 encode s{args.shard}] all {len(my_tasks)} clips cached!", flush=True)
        return 0

    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
        MODEL_ID, transformer=None, torch_dtype=torch.bfloat16
    )
    pipe.vae.to("cuda", dtype=torch.bfloat16).eval()
    pipe.vae.enable_tiling()
    pipe.image_encoder.to("cuda", dtype=torch.bfloat16).eval()
    pipe.text_encoder.to("cuda", dtype=torch.bfloat16).eval()
    pipe.text_encoder_2.to("cuda", dtype=torch.bfloat16).eval()
    sf = float(pipe.vae.config.scaling_factor)

    prompt_cache = {}
    for idx_t, (kind, cid, var, src_a, src_b, eye_p, prompt) in enumerate(todo):
        if kind == "grammar":
            orig_im = Image.open(src_a).convert("RGB")
            var_im = Image.open(src_b).convert("RGB")
            eye_im = Image.open(eye_p).convert("RGB") if eye_p else None
            if var in ("eye_highlight_off", "forehead_shadow_on", "dead_eyes_and_shadow"):
                frames49 = e39v2._synthesize_clean_clip(orig_im, var_im, var, T=49, size=args.size, eye_im=eye_im)
            else:
                frames49 = e39._synthesize_grammar_clip(orig_im, var_im, var, T=49, size=args.size, eye_im=eye_im)
            idxs = [round(k * (len(frames49) - 1) / (args.num_frames - 1)) for k in range(args.num_frames)]
            frames = [frames49[k] for k in idxs]
            orig_im = frames[0]
        else:
            frames49 = [
                Image.open(Path(src_a) / f"f_{fi:03d}.png").convert("RGB").resize((args.size, args.size), Image.LANCZOS)
                for fi in range(49)
            ]
            idxs = [round(k * (len(frames49) - 1) / (args.num_frames - 1)) for k in range(args.num_frames)]
            frames = [frames49[k] for k in idxs]
            orig_im = frames[0]

        vid_t = _pils_to_vae_tensor(frames, size=args.size).to("cuda", dtype=torch.bfloat16)
        img_t = vid_t[:, :, :1]
        x0 = pipe.vae.encode(vid_t).latent_dist.mode() * sf
        img_lat = pipe.vae.encode(img_t).latent_dist.mode() * sf

        img_emb = pipe.encode_image(image=orig_im, batch_size=1, device=torch.device("cuda"), dtype=torch.bfloat16)

        if prompt not in prompt_cache:
            pe, pm, pe2, pm2 = pipe.encode_prompt(
                prompt=prompt,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
                batch_size=1,
            )
            prompt_cache[prompt] = (pe.cpu().to(torch.bfloat16), pm.cpu(), pe2.cpu().to(torch.bfloat16), pm2.cpu())
        pe_c, pm_c, pe2_c, pm2_c = prompt_cache[prompt]

        rec = {
            "x0": x0.cpu().to(torch.bfloat16),
            "image_latents": img_lat.cpu().to(torch.bfloat16),
            "image_embeds": img_emb.cpu().to(torch.bfloat16),
            "prompt_embeds": pe_c,
            "prompt_embeds_mask": pm_c,
            "prompt_embeds_2": pe2_c,
            "prompt_embeds_mask_2": pm2_c,
            "kind": kind,
            "char_id": cid,
            "variant": var,
        }
        torch.save(rec, cache_dir / f"{cid}__{var}.pt")
        if (idx_t + 1) % 5 == 0 or idx_t == len(todo) - 1:
            print(
                f"[hy15 encode s{args.shard}] {idx_t+1}/{len(todo)} | {cid}__{var} "
                f"x0={tuple(x0.shape)} img_emb={tuple(img_emb.shape)} pe={tuple(pe_c.shape)}",
                flush=True,
            )
    return 0


def _build_cond_and_mask(x0: torch.Tensor, img_lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, C, F_lat, H_lat, W_lat = x0.shape
    cond = img_lat.repeat(1, 1, F_lat, 1, 1).clone()
    cond[:, :, 1:, :, :] = 0.0
    mask = torch.zeros((B, 1, F_lat, H_lat, W_lat), device=x0.device, dtype=x0.dtype)
    mask[:, :, 0, :, :] = 1.0
    return cond, mask


@torch.no_grad()
def stage_sign_check(args) -> int:
    from diffusers import HunyuanVideo15Transformer3DModel

    pts = sorted((args.out_dir / "latent_cache").glob("*.pt"))
    rec = torch.load(pts[0], map_location="cuda")
    tr = HunyuanVideo15Transformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    x0 = rec["x0"].float()
    img_lat = rec["image_latents"].float()
    cond, mask = _build_cond_and_mask(x0, img_lat)

    torch.manual_seed(42)
    eps = torch.randn_like(x0)
    for sigma in (0.50, 0.85):
        xt = (1.0 - sigma) * x0 + sigma * eps
        inp = torch.cat([xt, cond, mask], dim=1).to(torch.bfloat16)
        t_tensor = torch.tensor([sigma * 1000.0], device="cuda", dtype=torch.bfloat16)
        pred = tr(
            hidden_states=inp,
            image_embeds=rec["image_embeds"].to(torch.bfloat16),
            timestep=t_tensor,
            timestep_r=None,
            encoder_hidden_states=rec["prompt_embeds"].to(torch.bfloat16),
            encoder_attention_mask=rec["prompt_embeds_mask"].to("cuda"),
            encoder_hidden_states_2=rec["prompt_embeds_2"].to(torch.bfloat16),
            encoder_attention_mask_2=rec["prompt_embeds_mask_2"].to("cuda"),
            return_dict=False,
        )[0].float()
        target_v = eps - x0
        cos = F.cosine_similarity(pred.flatten(), target_v.flatten(), dim=0).item()
        mse = F.mse_loss(pred, target_v).item()
        print(f"[hy15 sign_check] sigma={sigma:.2f} | cos(pred, eps-x0)={cos:+.4f} | mse={mse:.4f}", flush=True)
        assert cos > 0.70, f"Sign check FAILED at sigma={sigma}: cos={cos}"
    print("[hy15 sign_check] PASSED!", flush=True)
    return 0


def stage_train(args) -> int:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from diffusers import HunyuanVideo15Transformer3DModel
    from peft import LoraConfig, get_peft_model

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        store_path = os.environ.get("DDP_STORE_FILE", "/tmp/exp56_hy15_ddp_store")
        dist.init_process_group(
            "nccl",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
        )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    pts = sorted((args.out_dir / "latent_cache").glob("*.pt"))
    recs = [torch.load(p, map_location="cpu") for p in pts]
    if local_rank == 0:
        print(f"[hy15 train] Loaded {len(recs)} cached clips | world_size={world_size} steps={args.steps}", flush=True)

    tr = HunyuanVideo15Transformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    tr.requires_grad_(False)
    tr.enable_gradient_checkpointing()

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0.0,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        ],
    )
    peft_tr = get_peft_model(tr, lora_cfg)
    n_train = sum(p.numel() for p in peft_tr.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in peft_tr.parameters())
    if local_rank == 0:
        print(f"[hy15 train] Trainable LoRA params: {n_train:,} / {n_tot:,} ({100.0*n_train/n_tot:.3f}%)", flush=True)

    if world_size > 1:
        ddp_tr = DDP(peft_tr, device_ids=[local_rank], find_unused_parameters=True)
    else:
        ddp_tr = peft_tr

    opt = torch.optim.AdamW([p for p in peft_tr.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-2)
    rng = random.Random(2026 + local_rank * 131)
    torch.manual_seed(2026 + local_rank * 131)

    t0 = time.time()
    loss_hist = []
    shift = 7.0

    for step in range(1, args.steps + 1):
        # Cosine LR with 40-step linear warmup
        if step <= 40:
            lr_now = args.lr * (step / 40.0)
        else:
            prog = (step - 40) / max(1, args.steps - 40)
            lr_now = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for acc in range(args.grad_accum):
            rec = recs[rng.randrange(len(recs))]
            x0 = rec["x0"].to(device, dtype=torch.float32)
            img_lat = rec["image_latents"].to(device, dtype=torch.float32)
            cond, mask = _build_cond_and_mask(x0, img_lat)

            # Logit-normal with exact HunyuanVideo-1.5 shift=7.0
            u_raw = torch.sigmoid(torch.randn(1, device=device) * 1.0)
            sigma = (shift * u_raw) / (1.0 + (shift - 1.0) * u_raw)
            sigma = sigma.clamp(0.02, 0.995)

            eps = torch.randn_like(x0)
            s_view = sigma.view(1, 1, 1, 1, 1)
            xt = (1.0 - s_view) * x0 + s_view * eps
            target_v = eps - x0

            inp = torch.cat([xt, cond, mask], dim=1).to(torch.bfloat16)
            t_tensor = (sigma * 1000.0).to(torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = ddp_tr(
                    hidden_states=inp,
                    image_embeds=rec["image_embeds"].to(device, dtype=torch.bfloat16),
                    timestep=t_tensor,
                    timestep_r=None,
                    encoder_hidden_states=rec["prompt_embeds"].to(device, dtype=torch.bfloat16),
                    encoder_attention_mask=rec["prompt_embeds_mask"].to(device),
                    encoder_hidden_states_2=rec["prompt_embeds_2"].to(device, dtype=torch.bfloat16),
                    encoder_attention_mask_2=rec["prompt_embeds_mask_2"].to(device),
                    return_dict=False,
                )[0]
                loss = F.mse_loss(pred.float(), target_v) / args.grad_accum

            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_([p for p in peft_tr.parameters() if p.requires_grad], 1.0)
        opt.step()
        loss_hist.append(step_loss)

        if local_rank == 0 and (step % 25 == 0 or step == 1 or step == args.steps):
            avg25 = sum(loss_hist[-25:]) / len(loss_hist[-25:])
            el = time.time() - t0
            print(
                f"[hy15 train] step {step:4d}/{args.steps} | loss={step_loss:.4f} avg25={avg25:.4f} "
                f"| lr={lr_now:.2e} | {el:.1f}s",
                flush=True,
            )

    if local_rank == 0:
        save_dir = args.out_dir / f"lora_hy15_r{args.rank}"
        save_dir.mkdir(parents=True, exist_ok=True)
        peft_tr.save_pretrained(save_dir)
        meta = {
            "model_id": MODEL_ID,
            "rank": args.rank,
            "steps": args.steps,
            "trainable_params": n_train,
            "total_params": n_tot,
            "pct_params": round(100.0 * n_train / n_tot, 4),
            "final_avg25_loss": round(sum(loss_hist[-25:]) / len(loss_hist[-25:]), 5),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        (save_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[hy15 train] Saved LoRA -> {save_dir} | {meta}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()
    return 0


def _collect_lora_scales(peft_model) -> dict[str, tuple[object, float]]:
    out = {}
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict) and "default" in mod.scaling:
            out[name] = (mod, float(mod.scaling["default"]))
    return out


def _apply_mode_scale(scale_map: dict, mode: str, sigma: float | None = None) -> None:
    if mode == "H0_Base":
        for mod, _ in scale_map.values():
            mod.scaling["default"] = 0.0
    elif mode == "H3_Full":
        for mod, orig in scale_map.values():
            mod.scaling["default"] = orig
    elif mode == "H2_Sigma_ge800":
        on = (sigma is not None and sigma >= 0.80)
        for mod, orig in scale_map.values():
            mod.scaling["default"] = orig if on else 0.0
    elif mode == "H1_Sigma_lt800":
        on = (sigma is not None and sigma < 0.80)
        for mod, orig in scale_map.values():
            mod.scaling["default"] = orig if on else 0.0
    elif mode in ("HD1_Early_0_17", "HD2_Mid_18_35", "HD3_Late_36_53"):
        lo, hi = {"HD1_Early_0_17": (0, 17), "HD2_Mid_18_35": (18, 35), "HD3_Late_36_53": (36, 53)}[mode]
        for name, (mod, orig) in scale_map.items():
            m = re.search(r"transformer_blocks\.(\d+)\.", name)
            bidx = int(m.group(1)) if m else -1
            mod.scaling["default"] = orig if (lo <= bidx <= hi) else 0.0
    else:
        raise ValueError(mode)


@torch.no_grad()
def stage_eval(args) -> int:
    from diffusers import HunyuanVideo15ImageToVideoPipeline
    from peft import PeftModel

    out_dir = args.out_dir / "eval"
    for sub in ("mp4", "strips", "keyframes", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    sel_pids = [0, 1, 2, 3]
    neg = "blurry, low quality, distorted, extra limbs, watermark, text"

    modes_main = ["H0_Base", "H1_Sigma_lt800", "H2_Sigma_ge800", "H3_Full"]
    modes_depth = ["HD1_Early_0_17", "HD2_Mid_18_35", "HD3_Late_36_53"]

    tasks = []
    for pid in sel_pids:
        cid = pairs[pid]["anime_id"]
        anime_p = Path(pairs[pid]["anime_path"])
        real_p = Path(pairs[pid]["real_path"])
        for rname, rprompt, domain in HY15_EVAL_REGIMES:
            img_p = real_p if domain == "real" else anime_p
            for m in modes_main:
                tasks.append((pid, cid, img_p, rname, rprompt, m))
            if rname in DEPTH_REGIMES:
                for m in modes_depth:
                    tasks.append((pid, cid, img_p, rname, rprompt, m))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (out_dir / "meta" / f"{t[0]:02d}_{t[3]}_{t[5]}.json").exists()]
    if not todo:
        print(f"[hy15 eval s{args.shard}] all {len(my_tasks)} clips done!", flush=True)
        return 0

    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.target_size = 512
    pipe.vae.enable_tiling()

    lora_dir = args.out_dir / f"lora_hy15_r{args.rank}"
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_dir)
    pipe.transformer.eval()
    scale_map = _collect_lora_scales(pipe.transformer)

    # Pre-encode all unique prompts & images on GPU, then offload encoders to CPU for blazing-fast GPU DiT inference!
    pipe.text_encoder.to("cuda", dtype=torch.bfloat16).eval()
    pipe.text_encoder_2.to("cuda", dtype=torch.bfloat16).eval()
    pipe.image_encoder.to("cuda", dtype=torch.bfloat16).eval()

    prompt_emb_cache = {}
    unique_prompts = sorted(set([t[4] for t in todo] + [neg]))
    for p in unique_prompts:
        pe, pm, pe2, pm2 = pipe.encode_prompt(prompt=p, device=torch.device("cuda"), dtype=torch.bfloat16, batch_size=1)
        prompt_emb_cache[p] = (pe.cpu(), pm.cpu(), pe2.cpu(), pm2.cpu())

    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    pipe.transformer.to("cuda", dtype=torch.bfloat16)
    pipe.vae.to("cuda", dtype=torch.bfloat16)

    state = {"mode": "H0_Base"}

    def _pre_hook(module, args_in, kwargs_in):
        t_arg = kwargs_in.get("timestep", None)
        sigma = float(t_arg[0].item()) / 1000.0 if t_arg is not None else None
        _apply_mode_scale(scale_map, state["mode"], sigma=sigma)

    hook_handle = pipe.transformer.register_forward_pre_hook(_pre_hook, with_kwargs=True)

    neg_pe, neg_pm, neg_pe2, neg_pm2 = [x.to("cuda") for x in prompt_emb_cache[neg]]

    for idx_t, (pid, cid, img_p, rname, rprompt, mode) in enumerate(todo):
        t_start = time.time()
        state["mode"] = mode
        im = Image.open(img_p).convert("RGB").resize((512, 512), Image.LANCZOS)
        pe, pm, pe2, pm2 = [x.to("cuda") for x in prompt_emb_cache[rprompt]]

        ridx = [r[0] for r in HY15_EVAL_REGIMES].index(rname)
        seed = 56000 + pid * 47 + ridx
        gen = torch.Generator(device="cuda").manual_seed(seed)

        frames = pipe(
            image=im,
            prompt=None,
            negative_prompt=None,
            prompt_embeds=pe,
            prompt_embeds_mask=pm,
            prompt_embeds_2=pe2,
            prompt_embeds_mask_2=pm2,
            negative_prompt_embeds=neg_pe,
            negative_prompt_embeds_mask=neg_pm,
            negative_prompt_embeds_2=neg_pe2,
            negative_prompt_embeds_mask_2=neg_pm2,
            num_frames=33,
            num_inference_steps=30,
            generator=gen,
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
        try:
            import subprocess, imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
            w0, h0 = pil_frames[0].size
            proc = subprocess.Popen(
                [ff, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w0}x{h0}", "-r", "16", "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                 str(out_dir / "mp4" / f"{key}.mp4")],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            for pf in pil_frames:
                proc.stdin.write(pf.convert("RGB").tobytes())
            proc.stdin.close()
            proc.wait()
        except Exception as exc:
            print(f"[hy15 mp4 warn] {key}: {exc}", flush=True)
        e31b._save_strip(
            pil_frames, out_dir / "strips" / f"{key}.png",
            f"HY1.5-54L | fc_{pid:02d} | {rname} | {mode} | dL_up={st['upper_face_luma_delta']:+.2f} dL_lo={st['lower_face_luma_delta']:+.2f} shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=160,
        )
        kf_dir = out_dir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        for fi, frac in enumerate([0.0, 0.25, 0.50, 0.75, 1.0], start=1):
            pil_frames[int(round(frac * (len(pil_frames) - 1)))].save(kf_dir / f"k{fi}.png")

        rec = {"pair_idx": pid, "char_id": cid, "regime": rname, "mode": mode, "seed": seed, "stats": st}
        (out_dir / "meta" / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[hy15 eval s{args.shard}] {idx_t+1}/{len(todo)} {key} ({time.time()-t_start:.1f}s) | "
            f"dL_up={st['upper_face_luma_delta']:+.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}",
            flush=True,
        )

    hook_handle.remove()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["encode_cache", "sign_check", "train", "eval"])
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp56_hy15_4th_arch"))
    ap.add_argument("--scale48-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=33)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    if args.stage == "encode_cache":
        return stage_encode_cache(args)
    if args.stage == "sign_check":
        return stage_sign_check(args)
    if args.stage == "train":
        return stage_train(args)
    if args.stage == "eval":
        return stage_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
