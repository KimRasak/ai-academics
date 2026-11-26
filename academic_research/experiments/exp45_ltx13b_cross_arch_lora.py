#!/usr/bin/env python3
"""
Exp 45 (Cross-Architecture Second Flagship Weight-Level Cure on Non-MoE 13B `LTX-Video-0.9.7-dev`):

Proves that our Clean Self-Bootstrapped 2D Anime Grammar Flow-Matching LoRA (`exp39v2` 32 training chars `fc_08..39`,
holding out `fc_00..07` unseen!) AND our High-Noise Semantic Grammar Principle generalize to a SECOND completely
independent flagship Video DiT architecture (`Lightricks/LTX-Video-0.9.7-dev`: 13B 48-layer monolithic non-MoE
Spatiotemporal DiT, T5-XXL, 128ch 3D-VAE — which failed `0/20` on `S3, G1-G4` in Exp 37B!):

  Stage 1 (`--stage encode_cache`):
    Pre-encodes 160 Clean v2 2D Anime Grammar clips (`49f x 512x512` -> `7 x 16 x 16 = 1792` packed 128ch tokens!)
    + 16 clean base LTX-13B 3D-motion anchor clips with exact LTX VAE normalization & T5-XXL embeddings (~2 min!).
  Stage 2 (`--stage train`):
    4-GPU FileStore+NCCL DDP Rank-16 Flow-Matching LoRA (`~50M` params, `300` steps x batch 4, ~5 min on 4xH100!)
    trained with exact LTX I2V conditioning (`conditioning_mask` frame-0 t=0 + noisy frames t>0, `v = eps - x0`).
  Stage 3 (`--stage eval`):
    Evaluates on 4 UNSEEN held-out characters (`fc_00..fc_03`) x 7 regimes (`S3_NonBioAffect`, `G1_HighlightOff_Only`,
    `G2_ForeheadShadow_Clean`, `G3_AngerVein_Pop`, `G4_SweatDrop_Slide`, `C1_Comp_HeadTurn_Shadow`, `S1_Safety_RealHeadTurn`)
    across 3 modes:
      - `M0_LTX13B_Base`:       Frozen 13B LTX-Video-0.9.7
      - `M1_LTX13B_LoRA_High`:  LoRA enabled ONLY on high-noise steps (`t >= 550`), frozen base LTX on low-noise steps!
      - `M2_LTX13B_LoRA_Full`:  LoRA enabled on all denoising steps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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


LTX_EVAL_REGIMES = [
    r for r in e40.EVAL_REGIMES
    if r[0] in (
        "S3_NonBioAffect",
        "G1_HighlightOff_Only",
        "G2_ForeheadShadow_Clean",
        "G3_AngerVein_Pop",
        "G4_SweatDrop_Slide",
        "C1_Comp_HeadTurn_Shadow",
        "S1_Safety_RealHeadTurn",
    )
]


@torch.no_grad()
def stage_encode_cache(args) -> int:
    from diffusers import LTXConditionPipeline

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

    # Reuse the 16 clean 3D-motion anchor videos already on disk from Exp 39!
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
        print(f"[ltx encode s{args.shard}] all complete!", flush=True)
        return 0

    pipe = LTXConditionPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev",
        transformer=None,
        torch_dtype=torch.bfloat16,
    )
    pipe.vae.to("cuda", dtype=torch.bfloat16).eval()
    pipe.text_encoder.to("cuda", dtype=torch.bfloat16).eval()

    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for t in todo:
        pr = t[6]
        if pr not in prompt_cache:
            pe, pam, _, _ = pipe.encode_prompt(
                prompt=pr, do_classifier_free_guidance=False, num_videos_per_prompt=1,
                max_sequence_length=256, device=torch.device("cuda"), dtype=torch.bfloat16,
            )
            prompt_cache[pr] = (pe.cpu(), pam.cpu())

    # Precompute shared video_coords for 49f x 512x512 (7 x 16 x 16 = 1792 tokens)
    num_lat_f = (49 - 1) // pipe.vae_temporal_compression_ratio + 1  # 7
    lat_h = 512 // pipe.vae_spatial_compression_ratio  # 16
    lat_w = 512 // pipe.vae_spatial_compression_ratio  # 16
    video_coords = pipe._prepare_video_ids(
        1, num_lat_f, lat_h, lat_w,
        patch_size_t=pipe.transformer_temporal_patch_size,
        patch_size=pipe.transformer_spatial_patch_size,
        device=torch.device("cuda"),
    )
    video_coords = pipe._scale_video_ids(
        video_coords,
        scale_factor=pipe.vae_spatial_compression_ratio,
        scale_factor_t=pipe.vae_temporal_compression_ratio,
        frame_index=0,
        device=torch.device("cuda"),
    ).float()
    video_coords[:, 0] = video_coords[:, 0] * (1.0 / 25.0)
    video_coords_cpu = video_coords.cpu()

    for kind, cid, var, src_a, src_b, eye_p, prompt in todo:
        if kind == "grammar":
            orig_im = Image.open(src_a).convert("RGB")
            var_im = Image.open(src_b).convert("RGB")
            eye_im = Image.open(eye_p).convert("RGB") if eye_p else None
            if var in ("eye_highlight_off", "forehead_shadow_on", "dead_eyes_and_shadow"):
                pil_frames = e39v2._synthesize_clean_clip(orig_im, var_im, var, T=49, size=512, eye_im=eye_im)
            else:
                pil_frames = e39._synthesize_grammar_clip(orig_im, var_im, var, T=49, size=512, eye_im=eye_im)
        else:
            pil_frames = [Image.open(src_a / f"f_{fi:03d}.png").convert("RGB").resize((512, 512), Image.LANCZOS) for fi in range(49)]

        arr = np.stack([np.asarray(f, dtype=np.float32) / 127.5 - 1.0 for f in pil_frames], axis=0)
        vid = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).to("cuda", dtype=torch.bfloat16)  # [1, 3, 49, 512, 512]
        x0_lat = pipe.vae.encode(vid).latent_dist.mode()
        x0_lat = pipe._normalize_latents(x0_lat, pipe.vae.latents_mean, pipe.vae.latents_std)  # [1, 128, 7, 16, 16]

        # Frame-0 conditioning latent (single image encoded identically to LTXConditionPipeline!)
        img0 = vid[:, :, :1]
        y0_lat = pipe.vae.encode(img0).latent_dist.mode()
        y0_lat = pipe._normalize_latents(y0_lat, pipe.vae.latents_mean, pipe.vae.latents_std)  # [1, 128, 1, 16, 16]

        x0_packed = pipe._pack_latents(x0_lat, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size)  # [1, 1792, 128]
        y0_packed = pipe._pack_latents(y0_lat, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size)  # [1, 256, 128]
        pe, pam = prompt_cache[prompt]

        item = {
            "char_id": cid, "variant": var, "kind": kind, "prompt": prompt,
            "x0_packed": x0_packed.to(torch.bfloat16).cpu(),
            "y0_packed": y0_packed.to(torch.bfloat16).cpu(),
            "video_coords": video_coords_cpu,
            "prompt_embeds": pe,
            "prompt_attention_mask": pam,
        }
        torch.save(item, cache_dir / f"{cid}__{var}.pt")
        print(f"[ltx encode s{args.shard}] cached {cid}__{var}.pt | x0={tuple(item['x0_packed'].shape)}", flush=True)
    return 0


def stage_train(args) -> int:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from diffusers import LTXVideoTransformer3DModel
    from peft import LoraConfig, get_peft_model

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method=os.environ.get("DDP_INIT_METHOD", "env://"), rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")

    save_dir = args.out_dir / f"lora_ltx13b_r{args.rank}"
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "adapter_config.json").exists() and not args.overwrite:
        if rank == 0:
            print(f"[ltx train] already trained at {save_dir}, skipping!", flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return 0

    cache_files = sorted((args.out_dir / "latent_cache").glob("*.pt"))
    grammar_items, anchor_items = [], []
    for p in cache_files:
        it = torch.load(p, map_location="cpu", weights_only=True)
        (anchor_items if it.get("kind") == "anchor" else grammar_items).append(it)
    if rank == 0:
        print(f"[ltx train] world={world_size} loaded {len(grammar_items)} grammar + {len(anchor_items)} anchors", flush=True)

    transformer = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev", subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.rank, lora_dropout=0.05, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_cfg)
    transformer.train()
    if rank == 0:
        nt = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        print(f"[ltx train] trainable LoRA params: {nt/1e6:.2f}M (rank={args.rank})", flush=True)

    model = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) if world_size > 1 else transformer
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    rng = random.Random(2026 + rank * 101)
    torch.manual_seed(2026 + rank * 101)
    n_f0 = 256  # 16x16 frame-0 tokens

    t0_wall = time.time()
    loss_ema = None
    history = []

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        if step <= 30:
            lr_now = args.lr * (step / 30.0)
        else:
            prog = (step - 30) / max(args.steps - 30, 1)
            lr_now = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        step_loss = 0.0
        for _ in range(args.grad_accum):
            if anchor_items and rng.random() < 0.20:
                it = rng.choice(anchor_items)
            else:
                it = rng.choice(grammar_items)

            x0 = it["x0_packed"].to(device, dtype=torch.bfloat16)  # [1, 1792, 128]
            y0 = it["y0_packed"].to(device, dtype=torch.bfloat16)  # [1, 256, 128]
            vc = it["video_coords"].to(device, dtype=torch.float32)
            pe = it["prompt_embeds"].to(device, dtype=torch.bfloat16)
            pam = it["prompt_attention_mask"].to(device)

            # Logit-normal / uniform flow-matching timestep sampling favoring structural high-noise + full range
            u = rng.uniform(0.05, 0.99)
            sigma = float(u)
            eps = torch.randn_like(x0)
            x_t = ((1.0 - sigma) * x0.float() + sigma * eps.float()).to(torch.bfloat16)

            # Exact LTX I2V frame-0 conditioning: overwrite first 256 tokens with y0 + tiny cond noise & t=0!
            cond_noise_s = rng.uniform(0.0, 0.05)
            x_t[:, :n_f0] = ((1.0 - cond_noise_s) * y0.float() + cond_noise_s * torch.randn_like(y0).float()).to(torch.bfloat16)
            timestep = torch.full((1, x0.shape[1]), sigma * 1000.0, device=device, dtype=torch.float32)
            timestep[:, :n_f0] = 0.0

            v_pred = model(
                hidden_states=x_t,
                encoder_hidden_states=pe,
                timestep=timestep,
                encoder_attention_mask=pam,
                video_coords=vc,
                return_dict=False,
            )[0]

            # Flow-matching loss strictly on generated frames (tokens n_f0:) with v_target = eps - x0!
            v_target = (eps.float() - x0.float())[:, n_f0:]
            loss = F.mse_loss(v_pred[:, n_f0:].float(), v_target) / args.grad_accum
            loss.backward()
            step_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

        loss_ema = step_loss if loss_ema is None else (0.9 * loss_ema + 0.1 * step_loss)
        if rank == 0 and (step == 1 or step % 25 == 0 or step == args.steps):
            el = time.time() - t0_wall
            print(f"[ltx train] step {step:4d}/{args.steps} | loss={step_loss:.4f} ema={loss_ema:.4f} | lr={lr_now:.2e} | elapsed={el:.0f}s ({el/step:.2f}s/step)", flush=True)
            history.append({"step": step, "loss": round(step_loss, 5), "ema": round(loss_ema, 5)})

    if rank == 0:
        unwrapped = model.module if world_size > 1 else model
        unwrapped.save_pretrained(save_dir)
        (save_dir / "train_log.json").write_text(json.dumps({"steps": args.steps, "rank": args.rank, "history": history}, indent=2))
        print(f"[ltx train] SAVED LoRA adapter to {save_dir}!", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


@torch.no_grad()
def stage_eval(args) -> int:
    from diffusers import LTXConditionPipeline
    from peft import PeftModel

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    sel_pids = list(range(args.n_test_chars))
    modes = ["M0_LTX13B_Base", "M1_LTX13B_LoRA_High", "M2_LTX13B_LoRA_Full"]

    tasks = []
    for pid in sel_pids:
        p = pairs[pid]
        for ri, (rname, rprompt, dom) in enumerate(LTX_EVAL_REGIMES):
            img_path = Path(p["anime_path"]) if dom == "anime" else Path(p["real_path"])
            cid = p["anime_id"] if dom == "anime" else p["real_id"]
            for mode in modes:
                tasks.append((pid, dom, img_path, cid, rname, rprompt, mode, 9500 + pid * 41 + ri))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    out_e = args.out_dir / "eval"
    for d in [out_e / "meta", out_e / "strips", out_e / "mp4", out_e / "keyframes"]:
        d.mkdir(parents=True, exist_ok=True)
    todo = [t for t in my_tasks if not (out_e / "meta" / f"{t[0]:02d}_{t[4]}_{t[6]}.json").exists()]
    if not todo:
        print(f"[ltx eval s{args.shard}] all complete!", flush=True)
        return 0

    pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.out_dir / f"lora_ltx13b_r{args.rank}")
    pipe.enable_model_cpu_offload()

    for pid, dom, img_path, cid, rname, rprompt, mode, seed in todo:
        if mode == "M0_LTX13B_Base":
            pipe.transformer.disable_adapter_layers()
            cb = None
        elif mode == "M2_LTX13B_LoRA_Full":
            pipe.transformer.enable_adapter_layers()
            cb = None
        else:
            # High-Noise-Only inference gating on a MONOLITHIC non-MoE 13B DiT!
            pipe.transformer.enable_adapter_layers()
            def _high_only_cb(p_self, step_i, t_val, cb_kwargs):
                tv = float(t_val.item()) if hasattr(t_val, "item") else float(t_val)
                if tv < 550.0:
                    p_self.transformer.disable_adapter_layers()
                else:
                    p_self.transformer.enable_adapter_layers()
                return cb_kwargs
            cb = _high_only_cb

        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=im, prompt=rprompt,
            negative_prompt="blurry, low quality, distorted, extra limbs, watermark, text",
            height=512, width=512, num_frames=49, num_inference_steps=36,
            guidance_scale=3.0, generator=gen,
            callback_on_step_end=cb,
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
        e36._save_single_mp4(pil_frames, out_e / "mp4" / f"{key}.mp4", fps=16)
        e31b._save_strip(
            pil_frames, out_e / "strips" / f"{key}.png",
            f"{mode} | {pid:02d}_{rname} ({cid}) | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=170,
        )
        kf_dir = out_e / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        (out_e / "meta" / f"{key}.json").write_text(json.dumps({
            "pair_idx": pid, "domain": dom, "char_id": cid, "regime": rname, "mode": mode, "seed": seed, "stats": st
        }, indent=2), encoding="utf-8")
        print(f"[ltx eval s{args.shard}] {key:42s} | dL_glob={st['global_luma_delta']:+.3f} dL_up={st['upper_face_luma_delta']:+.3f} dL_low={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


def stage_stack(args) -> int:
    out_e = args.out_dir / "eval"
    for pid in range(args.n_test_chars):
        for rname, _, _ in LTX_EVAL_REGIMES:
            ps = [out_e / "strips" / f"{pid:02d}_{rname}_{m}.png" for m in ["M0_LTX13B_Base", "M1_LTX13B_LoRA_High", "M2_LTX13B_LoRA_Full"]]
            ps = [p for p in ps if p.exists()]
            if len(ps) >= 2:
                e31b._stack_strips(ps, out_e / f"cmp45_ltx13b_{pid:02d}_{rname}.png")
    print(f"Stacked cmp45_ltx13b_*.png in {out_e}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["encode_cache", "train", "eval", "stack"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n-test-chars", type=int, default=4)
    ap.add_argument("--scale48-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp45_ltx13b_cross_arch_lora"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "encode_cache":
        return stage_encode_cache(args)
    if args.stage == "train":
        return stage_train(args)
    if args.stage == "eval":
        return stage_eval(args)
    return stage_stack(args)


if __name__ == "__main__":
    raise SystemExit(main())
