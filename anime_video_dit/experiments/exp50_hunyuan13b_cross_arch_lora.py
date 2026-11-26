#!/usr/bin/env python3
"""
Exp 50 (3rd Independent Flagship Architecture Weight-Level Cure: 13B Multimodal MM-DiT `HunyuanVideo-I2V`):

Proves BOTH our Clean Self-Bootstrapped 2D Anime Grammar Flow-Matching LoRA AND our High-Noise Semantic
Grammar Principle across a THIRD completely independent flagship Video DiT family:
  - Architecture: `hunyuanvideo-community/HunyuanVideo-I2V` (13B MM-DiT: 20 Dual-Stream + 40 Single-Stream
    Token-Replace blocks, LLaVA + CLIP dual text encoders, 16ch 4x8x8 3D-VAE, Embedded Guidance `6.0`,
    FlowMatch `shift=17.0`), which failed `0/20` on `S3, G1-G4` in Exp 37A!
  - Exact 50%/50% Step-Matched Causal Test (30 inference steps):
      * `M0_HY13B_Base`:          Frozen 13B HunyuanVideo-I2V (0/30 steps LoRA)
      * `M1_HY13B_LoRA_HighHalf`: LoRA ON strictly on FIRST 15/30 steps (`step_i < 15`), frozen base on last 15/30!
      * `M2_HY13B_LoRA_LowHalf`:  LoRA ON strictly on LAST 15/30 steps (`step_i >= 15`), frozen base on first 15/30!
      * `M3_HY13B_LoRA_Full`:     LoRA ON on all 30/30 steps
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

MODEL_ID = "hunyuanvideo-community/HunyuanVideo-I2V"

HY_EVAL_REGIMES = [
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


def _pils_to_vae_tensor(frames: list[Image.Image], size: int = 512) -> torch.Tensor:
    arr = np.stack([np.asarray(f.resize((size, size), Image.LANCZOS), dtype=np.float32) for f in frames], axis=0)
    # [T, H, W, 3] -> [1, 3, T, H, W] in [-1, 1]
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)
    return t / 127.5 - 1.0


@torch.no_grad()
def stage_encode_cache(args) -> int:
    from diffusers import HunyuanVideoImageToVideoPipeline

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
        print(f"[hy13b encode s{args.shard}] all {len(my_tasks)} clips cached!", flush=True)
        return 0

    pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(MODEL_ID, transformer=None, torch_dtype=torch.bfloat16)
    pipe.vae.to("cuda", dtype=torch.bfloat16).eval()
    pipe.vae.enable_tiling()
    pipe.text_encoder.to("cuda", dtype=torch.bfloat16).eval()
    pipe.text_encoder_2.to("cuda", dtype=torch.bfloat16).eval()
    sf = float(pipe.vae.config.scaling_factor)

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
            frames49 = [Image.open(Path(src_a) / f"f_{fi:03d}.png").convert("RGB").resize((args.size, args.size), Image.LANCZOS) for fi in range(49)]
            idxs = [round(k * (len(frames49) - 1) / (args.num_frames - 1)) for k in range(args.num_frames)]
            frames = [frames49[k] for k in idxs]
            orig_im = frames[0]

        vid_t = _pils_to_vae_tensor(frames, size=args.size).to("cuda", dtype=torch.bfloat16)
        img_t = vid_t[:, :, :1]
        x0 = pipe.vae.encode(vid_t).latent_dist.mode() * sf
        img_lat = pipe.vae.encode(img_t).latent_dist.mode() * sf

        pe, ppe, pam = pipe.encode_prompt(
            image=orig_im,
            prompt=prompt,
            device=torch.device("cuda"),
            max_sequence_length=256,
            image_embed_interleave=4,
        )

        rec = {
            "x0": x0.cpu().to(torch.bfloat16),
            "image_latents": img_lat.cpu().to(torch.bfloat16),
            "prompt_embeds": pe.cpu().to(torch.bfloat16),
            "pooled_prompt_embeds": ppe.cpu().to(torch.bfloat16),
            "prompt_attention_mask": pam.cpu().to(torch.bfloat16),
            "kind": kind,
            "char_id": cid,
            "variant": var,
        }
        torch.save(rec, cache_dir / f"{cid}__{var}.pt")
        if (idx_t + 1) % 5 == 0 or idx_t == len(todo) - 1:
            print(f"[hy13b encode s{args.shard}] {idx_t+1}/{len(todo)} | {cid}__{var} x0={tuple(x0.shape)} pe={tuple(pe.shape)}", flush=True)
    return 0


@torch.no_grad()
def stage_sign_check(args) -> int:
    from diffusers import HunyuanVideoTransformer3DModel

    pts = sorted((args.out_dir / "latent_cache").glob("*.pt"))
    rec = torch.load(pts[0], map_location="cuda")
    tr = HunyuanVideoTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda").eval()
    x0 = rec["x0"].float()
    img_lat = rec["image_latents"].float()
    eps = torch.randn_like(x0)
    sigma = 0.85
    xt_tail = (1.0 - sigma) * x0[:, :, 1:] + sigma * eps[:, :, 1:]
    inp = torch.cat([img_lat, xt_tail], dim=2).to(torch.bfloat16)
    t_tensor = torch.tensor([sigma * 1000.0], device="cuda", dtype=torch.bfloat16)
    guid = torch.tensor([6.0], device="cuda", dtype=torch.bfloat16) * 1000.0
    pred = tr(
        hidden_states=inp,
        timestep=t_tensor,
        encoder_hidden_states=rec["prompt_embeds"].to(torch.bfloat16),
        encoder_attention_mask=rec["prompt_attention_mask"].to(torch.bfloat16),
        pooled_projections=rec["pooled_prompt_embeds"].to(torch.bfloat16),
        guidance=guid,
        return_dict=False,
    )[0][:, :, 1:].float()
    v_pos = eps[:, :, 1:] - x0[:, :, 1:]
    v_neg = x0[:, :, 1:] - eps[:, :, 1:]
    c_pos = F.cosine_similarity(pred.flatten(), v_pos.flatten(), dim=0).item()
    c_neg = F.cosine_similarity(pred.flatten(), v_neg.flatten(), dim=0).item()
    m_pos = F.mse_loss(pred, v_pos).item()
    m_neg = F.mse_loss(pred, v_neg).item()
    print(f"[hy13b sign_check] pred vs (eps-x0): cos={c_pos:+.4f} mse={m_pos:.4f} | pred vs (x0-eps): cos={c_neg:+.4f} mse={m_neg:.4f}", flush=True)
    return 0


def stage_train(args) -> int:
    import torch.distributed as dist
    from diffusers import HunyuanVideoTransformer3DModel
    from peft import LoraConfig, get_peft_model

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world > 1:
        store = dist.FileStore(str(args.out_dir / "ddp_filestore"), world)
        dist.init_process_group(backend="nccl", store=store, rank=rank, world_size=world)

    pts = sorted((args.out_dir / "latent_cache").glob("*.pt"))
    assert len(pts) >= 50, f"Expected cached .pt files in {args.out_dir / 'latent_cache'}, found {len(pts)}"
    # Keep latent cache on CPU RAM (176 clips ~ 0.6 GB total!)
    dataset = [torch.load(p, map_location="cpu") for p in pts]
    if rank == 0:
        print(f"[hy13b train] Loaded {len(dataset)} clips | world={world} steps={args.steps} r={args.rank} lr={args.lr}", flush=True)

    tr = HunyuanVideoTransformer3DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16)
    tr.requires_grad_(False)
    tr.enable_gradient_checkpointing()
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0.0,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            "proj_mlp",
        ],
    )
    tr = get_peft_model(tr, lora_cfg).to(device)
    if rank == 0:
        tr.print_trainable_parameters()

    model = torch.nn.parallel.DistributedDataParallel(tr, device_ids=[local_rank], find_unused_parameters=False) if world > 1 else tr
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)

    rng = random.Random(2026 + rank * 97)
    torch.manual_seed(2026 + rank * 97)
    t0 = time.time()
    ema = None

    for step in range(1, args.steps + 1):
        # Cosine LR with 25-step linear warmup
        if step <= 25:
            lr_now = args.lr * step / 25.0
        else:
            prog = (step - 25) / max(args.steps - 25, 1)
            lr_now = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        rec = rng.choice(dataset)
        x0 = rec["x0"].to(device, dtype=torch.float32)
        img_lat = rec["image_latents"].to(device, dtype=torch.float32)
        pe = rec["prompt_embeds"].to(device, dtype=torch.bfloat16)
        ppe = rec["pooled_prompt_embeds"].to(device, dtype=torch.bfloat16)
        pam = rec["prompt_attention_mask"].to(device, dtype=torch.bfloat16)

        # Match HunyuanVideo-I2V's exact shift=17.0 schedule (75%) + broader mid/high noise (25%)
        if rng.random() < 0.75:
            u = rng.uniform(0.03, 0.995)
            sigma = (17.0 * u) / (1.0 + 16.0 * u)
        else:
            sigma = rng.uniform(0.35, 0.995)

        eps = torch.randn_like(x0)
        xt_tail = (1.0 - sigma) * x0[:, :, 1:] + sigma * eps[:, :, 1:]
        inp = torch.cat([img_lat, xt_tail], dim=2).to(torch.bfloat16)
        t_tensor = torch.tensor([sigma * 1000.0], device=device, dtype=torch.bfloat16)
        guid = torch.tensor([6.0], device=device, dtype=torch.bfloat16) * 1000.0

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(
                hidden_states=inp,
                timestep=t_tensor,
                encoder_hidden_states=pe,
                encoder_attention_mask=pam,
                pooled_projections=ppe,
                guidance=guid,
                return_dict=False,
            )[0]
            target = eps[:, :, 1:] - x0[:, :, 1:]
            loss = F.mse_loss(pred[:, :, 1:].float(), target.float())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        lv = float(loss.item())
        ema = lv if ema is None else 0.95 * ema + 0.05 * lv
        if rank == 0 and (step % 25 == 0 or step == 1 or step == args.steps):
            dt = time.time() - t0
            print(f"[hy13b train] step {step:4d}/{args.steps} | loss={lv:.4f} ema={ema:.4f} lr={lr_now:.2e} ({dt:.1f}s)", flush=True)

    if rank == 0:
        save_dir = args.out_dir / f"lora_hy13b_r{args.rank}"
        save_dir.mkdir(parents=True, exist_ok=True)
        raw = model.module if hasattr(model, "module") else model
        raw.save_pretrained(save_dir)
        print(f"[hy13b train] Saved LoRA -> {save_dir} in {time.time()-t0:.1f}s!", flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


@torch.no_grad()
def stage_eval(args) -> int:
    from diffusers import HunyuanVideoImageToVideoPipeline
    from peft import PeftModel

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_dir = args.out_dir / "eval"
    for sub in ("meta", "strips", "mp4", "keyframes"):
        (eval_dir / sub).mkdir(parents=True, exist_ok=True)

    modes = [
        ("M0_HY13B_Base", "off"),
        ("M1_HY13B_LoRA_HighHalf", "high_half"),
        ("M2_HY13B_LoRA_LowHalf", "low_half"),
        ("M3_HY13B_LoRA_Full", "on"),
    ]
    tasks = []
    for pid in range(args.n_test_chars):
        p = pairs[pid]
        for ri, (rname, rprompt, dom) in enumerate(HY_EVAL_REGIMES):
            img_path = Path(p["anime_path"]) if dom == "anime" else Path(p["real_path"])
            cid = p["anime_id"] if dom == "anime" else p["real_id"]
            seed = 9700 + pid * 47 + ri
            for mname, mkind in modes:
                tasks.append((pid, dom, img_path, cid, rname, rprompt, mname, mkind, seed))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (eval_dir / "meta" / f"{t[0]:02d}_{t[4]}_{t[6]}.json").exists()]
    if not todo:
        print(f"[hy13b eval s{args.shard}] all complete!", flush=True)
        return 0

    pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    lora_dir = args.out_dir / f"lora_hy13b_r{args.rank}"
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_dir)
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    n_steps = 30
    half_step = n_steps // 2  # Exact 15/30 vs 15/30 step-matched causal split!

    for pid, dom, img_path, cid, rname, rprompt, mname, mkind, seed in todo:
        if mkind == "off":
            pipe.transformer.disable_adapter_layers()
            cb = None
        elif mkind == "on":
            pipe.transformer.enable_adapter_layers()
            cb = None
        elif mkind == "high_half":
            # ON for steps 0..14 (first 15/30 high-noise steps); callback fires AFTER step_i, so disable after step 14!
            pipe.transformer.enable_adapter_layers()
            def _cb_hi(p_self, step_i, t_val, cb_kwargs):
                if step_i + 1 >= half_step:
                    p_self.transformer.disable_adapter_layers()
                return cb_kwargs
            cb = _cb_hi
        else:
            # low_half: OFF for steps 0..14, ON for steps 15..29 (last 15/30 low-noise steps!)
            pipe.transformer.disable_adapter_layers()
            def _cb_lo(p_self, step_i, t_val, cb_kwargs):
                if step_i + 1 >= half_step:
                    p_self.transformer.enable_adapter_layers()
                return cb_kwargs
            cb = _cb_lo

        im = Image.open(img_path).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=im,
            prompt=rprompt,
            height=args.size,
            width=args.size,
            num_frames=args.num_frames,
            num_inference_steps=n_steps,
            guidance_scale=6.0,
            generator=gen,
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

        key = f"{pid:02d}_{rname}_{mname}"
        e36._save_single_mp4(pil_frames, eval_dir / "mp4" / f"{key}.mp4", fps=15)
        e31b._save_strip(
            pil_frames, eval_dir / "strips" / f"{key}.png",
            f"{mname} | {pid:02d}_{rname} ({cid}) | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=170,
        )
        kf_dir = eval_dir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        rec = {"pair_idx": pid, "idx": pid, "domain": dom, "char_id": cid, "regime": rname, "mode": mname, "seed": seed, "stats": st}
        (eval_dir / "meta" / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[hy13b eval s{args.shard}] {key:44s} | dL_g={st['global_luma_delta']:+.3f} dL_up={st['upper_face_luma_delta']:+.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}",
            flush=True,
        )
    return 0


def stage_summary(args) -> int:
    eval_dir = args.out_dir / "eval"
    modes = ["M0_HY13B_Base", "M2_HY13B_LoRA_LowHalf", "M1_HY13B_LoRA_HighHalf", "M3_HY13B_LoRA_Full"]
    summary = {}
    for rname, _, _ in HY_EVAL_REGIMES:
        summary[rname] = {}
        for m in modes:
            recs = []
            for pid in range(args.n_test_chars):
                p = eval_dir / "meta" / f"{pid:02d}_{rname}_{m}.json"
                if p.exists():
                    recs.append(json.loads(p.read_text(encoding="utf-8"))["stats"])
            if recs:
                summary[rname][m] = {
                    "n": len(recs),
                    "dL_glob": round(float(np.mean([r["global_luma_delta"] for r in recs])), 4),
                    "dL_hair": round(float(np.mean([r["hair_luma_delta"] for r in recs])), 4),
                    "dL_up": round(float(np.mean([r["upper_face_luma_delta"] for r in recs])), 4),
                    "dL_low": round(float(np.mean([r["lower_face_luma_delta"] for r in recs])), 4),
                    "shR": round(float(np.mean([r["sharp_ratio_vs_inp"] for r in recs])), 4),
                }
    (args.out_dir / "summary_exp50_hy13b.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for pid in [0, 2]:
        for rname, _, _ in HY_EVAL_REGIMES:
            ps = [eval_dir / "strips" / f"{pid:02d}_{rname}_{m}.png" for m in modes if (eval_dir / "strips" / f"{pid:02d}_{rname}_{m}.png").exists()]
            if len(ps) >= 2:
                e31b._stack_strips(ps, args.out_dir / f"qc50_hy13b_{pid:02d}_{rname}.png")
    print(f"[exp50] Saved summary & QC sheets -> {args.out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["encode_cache", "sign_check", "train", "eval", "summary"])
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp50_hy13b_cross_arch"))
    ap.add_argument("--scale48-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=33)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-test-chars", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "encode_cache":
        return stage_encode_cache(args)
    if args.stage == "sign_check":
        return stage_sign_check(args)
    if args.stage == "train":
        return stage_train(args)
    if args.stage == "eval":
        return stage_eval(args)
    return stage_summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
