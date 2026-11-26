#!/usr/bin/env python3
"""
Exp 40 (Solving Flagship 27B MoE Physical World-Model Literalism — Real 14B MoE LoRA Training & 2x2 Causal Expert Ablation):

Directly trains Rank-16 Rectified Flow-Matching LoRAs on the TWO disjoint 14B MoE experts of `Wan2.2-I2V-A14B` (27B MoE):
  - `high`: `transformer`   (14B High-Noise Expert, exact shifted domain u in [0.75, 0.998] -> sigma in [0.90, 0.999])
  - `low`:  `transformer_2` (14B Low-Noise Expert,  exact shifted domain u in [0.01, 0.75)  -> sigma in [0.03, 0.90))

Using 100% pre-encoded clean synthetic 2D Anime Grammar latents (32 training chars `fc_08..fc_39` x 5 tropes + 20% clean
base-motion anti-forgetting anchors from Exp 39; 16 test chars `fc_00..fc_07` & `fc_40..fc_47` STRICTLY HELD OUT!).

Because the two 14B experts operate on strictly disjoint timestep domains, training `LoRA_High` and `LoRA_Low` cleanly
gives us an exact 2x2 Causal MoE Specialization Ablation at inference (`--stage eval_2x2`):
  1. `M0_Base27B`     (No LoRA — baseline flagship 27B MoE)
  2. `M1_LoRA_High`   (LoRA ONLY on 14B High-Noise Expert `t >= 900`)
  3. `M2_LoRA_Low`    (LoRA ONLY on 14B Low-Noise Expert `t < 900`)
  4. `M3_LoRA_Dual`   (Matched Dual-Expert LoRA on BOTH 14B Experts — OURS!)

Evaluated ZERO-SHOT on UNSEEN held-out characters across:
  - All 5 2D Anime Grammar Regimes (`S3_NonBioAffect`, `G1_HighlightOff_Only`, `G2_ForeheadShadow_Clean`, `G3_AngerVein_Pop`, `G4_SweatDrop_Slide`)
  - 2 UNSEEN Compositional Motion+Grammar Prompts (`C1_HeadTurn_ForeheadShadow`, `C2_HeadTurn_AngerVein` — NEVER seen in training!)
  - 2 Zero-Forgetting Safety Controls (`S1_HeadTurn` anime + `S1_HeadTurn` Real Human!)
"""

from __future__ import annotations

import argparse
import json
import math
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


EVAL_REGIMES = [
    (
        "S3_NonBioAffect",
        "The character's bright eye highlights suddenly vanish leaving hollow dark lifeless eyes while a sharp ominous dark shadow falls across their upper forehead, yet their mouth stays frozen in a calm eerie smile, 2D Japanese TV anime cel shading.",
        "anime",
    ),
    (
        "G1_HighlightOff_Only",
        "The character stays still in the exact same bright even studio lighting with unchanged bright hair, clothes, and gentle smile, while only the small white specular highlight dots inside both eyes smoothly fade away leaving flat matte dull irises, clean 2D Japanese TV anime.",
        "anime",
    ),
    (
        "G2_ForeheadShadow_Clean",
        "In bright even studio daylight, a stylized flat slate-blue 2D anime gloom shadow band with fine vertical ink lines appears strictly across the character's upper forehead and brow while the character keeps their exact gentle smile, bright hair color, bright cheeks, clothes, and bright white background completely unchanged.",
        "anime",
    ),
    (
        "G3_AngerVein_Pop",
        "In bright flat 2D Japanese anime style, the character holds a polite closed-mouth smile as a classic stylized red cross-shaped anime anger vein symbol pops onto their upper forehead temple and pulses gently, everything else stays bright, clean, and consistent.",
        "anime",
    ),
    (
        "G4_SweatDrop_Slide",
        "In bright flat 2D Japanese anime cel shading, the character gives an awkward nervous smile as a single large stylized translucent blue comic sweat drop appears beside their temple and slides gently downward, clean crisp lineart.",
        "anime",
    ),
    (
        "C1_Comp_HeadTurn_Shadow",
        "The character smoothly turns their head from front view to a three-quarter side profile with a calm eerie smile while their white eye highlights vanish and a translucent indigo 2D anime gloom shadow falls strictly across their upper forehead, bright hair and background unchanged.",
        "anime",
    ),
    (
        "C2_Comp_HeadTurn_Vein",
        "The character smoothly turns their head to a three-quarter profile holding a polite smile as a classic red cross-shaped 2D anime anger vein symbol pops onto their temple and pulses, bright clean 2D TV anime.",
        "anime",
    ),
    (
        "S1_Safety_AnimeHeadTurn",
        "The character smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent appearance.",
        "anime",
    ),
    (
        "S1_Safety_RealHeadTurn",
        "The character smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent appearance.",
        "real",
    ),
]


def stage_train(args) -> int:
    import os
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from diffusers import WanTransformer3DModel
    from peft import LoraConfig, get_peft_model

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        init_m = os.environ.get("DDP_INIT_METHOD", "env://")
        dist.init_process_group(backend="nccl", init_method=init_m, rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")

    subfolder = "transformer" if args.expert == "high" else "transformer_2"
    save_dir = args.out_dir / f"lora_{args.expert}_r{args.rank}"
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "adapter_config.json").exists() and not args.overwrite:
        if rank == 0:
            print(f"[train {args.expert}] already trained at {save_dir}, skipping!", flush=True)
        if world_size > 1:
            dist.destroy_process_group()
        return 0

    # Load pre-encoded latent cache into CPU RAM (tiny ~650 MB!)
    cache_files = sorted(args.cache_dir.glob("*.pt"))
    assert len(cache_files) > 0, f"No cache files found in {args.cache_dir}"
    grammar_items, anchor_items = [], []
    for p in cache_files:
        it = torch.load(p, map_location="cpu", weights_only=True)
        if it.get("kind") == "anchor":
            anchor_items.append(it)
        else:
            grammar_items.append(it)
    if rank == 0:
        print(f"[train {args.expert}] world={world_size} loaded {len(grammar_items)} grammar + {len(anchor_items)} anchor clips", flush=True)

    transformer = WanTransformer3DModel.from_pretrained(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        subfolder=subfolder,
        torch_dtype=torch.bfloat16,
    ).to(device)
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0.05,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_cfg)
    transformer.train()
    n_train = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    if rank == 0:
        print(f"[train {args.expert}] trainable LoRA params: {n_train/1e6:.2f}M (rank={args.rank})", flush=True)

    model_wrap = DDP(transformer, device_ids=[local_rank], find_unused_parameters=False) if world_size > 1 else transformer
    opt = torch.optim.AdamW(
        [p for p in model_wrap.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    seed_base = 42 + (0 if args.expert == "high" else 1000) + rank * 131
    rng = random.Random(seed_base)
    torch.manual_seed(seed_base)

    total_opt_steps = args.steps
    grad_accum = args.grad_accum
    warmup = min(30, total_opt_steps // 10)
    flow_shift = 3.0  # exact Wan2.2-I2V-A14B scheduler flow_shift!

    t0 = time.time()
    loss_hist = []
    running_loss = 0.0

    for step in range(1, total_opt_steps + 1):
        if step <= warmup:
            lr_mult = step / max(warmup, 1)
        else:
            prog = (step - warmup) / max(total_opt_steps - warmup, 1)
            lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = args.lr * lr_mult

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            if anchor_items and rng.random() < 0.20:
                it = rng.choice(anchor_items)
            else:
                it = rng.choice(grammar_items)

            x0 = it["x0"].to(device, dtype=torch.float32)
            cond = it["condition"].to(device, dtype=torch.float32)
            pe = it["prompt_embeds"].to(device, dtype=torch.bfloat16)

            if args.expert == "high":
                u = rng.uniform(0.75, 0.998)
            else:
                u = rng.uniform(0.015, 0.749)
            sigma = (flow_shift * u) / (1.0 + (flow_shift - 1.0) * u)
            t_val = torch.tensor([sigma * 1000.0], device=device, dtype=torch.float32)

            noise = torch.randn_like(x0)
            xt = (1.0 - sigma) * x0 + sigma * noise
            v_target = noise - x0

            model_in = torch.cat([xt, cond], dim=1).to(torch.bfloat16)
            v_pred = model_wrap(
                hidden_states=model_in,
                timestep=t_val,
                encoder_hidden_states=pe,
                return_dict=False,
            )[0].float()

            loss = F.mse_loss(v_pred[:, :, 1:], v_target[:, :, 1:]) + 0.1 * F.mse_loss(v_pred[:, :, :1], v_target[:, :, :1])
            (loss / grad_accum).backward()
            step_loss += float(loss.item()) / grad_accum

        torch.nn.utils.clip_grad_norm_(model_wrap.parameters(), 1.0)
        opt.step()
        running_loss = 0.9 * running_loss + 0.1 * step_loss if step > 1 else step_loss
        loss_hist.append(round(step_loss, 5))

        if rank == 0 and (step % 25 == 0 or step == 1 or step == total_opt_steps):
            el = time.time() - t0
            print(
                f"[train {args.expert}] step {step:4d}/{total_opt_steps} | loss={step_loss:.4f} ema={running_loss:.4f} | "
                f"lr={opt.param_groups[0]['lr']:.2e} | elapsed={el:.0f}s ({el/step:.2f}s/step)",
                flush=True,
            )

    if rank == 0:
        transformer.save_pretrained(save_dir)
        (save_dir / "train_log.json").write_text(
            json.dumps({"expert": args.expert, "rank": args.rank, "steps": total_opt_steps, "world_size": world_size, "lr": args.lr, "final_ema": running_loss, "loss_hist": loss_hist}, indent=2),
            encoding="utf-8",
        )
        print(f"[train {args.expert}] SAVED LoRA adapter to {save_dir}!", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


@torch.no_grad()
def stage_eval_2x2(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from peft import PeftModel

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Strictly HELD-OUT UNSEEN characters (`fc_00..fc_07` — NEVER touched in Exp 39 training!)
    sel_pids = list(range(args.n_test_chars))  # default 4 or 8 unseen chars (0..7)
    modes = ["M0_Base27B", "M1_LoRA_High", "M2_LoRA_Low", "M3_LoRA_Dual"]

    tasks = []
    for pid in sel_pids:
        p = pairs[pid]
        for rname, rprompt, dom in EVAL_REGIMES:
            img_path = Path(p["anime_path"]) if dom == "anime" else Path(p["real_path"])
            cid = p["anime_id"] if dom == "anime" else p["real_id"]
            for mode in modes:
                tasks.append((pid, dom, img_path, cid, rname, rprompt, mode))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    out_e = args.out_dir / "eval_2x2"
    meta_dir, strips_dir, mp4_dir = out_e / "meta", out_e / "strips", out_e / "mp4"
    for d in [meta_dir, strips_dir, mp4_dir]:
        d.mkdir(parents=True, exist_ok=True)
    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[4]}_{t[6]}.json").exists()]
    if not todo:
        print(f"[eval_2x2 s{args.shard}] all complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)

    # Wrap BOTH 14B experts with their trained PEFT LoRA adapters and toggle enable/disable cleanly!
    high_dir = args.out_dir / f"lora_high_r{args.rank}"
    low_dir = args.out_dir / f"lora_low_r{args.rank}"
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, high_dir)
    pipe.transformer_2 = PeftModel.from_pretrained(pipe.transformer_2, low_dir)
    pipe.enable_model_cpu_offload()

    for pid, dom, img_path, cid, rname, rprompt, mode in todo:
        # Toggle exact 2x2 MoE expert adapters!
        if mode == "M0_Base27B":
            pipe.transformer.disable_adapter_layers()
            pipe.transformer_2.disable_adapter_layers()
        elif mode == "M1_LoRA_High":
            pipe.transformer.enable_adapter_layers()
            pipe.transformer_2.disable_adapter_layers()
        elif mode == "M2_LoRA_Low":
            pipe.transformer.disable_adapter_layers()
            pipe.transformer_2.enable_adapter_layers()
        elif mode == "M3_LoRA_Dual":
            pipe.transformer.enable_adapter_layers()
            pipe.transformer_2.enable_adapter_layers()

        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        seed = 9400 + pid * 43 + [r[0] for r in EVAL_REGIMES].index(rname)
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
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))
        st.update(e37a._grammar_diagnostics(np_frames, inp_u8))

        key = f"{pid:02d}_{rname}_{mode}"
        e36._save_single_mp4(pil_frames, mp4_dir / f"{key}.mp4", fps=16)
        e31b._save_strip(
            pil_frames,
            strips_dir / f"{key}.png",
            f"{mode:14s} | {pid:02d}_{rname} ({cid}) | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6,
            thumb_h=170,
        )
        kf_dir = out_e / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for ki in range(6):
            pil_frames[round(ki * (T - 1) / 5)].save(kf_dir / f"k{ki}.png")

        rec = {"pair_idx": pid, "domain": dom, "char_id": cid, "regime": rname, "mode": mode, "stats": st}
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[eval_2x2 s{args.shard}] {key:42s} | dL(g={st['global_luma_delta']:+.3f} h={st['hair_luma_delta']:+.3f} up={st['upper_face_luma_delta']:+.3f} lo={st['lower_face_luma_delta']:+.3f}) shR={st['sharp_ratio_vs_inp']:.3f}",
            flush=True,
        )
    return 0


def stage_summarize(args) -> int:
    out_e = args.out_dir / "eval_2x2"
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_e / "meta").glob("*.json"))]
    print(f"\n=== EXP 40 2x2 MoE EXPERT LoRA SUMMARY (n={len(recs)} clips on UNSEEN Held-Out Characters!) ===", flush=True)
    for rname, _, _ in EVAL_REGIMES:
        sub_r = [r for r in recs if r["regime"] == rname]
        if not sub_r: continue
        print(f"  -- {rname} --", flush=True)
        for mode in ["M0_Base27B", "M1_LoRA_High", "M2_LoRA_Low", "M3_LoRA_Dual"]:
            sub = [r for r in sub_r if r["mode"] == mode]
            if not sub: continue
            print(
                f"    {mode:15s} (n={len(sub)}) | mot={np.mean([r['stats']['motion_l1'] for r in sub]):.4f} | "
                f"dL(glob={np.mean([r['stats']['global_luma_delta'] for r in sub]):+.3f} "
                f"hair={np.mean([r['stats']['hair_luma_delta'] for r in sub]):+.3f} "
                f"up={np.mean([r['stats']['upper_face_luma_delta'] for r in sub]):+.3f} "
                f"low={np.mean([r['stats']['lower_face_luma_delta'] for r in sub]):+.3f}) | "
                f"shR={np.mean([r['stats']['sharp_ratio_vs_inp'] for r in sub]):.3f}",
                flush=True,
            )
        # Stack 4-way visual comparisons per unseen char
        for pid in sorted(set(r["pair_idx"] for r in sub_r)):
            paths = [out_e / "strips" / f"{pid:02d}_{rname}_{m}.png" for m in ["M0_Base27B", "M1_LoRA_High", "M2_LoRA_Low", "M3_LoRA_Dual"] if (out_e / "strips" / f"{pid:02d}_{rname}_{m}.png").exists()]
            if paths:
                e31b._stack_strips(paths, out_e / f"cmp40_{pid:02d}_{rname}.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "eval_2x2", "summarize"])
    ap.add_argument("--expert", choices=["high", "low"], default="high")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n-test-chars", type=int, default=4)
    ap.add_argument("--cache-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp39_2d_grammar_train/latent_cache"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp40_wan22_a14b_moe_lora"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "train":
        return stage_train(args)
    if args.stage == "eval_2x2":
        return stage_eval_2x2(args)
    return stage_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
