#!/usr/bin/env python3
"""
Exp 54: Multi-Seed Reproducibility & Statistical Stability Audit on Flagship Wan2.2-I2V-A14B (27B MoE)

Directly answers Reviewer-2's statistical robustness question:
  "Are your visual cures and blind VLM win rates stable across independent random diffusion seeds,
   or could single-seed variance explain the gains?"

Evaluates 2 brand-new independent random seeds (`seed1 = seed0 + 10000`, `seed2 = seed0 + 20000`)
to pair with existing `seed0` from Exp 44 (`3` full independent seeds total!) across:
  - 4 unseen held-out characters (`fc_00..03`)
  - 4 core 2D grammar regimes (`S3_NonBioAffect`, `G2_ForeheadShadow_Clean`, `G3_AngerVein_Pop`, `G4_SweatDrop_Slide`)
  - 3 modes: `M0_Base27B`, `M4_HighLoRA_v2` (High-Noise ONLY, Low Frozen), `M5_DualLoRA_v2`
  = `96` new clips (`12` clips/GPU across 8xH100 -> ~18 min!), followed by per-seed & pooled
  order-balanced High-Res CoT `Qwen3-VL-32B` Blind Judges (`n=16` per seed, `n=48` pooled across 3 seeds!).
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
import exp37a_flagship_2d_grammar_benchmark as e37a
import exp40_train_wan22_a14b_moe_lora as e40

TARGET_REGIMES = [0, 2, 3, 4]  # S3, G2, G3, G4
MODES = ["M0_Base27B", "M4_HighLoRA_v2", "M5_DualLoRA_v2"]
NEW_SEEDS = [1, 2]  # seed_k = (9400 + pid*43 + ri) + 10000 * k


@torch.no_grad()
def stage_eval(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from peft import PeftModel

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = []
    for sk in NEW_SEEDS:
        sdir = args.out_dir / f"seed{sk}"
        for sub in ("meta", "strips", "keyframes"):
            (sdir / sub).mkdir(parents=True, exist_ok=True)
        for pid in range(4):
            p = pairs[pid]
            img_path, cid = Path(p["anime_path"]), p["anime_id"]
            for ri in TARGET_REGIMES:
                rname, rprompt, dom = e40.EVAL_REGIMES[ri]
                seed = (9400 + pid * 43 + ri) + 10000 * sk
                for m in MODES:
                    tasks.append((sk, sdir, pid, dom, img_path, cid, rname, rprompt, m, seed))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (t[1] / "meta" / f"{t[2]:02d}_{t[6]}_{t[8]}.json").exists()]
    if not todo:
        print(f"[exp54 s{args.shard}] all complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_high_dir)
    pipe.transformer_2 = PeftModel.from_pretrained(pipe.transformer_2, args.lora_low_dir)
    pipe.enable_model_cpu_offload()

    for sk, sdir, pid, dom, img_path, cid, rname, rprompt, m, seed in todo:
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

        key = f"{pid:02d}_{rname}_{m}"
        e31b._save_strip(
            pil_frames, sdir / "strips" / f"{key}.png",
            f"seed{sk} {m} | {pid:02d}_{rname} | dL(g={st['global_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=160,
        )
        kf_dir = sdir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        rec = {"seed_idx": sk, "pair_idx": pid, "idx": pid, "domain": dom, "char_id": cid, "regime": rname, "mode": m, "seed": seed, "stats": st}
        (sdir / "meta" / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[exp54 s{args.shard}] seed{sk} {key:42s} | dL_up={st['upper_face_luma_delta']:+.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


def stage_summary(args) -> int:
    seed_dirs = {
        "seed0": Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp40_wan22_a14b_moe_lora/eval_2x2"),
        "seed1": args.out_dir / "seed1",
        "seed2": args.out_dir / "seed2",
    }
    rnames = [e40.EVAL_REGIMES[ri][0] for ri in TARGET_REGIMES]
    summary = {}
    for rname in rnames:
        summary[rname] = {}
        print(f"\n--- {rname} (3 Independent Seeds x 4 Unseen Chars = 12 clips/mode) ---", flush=True)
        for m in MODES:
            per_seed_lo, per_seed_up, per_seed_sh = [], [], []
            for sname, sdir in seed_dirs.items():
                recs = []
                for pid in range(4):
                    p = sdir / "meta" / f"{pid:02d}_{rname}_{m}.json"
                    if p.exists():
                        recs.append(json.loads(p.read_text(encoding="utf-8"))["stats"])
                if recs:
                    per_seed_lo.append(float(np.mean([r["lower_face_luma_delta"] for r in recs])))
                    per_seed_up.append(float(np.mean([r["upper_face_luma_delta"] for r in recs])))
                    per_seed_sh.append(float(np.mean([r["sharp_ratio_vs_inp"] for r in recs])))
            if per_seed_lo:
                summary[rname][m] = {
                    "n_seeds": len(per_seed_lo),
                    "dL_up_mean": round(float(np.mean(per_seed_up)), 4),
                    "dL_up_sd": round(float(np.std(per_seed_up, ddof=1)), 4) if len(per_seed_up) > 1 else 0.0,
                    "dL_low_mean": round(float(np.mean(per_seed_lo)), 4),
                    "dL_low_sd": round(float(np.std(per_seed_lo, ddof=1)), 4) if len(per_seed_lo) > 1 else 0.0,
                    "shR_mean": round(float(np.mean(per_seed_sh)), 4),
                    "shR_sd": round(float(np.std(per_seed_sh, ddof=1)), 4) if len(per_seed_sh) > 1 else 0.0,
                }
                v = summary[rname][m]
                print(f"  {m:16s} | dL_up={v['dL_up_mean']:+.3f}±{v['dL_up_sd']:.3f} | dL_lo={v['dL_low_mean']:+.3f}±{v['dL_low_sd']:.3f} | shR={v['shR_mean']:.3f}±{v['shR_sd']:.3f}", flush=True)
    (args.out_dir / "summary_exp54_multiseed.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[exp54] Saved 3-seed stability summary -> {args.out_dir / 'summary_exp54_multiseed.json'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["eval", "summary"])
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp54_multiseed_stability"))
    ap.add_argument("--lora-high-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp44_high_lora_v2/lora_high_r16"))
    ap.add_argument("--lora-low-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp44_high_lora_v2/lora_low_r16"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_eval(args) if args.stage == "eval" else stage_summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
