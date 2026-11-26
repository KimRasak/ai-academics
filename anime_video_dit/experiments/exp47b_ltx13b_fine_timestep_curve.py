#!/usr/bin/env python3
"""
Exp 47B (Fine-Grained Step-Fraction & Sigma-Threshold Phase Transition Curve on Monolithic Non-MoE 13B LTX-Video-0.9.7):

Because `LTXConditionPipeline` uses `linear_quadratic_schedule(36)` where:
  - t >= 980  covers 15/36 steps (41.7% of steps; sigma in [0.980, 1.000])
  - t >= 950  covers 21/36 steps (58.3% of steps; sigma in [0.950, 1.000])
  - t >= 800  covers 26/36 steps (72.2% of steps; sigma in [0.800, 1.000])  [already in Exp 47]
  - t <  950  covers 15/36 steps (41.7% of steps; sigma in [0.000, 0.950])  [step-matched inverse of t >= 980!]
  - t <  800  covers 10/36 steps (27.8% of steps; sigma in [0.000, 0.800])  [causal inverse of t >= 800!]
  - t <  550  covers  6/36 steps (16.7% of steps; sigma in [0.000, 0.550])  [already in Exp 47 -> 0W/0L/16T!]

Appends these 4 critical high-resolution windows into `exp47_ltx13b_timestep_curve/` (`64` clips = ~4 min on 6 H100s!)
to pinpoint the EXACT critical transition boundary in BOTH step-fraction AND noise-level space!
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
import exp47_ltx13b_critical_timestep_curve as e47

EXTRA_WINDOWS = [
    ("W2a_High_ge980", "ge", 980.0),      # 15/36 steps (top 41.7% steps, sigma >= 0.980)
    ("W2b_High_ge950", "ge", 950.0),      # 21/36 steps (top 58.3% steps, sigma >= 0.950)
    ("W1b_LowOnly_lt950", "lt", 950.0),   # 15/36 steps (bottom 41.7% steps, sigma < 0.950)
    ("W1c_LowOnly_lt800", "lt", 800.0),   # 10/36 steps (bottom 27.8% steps, sigma < 0.800)
]

ALL_ORDERED_WINDOWS = [
    "W0_Base",
    "W1_LowOnly_lt550",
    "W1c_LowOnly_lt800",
    "W1b_LowOnly_lt950",
    "W2a_High_ge980",
    "W2b_High_ge950",
    "W2_High_ge800",
    "W3_High_ge650",
    "W4_High_ge550",
    "W5_Full",
]


@torch.no_grad()
def stage_run(args) -> int:
    from diffusers import LTXConditionPipeline
    from peft import PeftModel

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = []
    for pid in range(4):
        p = pairs[pid]
        for ri, (rname, rprompt, _) in enumerate(e47.SWEEP_REGIMES):
            for wname, wkind, wthr in EXTRA_WINDOWS:
                tasks.append((pid, Path(p["anime_path"]), p["anime_id"], rname, rprompt, wname, wkind, wthr, 9500 + pid * 41 + ri))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    meta_dir, strips_dir, kf_root = args.out_dir / "meta", args.out_dir / "strips", args.out_dir / "keyframes"
    for d in [meta_dir, strips_dir, kf_root]:
        d.mkdir(parents=True, exist_ok=True)
    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[3]}_{t[5]}.json").exists()]
    if not todo:
        print(f"[exp47b s{args.shard}] all complete!", flush=True)
        return 0

    pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_dir)
    pipe.enable_model_cpu_offload()

    for pid, img_path, cid, rname, rprompt, wname, wkind, wthr, seed in todo:
        if wkind == "ge":
            pipe.transformer.enable_adapter_layers()
            def _make_ge(thr):
                def _cb(p_self, step_i, t_val, cb_kwargs):
                    tv = float(t_val.item()) if hasattr(t_val, "item") else float(t_val)
                    if tv < thr:
                        p_self.transformer.disable_adapter_layers()
                    else:
                        p_self.transformer.enable_adapter_layers()
                    return cb_kwargs
                return _cb
            cb = _make_ge(wthr)
        else:
            pipe.transformer.disable_adapter_layers()
            def _make_lt(thr):
                def _cb(p_self, step_i, t_val, cb_kwargs):
                    tv = float(t_val.item()) if hasattr(t_val, "item") else float(t_val)
                    if tv < thr:
                        p_self.transformer.enable_adapter_layers()
                    else:
                        p_self.transformer.disable_adapter_layers()
                    return cb_kwargs
                return _cb
            cb = _make_lt(wthr)

        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=im, prompt=rprompt,
            negative_prompt="blurry, low quality, distorted, extra limbs, watermark, text",
            height=512, width=512, num_frames=49, num_inference_steps=36,
            guidance_scale=3.0, generator=gen, callback_on_step_end=cb,
        ).frames[0]

        pil_frames = e31b._to_pil_frames(frames)
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))
        st.update(e37a._grammar_diagnostics(np_frames, inp_u8))

        key = f"{pid:02d}_{rname}_{wname}"
        e31b._save_strip(
            pil_frames, strips_dir / f"{key}.png",
            f"LTX13B {wname} | {pid:02d}_{rname} | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=160,
        )
        kf_dir = kf_root / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        (meta_dir / f"{key}.json").write_text(json.dumps({
            "pair_idx": pid, "char_id": cid, "regime": rname, "mode": wname, "seed": seed, "stats": st,
        }, indent=2), encoding="utf-8")
        print(f"[exp47b s{args.shard}] {key:42s} | dL_g={st['global_luma_delta']:+.3f} dL_up={st['upper_face_luma_delta']:+.3f} dL_low={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


def stage_summarize(args) -> int:
    meta_dir, strips_dir = args.out_dir / "meta", args.out_dir / "strips"
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(meta_dir.glob("*.json"))]
    summary = {}
    for rname, _, _ in e47.SWEEP_REGIMES:
        print(f"\n--- {rname} ---", flush=True)
        summary[rname] = {}
        for wname in ALL_ORDERED_WINDOWS:
            sub = [r["stats"] for r in recs if r["regime"] == rname and r["mode"] == wname]
            if not sub:
                continue
            row = {
                "n": len(sub),
                "dL_glob": round(float(np.mean([s["global_luma_delta"] for s in sub])), 4),
                "dL_hair": round(float(np.mean([s["hair_luma_delta"] for s in sub])), 4),
                "dL_up": round(float(np.mean([s["upper_face_luma_delta"] for s in sub])), 4),
                "dL_low": round(float(np.mean([s["lower_face_luma_delta"] for s in sub])), 4),
                "shR": round(float(np.mean([s["sharp_ratio_vs_inp"] for s in sub])), 4),
            }
            summary[rname][wname] = row
            print(f"  {wname:18s} | dL(g={row['dL_glob']:+.3f} h={row['dL_hair']:+.3f} up={row['dL_up']:+.3f} lo={row['dL_low']:+.3f}) | shR={row['shR']:.3f}", flush=True)

    for pid in [0, 2]:
        for rname, _, _ in e47.SWEEP_REGIMES:
            ps = [strips_dir / f"{pid:02d}_{rname}_{w}.png" for w in ALL_ORDERED_WINDOWS if (strips_dir / f"{pid:02d}_{rname}_{w}.png").exists()]
            if ps:
                e31b._stack_strips(ps, args.out_dir / f"qc47b_full_curve_{pid:02d}_{rname}.png")
    (args.out_dir / "summary_exp47b_full.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "summarize"])
    ap.add_argument("--lora-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp45b_ltx13b_r32_1200/lora_ltx13b_r32"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp47_ltx13b_timestep_curve"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    return stage_run(args) if args.stage == "run" else stage_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
