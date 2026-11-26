#!/usr/bin/env python3
"""
Exp 50B: Universal Noise-Level (sigma >= 0.80 vs sigma < 0.80) Law Proof on HunyuanVideo-I2V 13B MM-DiT!

Because `HunyuanVideo-I2V` uses an extreme `shift=17.0` flow schedule:
  - Step 0..14 (`M1_HighHalf`) only spans `sigma in [0.951, 1.000]` (ABOVE the `[0.80, 0.95]` formation band!)
  - Step 15..23 spans `sigma in [0.80, 0.95]` (INSIDE the critical formation band!)
  - Step 24..29 spans `sigma in [0.00, 0.80]` (the bottom 80% of noise space!)

Tests the EXACT sigma=0.80 (`t = 800`) causal split directly on our trained `lora_hy13b_r16`:
  - `M4_HY13B_Sigma_ge800`: LoRA ON for `t >= 800` (`sigma in [0.80, 1.00]`), OFF for `t < 800` (`sigma < 0.80`)!
  - `M5_HY13B_Sigma_lt800`: LoRA ON ONLY for `t < 800` (`sigma in [0.00, 0.80]`), OFF for `t >= 800`!

Saves directly into `exp50_hy13b_cross_arch/eval/` (`56` clips -> 4.5 min on 4 H100s!) for direct Blind CoT
`Qwen3-VL-32B` comparison against `M0_HY13B_Base` and `M3_HY13B_LoRA_Full`!
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
import exp36_flagship_27b_13b_anime_stress_test as e36
import exp37a_flagship_2d_grammar_benchmark as e37a
import exp50_hunyuan13b_cross_arch_lora as e50

MODES = [
    ("M4_HY13B_Sigma_ge800", "ge", 800.0),
    ("M5_HY13B_Sigma_lt800", "lt", 800.0),
]


@torch.no_grad()
def stage_eval(args) -> int:
    from diffusers import HunyuanVideoImageToVideoPipeline
    from peft import PeftModel

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_dir = args.out_dir / "eval"
    for sub in ("meta", "strips", "mp4", "keyframes"):
        (eval_dir / sub).mkdir(parents=True, exist_ok=True)

    tasks = []
    for pid in range(4):
        p = pairs[pid]
        for ri, (rname, rprompt, dom) in enumerate(e50.HY_EVAL_REGIMES):
            img_path = Path(p["anime_path"]) if dom == "anime" else Path(p["real_path"])
            cid = p["anime_id"] if dom == "anime" else p["real_id"]
            seed = 9700 + pid * 47 + ri
            for mname, mkind, mthr in MODES:
                tasks.append((pid, dom, img_path, cid, rname, rprompt, mname, mkind, mthr, seed))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (eval_dir / "meta" / f"{t[0]:02d}_{t[4]}_{t[6]}.json").exists()]
    if not todo:
        print(f"[exp50b s{args.shard}] all complete!", flush=True)
        return 0

    pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(e50.MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.out_dir / "lora_hy13b_r16")
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    logged_sched = False
    for pid, dom, img_path, cid, rname, rprompt, mname, mkind, mthr, seed in todo:
        if mkind == "ge":
            pipe.transformer.enable_adapter_layers()
            def _make_ge(thr):
                def _cb(p_self, step_i, t_val, cb_kwargs):
                    # Check NEXT step's timestep if available, else current t_val
                    ts = p_self.scheduler.timesteps
                    nxt = float(ts[step_i + 1].item()) if step_i + 1 < len(ts) else 0.0
                    if nxt < thr:
                        p_self.transformer.disable_adapter_layers()
                    else:
                        p_self.transformer.enable_adapter_layers()
                    return cb_kwargs
                return _cb
            cb = _make_ge(mthr)
        else:
            pipe.transformer.disable_adapter_layers()
            def _make_lt(thr):
                def _cb(p_self, step_i, t_val, cb_kwargs):
                    ts = p_self.scheduler.timesteps
                    nxt = float(ts[step_i + 1].item()) if step_i + 1 < len(ts) else 0.0
                    if nxt < thr:
                        p_self.transformer.enable_adapter_layers()
                    else:
                        p_self.transformer.disable_adapter_layers()
                    return cb_kwargs
                return _cb
            cb = _make_lt(mthr)

        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=im, prompt=rprompt, height=512, width=512, num_frames=33,
            num_inference_steps=30, guidance_scale=6.0, generator=gen, callback_on_step_end=cb,
        ).frames[0]

        if not logged_sched:
            ts = [round(float(x.item()), 1) for x in pipe.scheduler.timesteps]
            n_ge = sum(1 for x in ts if x >= 800.0)
            print(f"[exp50b s{args.shard}] HY13B 30-step timesteps (shift=17.0): {ts} | t>=800: {n_ge}/30, t<800: {30-n_ge}/30", flush=True)
            logged_sched = True

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
            f"{mname} | {pid:02d}_{rname} | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
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
        print(f"[exp50b s{args.shard}] {key:44s} | dL_g={st['global_luma_delta']:+.3f} dL_up={st['upper_face_luma_delta']:+.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


def stage_stack(args) -> int:
    eval_dir = args.out_dir / "eval"
    modes = [
        "M0_HY13B_Base",
        "M5_HY13B_Sigma_lt800",
        "M1_HY13B_LoRA_HighHalf",
        "M4_HY13B_Sigma_ge800",
        "M3_HY13B_LoRA_Full",
    ]
    for pid in [0, 2]:
        for rname, _, _ in e50.HY_EVAL_REGIMES:
            ps = [eval_dir / "strips" / f"{pid:02d}_{rname}_{m}.png" for m in modes if (eval_dir / "strips" / f"{pid:02d}_{rname}_{m}.png").exists()]
            if len(ps) >= 3:
                e31b._stack_strips(ps, args.out_dir / f"qc50b_sigma_law_{pid:02d}_{rname}.png")
    print(f"[exp50b] Stacked sigma-law QC sheets in {args.out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["eval", "stack"])
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp50_hy13b_cross_arch"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    return stage_eval(args) if args.stage == "eval" else stage_stack(args)


if __name__ == "__main__":
    raise SystemExit(main())
