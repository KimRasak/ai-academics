#!/usr/bin/env python3
"""
Exp 52: Combined 3-Axis Ultra-Sparse Micro-Subspace Verification (`0.017%` of a 27B Flagship Video DiT!)

Combines ALL THREE discovered mechanistic axes simultaneously with ZERO retraining:
  Axis 1 (Temporal):   14B High-Noise Expert ONLY (t >= 900; Low-Noise 100% frozen)
  Axis 2 (Rank/Proj):  Ultra-Low-Rank r=4 V/O/FFN LoRA (`exp46_rank_and_data_ablation/rank4_32char`)
  Axis 3 (Layer Depth): Early Semantic Blocks ONLY (`blocks 0..13` = 14/40 blocks -> `4.59M` = `0.017%` of 27B!!,
                        and `blocks 0..19` = 20/40 blocks -> `6.55M` = `0.024%` of 27B!)

Evaluates on unseen held-out `fc_00..03` x `S3, G2, G3, G4` (`32` clips total) saving directly into
`exp40_wan22_a14b_moe_lora/eval_2x2` for instant Blind CoT `Qwen3-VL-32B` Judge vs `M0_Base27B`!
"""

from __future__ import annotations

import argparse
import json
import re
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
import exp51_layer_depth_localization_wan22 as e51

MODES = [
    ("M8_Micro3Axis_0017pct", 0, 13),  # 14/40 blocks of r=4 High-Noise = 4.59M = 0.017% of 27B!
    ("M9_Micro3Axis_0024pct", 0, 19),  # 20/40 blocks of r=4 High-Noise = 6.55M = 0.024% of 27B!
]
TARGET_REGIMES = [0, 2, 3, 4]


@torch.no_grad()
def stage_eval(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from peft import PeftModel

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_root = args.exp40_dir / "eval_2x2"
    for sub in ("meta", "strips", "keyframes"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    tasks = []
    for pid in range(4):
        p = pairs[pid]
        img_path, cid = Path(p["anime_path"]), p["anime_id"]
        for ri in TARGET_REGIMES:
            rname, rprompt, dom = e40.EVAL_REGIMES[ri]
            seed = 9400 + pid * 43 + ri
            for mmode, lo_b, hi_b in MODES:
                tasks.append((pid, dom, img_path, cid, rname, rprompt, mmode, lo_b, hi_b, seed))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (out_root / "meta" / f"{t[0]:02d}_{t[4]}_{t[6]}.json").exists()]
    if not todo:
        print(f"[exp52 s{args.shard}] all complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.rank4_dir)
    pipe.enable_model_cpu_offload()

    orig_scales: dict[str, float] = {}
    logged: set[str] = set()
    for pid, dom, img_path, cid, rname, rprompt, mmode, lo_b, hi_b, seed in todo:
        n_on, n_off = e51._apply_block_mask(pipe.transformer, lo_b, hi_b, orig_scales)
        if mmode not in logged:
            print(f"[exp52 s{args.shard}] {mmode} [{lo_b}..{hi_b}] -> {n_on} ON, {n_off} OFF", flush=True)
            logged.add(mmode)

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

        key = f"{pid:02d}_{rname}_{mmode}"
        e31b._save_strip(
            pil_frames, out_root / "strips" / f"{key}.png",
            f"{mmode} | {pid:02d}_{rname} | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=160,
        )
        kf_dir = out_root / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        rec = {"pair_idx": pid, "idx": pid, "domain": dom, "char_id": cid, "regime": rname, "mode": mmode, "seed": seed, "stats": st}
        (out_root / "meta" / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[exp52 s{args.shard}] {key:44s} | dL_g={st['global_luma_delta']:+.3f} dL_up={st['upper_face_luma_delta']:+.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank4-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp46_rank_and_data_ablation/rank4_32char/lora_high_r4"))
    ap.add_argument("--exp40-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp40_wan22_a14b_moe_lora"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    return stage_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
