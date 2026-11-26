#!/usr/bin/env python3
"""
Exp 51: 40-Layer Depth Localization Inside the 14B High-Noise Expert of Flagship Wan2.2-I2V-A14B (27B MoE)

Completes the 3-Axis Mechanistic Localization of 2D Non-Physical Anime Grammar:
  Axis 1 (Temporal / Noise Band): High-Noise Expert t>=900 (Wan2.2) & Critical Band sigma in [0.80, 0.98] (LTX-13B)
  Axis 2 (Sub-Layer Component):   V/O/FFN velocity projection rewiring vs flat Q*K cross-attention (Exp 42)
  Axis 3 (Transformer Depth):     THIS EXPERIMENT — ZERO retraining! Pure inference-time block gating on our
                                  trained `M4_HighLoRA_v2` (40 DiT blocks: `blocks.0 .. blocks.39`):
                                    - `D1_Early_0_13`:  LoRA active ONLY in blocks  0..13 (14/40 blocks)
                                    - `D2_Mid_13_26`:   LoRA active ONLY in blocks 13..26 (14/40 blocks)
                                    - `D3_Late_26_39`:  LoRA active ONLY in blocks 26..39 (14/40 blocks)
                                  compared directly against already-computed `M0_Base27B` (0/40) and
                                  `M4_HighLoRA_v2` (all 40/40) on unseen held-out `fc_00..03` x `S3, G2, G3, G4`
                                  (`48` clips -> ~8 min on 6xH100!).
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
import exp36_flagship_27b_13b_anime_stress_test as e36
import exp37a_flagship_2d_grammar_benchmark as e37a
import exp40_train_wan22_a14b_moe_lora as e40

DEPTH_SPLITS = [
    ("D1_Early_0_13", 0, 13),
    ("D2_Mid_13_26", 13, 26),
    ("D3_Late_26_39", 26, 39),
]

TARGET_REGIMES = [0, 2, 3, 4]  # S3, G2, G3, G4


def _apply_block_mask(peft_model, lo_blk: int, hi_blk: int, orig_scales: dict) -> tuple[int, int]:
    """Enable LoRA adapters strictly in `blocks.{lo_blk..hi_blk}`, zeroing all other blocks."""
    n_on, n_off = 0, 0
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict) and "default" in mod.scaling:
            if name not in orig_scales:
                orig_scales[name] = float(mod.scaling["default"])
            m = re.search(r"\bblocks\.(\d+)\b", name)
            if m is not None:
                bi = int(m.group(1))
                if lo_blk <= bi <= hi_blk:
                    mod.scaling["default"] = orig_scales[name]
                    n_on += 1
                else:
                    mod.scaling["default"] = 0.0
                    n_off += 1
            else:
                mod.scaling["default"] = 0.0
                n_off += 1
    return n_on, n_off


@torch.no_grad()
def stage_eval(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from peft import PeftModel

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_root = args.exp40_dir / "eval_2x2"
    for sub in ("meta", "strips", "mp4", "keyframes"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    tasks = []
    for pid in range(4):
        p = pairs[pid]
        img_path, cid = Path(p["anime_path"]), p["anime_id"]
        for ri in TARGET_REGIMES:
            rname, rprompt, dom = e40.EVAL_REGIMES[ri]
            seed = 9400 + pid * 43 + ri
            for dmode, lo_b, hi_b in DEPTH_SPLITS:
                tasks.append((pid, dom, img_path, cid, rname, rprompt, dmode, lo_b, hi_b, seed))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (out_root / "meta" / f"{t[0]:02d}_{t[4]}_{t[6]}.json").exists()]
    if not todo:
        print(f"[exp51 s{args.shard}] all complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_v2_dir)
    pipe.enable_model_cpu_offload()

    orig_scales: dict[str, float] = {}
    logged_masks: set[str] = set()

    for pid, dom, img_path, cid, rname, rprompt, dmode, lo_b, hi_b, seed in todo:
        n_on, n_off = _apply_block_mask(pipe.transformer, lo_b, hi_b, orig_scales)
        if dmode not in logged_masks:
            print(f"[exp51 s{args.shard}] mask {dmode} [{lo_b}..{hi_b}] -> {n_on} LoRA modules ON, {n_off} OFF", flush=True)
            logged_masks.add(dmode)

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

        key = f"{pid:02d}_{rname}_{dmode}"
        e31b._save_strip(
            pil_frames, out_root / "strips" / f"{key}.png",
            f"{dmode} | {pid:02d}_{rname} | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=160,
        )
        kf_dir = out_root / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        rec = {"pair_idx": pid, "idx": pid, "domain": dom, "char_id": cid, "regime": rname, "mode": dmode, "seed": seed, "stats": st}
        (out_root / "meta" / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[exp51 s{args.shard}] {key:42s} | dL_g={st['global_luma_delta']:+.3f} dL_up={st['upper_face_luma_delta']:+.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


def stage_summary(args) -> int:
    out_root = args.exp40_dir / "eval_2x2"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["M0_Base27B", "D1_Early_0_13", "D2_Mid_13_26", "D3_Late_26_39", "M4_HighLoRA_v2"]
    rnames = [e40.EVAL_REGIMES[ri][0] for ri in TARGET_REGIMES]
    summary = {}
    for rname in rnames:
        print(f"\n--- {rname} ---", flush=True)
        summary[rname] = {}
        for m in modes:
            recs = []
            for pid in range(4):
                p = out_root / "meta" / f"{pid:02d}_{rname}_{m}.json"
                if p.exists():
                    recs.append(json.loads(p.read_text(encoding="utf-8"))["stats"])
            if recs:
                row = {
                    "n": len(recs),
                    "dL_glob": round(float(np.mean([r["global_luma_delta"] for r in recs])), 4),
                    "dL_hair": round(float(np.mean([r["hair_luma_delta"] for r in recs])), 4),
                    "dL_up": round(float(np.mean([r["upper_face_luma_delta"] for r in recs])), 4),
                    "dL_low": round(float(np.mean([r["lower_face_luma_delta"] for r in recs])), 4),
                    "shR": round(float(np.mean([r["sharp_ratio_vs_inp"] for r in recs])), 4),
                }
                summary[rname][m] = row
                print(f"  {m:18s} | dL(g={row['dL_glob']:+.3f} h={row['dL_hair']:+.3f} up={row['dL_up']:+.3f} lo={row['dL_low']:+.3f}) | shR={row['shR']:.3f}", flush=True)
    (args.out_dir / "summary_exp51_depth.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for pid in [0, 2]:
        for rname in rnames:
            ps = [out_root / "strips" / f"{pid:02d}_{rname}_{m}.png" for m in modes if (out_root / "strips" / f"{pid:02d}_{rname}_{m}.png").exists()]
            if len(ps) >= 3:
                e31b._stack_strips(ps, args.out_dir / f"qc51_depth_{pid:02d}_{rname}.png")
    print(f"[exp51] Saved depth localization summary & QC sheets -> {args.out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["eval", "summary"])
    ap.add_argument("--lora-v2-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp44_high_lora_v2/lora_high_r16"))
    ap.add_argument("--exp40-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp40_wan22_a14b_moe_lora"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp51_layer_depth_localization"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    return stage_eval(args) if args.stage == "eval" else stage_summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
