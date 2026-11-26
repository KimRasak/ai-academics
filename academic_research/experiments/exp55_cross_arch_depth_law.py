#!/usr/bin/env python3
"""
Exp 55: Cross-Architecture Layer-Depth Localization Law on 2nd & 3rd Flagship Architectures
        (`LTX-Video-0.9.7 13B` 48L Non-MoE + `HunyuanVideo-I2V 13B` 20 Dual-Stream + 40 Single-Stream MM-DiT)

Zero retraining! Pure inference-time block masking on our already-converged LoRAs:
  1. `--arch ltx13b` (48 transformer_blocks `0..47` in `exp45b_ltx13b_r32_1200/lora_ltx13b_r32`):
       - `LD1_Early_0_15`  (first 16/48 blocks)
       - `LD2_Mid_16_31`   (middle 16/48 blocks)
       - `LD3_Late_32_47`  (final 16/48 blocks)
  2. `--arch hy13b` (20 Dual-Stream `transformer_blocks.0..19` vs 40 Single-Stream `single_transformer_blocks.0..39`
     in `exp50_hy13b_cross_arch/lora_hy13b_r16`):
       - `HD1_DualStream_0_19`    (ONLY the 20 Dual-Stream multimodal blocks!)
       - `HD2_SingleEarly_0_19`   (ONLY the first 20 Single-Stream blocks!)
       - `HD3_SingleLate_20_39`   (ONLY the final 20 Single-Stream blocks!)

Completes the full 3-Architecture × 2-Axis (Sigma-Space + Layer-Depth) Mechanistic Matrix!
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
import exp47_ltx13b_critical_timestep_curve as e47
import exp50_hunyuan13b_cross_arch_lora as e50


def _mask_ltx(peft_model, lo_b: int, hi_b: int, orig_scales: dict) -> tuple[int, int]:
    n_on, n_off = 0, 0
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict) and "default" in mod.scaling:
            if name not in orig_scales:
                orig_scales[name] = float(mod.scaling["default"])
            m = re.search(r"transformer_blocks\.(\d+)\b", name)
            if m is not None and lo_b <= int(m.group(1)) <= hi_b:
                mod.scaling["default"] = orig_scales[name]
                n_on += 1
            else:
                mod.scaling["default"] = 0.0
                n_off += 1
    return n_on, n_off


def _mask_hy(peft_model, kind: str, orig_scales: dict) -> tuple[int, int]:
    n_on, n_off = 0, 0
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict) and "default" in mod.scaling:
            if name not in orig_scales:
                orig_scales[name] = float(mod.scaling["default"])
            m_dual = re.search(r"(?<!single_)transformer_blocks\.(\d+)\b", name)
            m_sing = re.search(r"single_transformer_blocks\.(\d+)\b", name)
            keep = False
            if kind == "dual" and m_dual is not None:
                keep = True
            elif kind == "single_early" and m_sing is not None and int(m_sing.group(1)) <= 19:
                keep = True
            elif kind == "single_late" and m_sing is not None and int(m_sing.group(1)) >= 20:
                keep = True
            if keep:
                mod.scaling["default"] = orig_scales[name]
                n_on += 1
            else:
                mod.scaling["default"] = 0.0
                n_off += 1
    return n_on, n_off


@torch.no_grad()
def run_ltx(args) -> int:
    from diffusers import LTXConditionPipeline
    from peft import PeftModel

    out_dir = Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp47_ltx13b_timestep_curve")
    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    splits = [
        ("LD1_Early_0_15", 0, 15),
        ("LD2_Mid_16_31", 16, 31),
        ("LD3_Late_32_47", 32, 47),
    ]
    tasks = []
    for pid in range(4):
        p = pairs[pid]
        for ri, (rname, rprompt, _) in enumerate(e47.SWEEP_REGIMES):
            seed = 9500 + pid * 41 + ri
            for sname, lo_b, hi_b in splits:
                tasks.append((pid, Path(p["anime_path"]), p["anime_id"], rname, rprompt, sname, lo_b, hi_b, seed))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (out_dir / "meta" / f"{t[0]:02d}_{t[3]}_{t[5]}.json").exists()]
    if not todo:
        print(f"[exp55 ltx s{args.shard}] all complete!", flush=True)
        return 0

    pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp45b_ltx13b_r32_1200/lora_ltx13b_r32"))
    pipe.enable_model_cpu_offload()

    orig_scales, logged = {}, set()
    for pid, img_path, cid, rname, rprompt, sname, lo_b, hi_b, seed in todo:
        n_on, n_off = _mask_ltx(pipe.transformer, lo_b, hi_b, orig_scales)
        if sname not in logged:
            print(f"[exp55 ltx s{args.shard}] {sname} [{lo_b}..{hi_b}] -> {n_on} ON, {n_off} OFF", flush=True)
            logged.add(sname)

        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=im, prompt=rprompt,
            negative_prompt="blurry, low quality, distorted, extra limbs, watermark, text",
            height=512, width=512, num_frames=49, num_inference_steps=36,
            guidance_scale=3.0, generator=gen,
        ).frames[0]

        pil_frames = e31b._to_pil_frames(frames)
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))
        st.update(e37a._grammar_diagnostics(np_frames, inp_u8))

        key = f"{pid:02d}_{rname}_{sname}"
        e31b._save_strip(
            pil_frames, out_dir / "strips" / f"{key}.png",
            f"LTX13B {sname} | {pid:02d}_{rname} | dL(g={st['global_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=160,
        )
        kf_dir = out_dir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")
        (out_dir / "meta" / f"{key}.json").write_text(json.dumps({"pair_idx": pid, "char_id": cid, "regime": rname, "mode": sname, "seed": seed, "stats": st}, indent=2), encoding="utf-8")
        print(f"[exp55 ltx s{args.shard}] {key:42s} | dL_up={st['upper_face_luma_delta']:+.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


@torch.no_grad()
def run_hy(args) -> int:
    from diffusers import HunyuanVideoImageToVideoPipeline
    from peft import PeftModel

    out_dir = Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp50_hy13b_cross_arch/eval")
    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    splits = [
        ("HD1_DualStream_0_19", "dual"),
        ("HD2_SingleEarly_0_19", "single_early"),
        ("HD3_SingleLate_20_39", "single_late"),
    ]
    target_r = [r for r in e50.HY_EVAL_REGIMES if r[0] in ("S3_NonBioAffect", "G2_ForeheadShadow_Clean", "G3_AngerVein_Pop", "G4_SweatDrop_Slide")]
    tasks = []
    for pid in range(4):
        p = pairs[pid]
        for ri_full, (rname, rprompt, dom) in enumerate(e50.HY_EVAL_REGIMES):
            if rname not in ("S3_NonBioAffect", "G2_ForeheadShadow_Clean", "G3_AngerVein_Pop", "G4_SweatDrop_Slide"):
                continue
            seed = 9700 + pid * 47 + ri_full
            for sname, skind in splits:
                tasks.append((pid, Path(p["anime_path"]), p["anime_id"], rname, rprompt, sname, skind, seed))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (out_dir / "meta" / f"{t[0]:02d}_{t[3]}_{t[5]}.json").exists()]
    if not todo:
        print(f"[exp55 hy s{args.shard}] all complete!", flush=True)
        return 0

    pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(e50.MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp50_hy13b_cross_arch/lora_hy13b_r16"))
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    orig_scales, logged = {}, set()
    for pid, img_path, cid, rname, rprompt, sname, skind, seed in todo:
        n_on, n_off = _mask_hy(pipe.transformer, skind, orig_scales)
        if sname not in logged:
            print(f"[exp55 hy s{args.shard}] {sname} ({skind}) -> {n_on} ON, {n_off} OFF", flush=True)
            logged.add(sname)

        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=im, prompt=rprompt, height=512, width=512, num_frames=33,
            num_inference_steps=30, guidance_scale=6.0, generator=gen,
        ).frames[0]

        pil_frames = e31b._to_pil_frames(frames)
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))
        st.update(e37a._grammar_diagnostics(np_frames, inp_u8))

        key = f"{pid:02d}_{rname}_{sname}"
        e31b._save_strip(
            pil_frames, out_dir / "strips" / f"{key}.png",
            f"HY13B {sname} | {pid:02d}_{rname} | dL(g={st['global_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=160,
        )
        kf_dir = out_dir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")
        (out_dir / "meta" / f"{key}.json").write_text(json.dumps({"pair_idx": pid, "char_id": cid, "regime": rname, "mode": sname, "seed": seed, "stats": st}, indent=2), encoding="utf-8")
        print(f"[exp55 hy s{args.shard}] {key:42s} | dL_up={st['upper_face_luma_delta']:+.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["ltx13b", "hy13b"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    return run_ltx(args) if args.arch == "ltx13b" else run_hy(args)


if __name__ == "__main__":
    raise SystemExit(main())
