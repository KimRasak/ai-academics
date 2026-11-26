#!/usr/bin/env python3
"""
Exp 58: Direct Causal Component-Axis Dissection Inside the 14B High-Noise Expert of Wan2.2-I2V-A14B (27B MoE)

Zero retraining! 100% parameter-matched (160 / 320 LoRA modules = 26.2M = 0.097% of 27B each) inference-time
masking on our converged `M4_HighLoRA_v2` adapter across TWO orthogonal component axes:
  Axis 2A — Attention Routing (`Q/K`) vs. Value/Output Velocity Projection (`V/O`):
    - `Comp1_QK_Only`:         ONLY `to_q` & `to_k` enabled (160/320 layers; `to_v, to_out.0` zeroed!)
    - `Comp2_VO_Only`:         ONLY `to_v` & `to_out.0` enabled (160/320 layers; `to_q, to_k` zeroed!)
  Axis 2B — 3D Video Self-Attention (`attn1`) vs. Text-to-Video Cross-Attention (`attn2`):
    - `Comp3_SelfAttn1_Only`:  ONLY `attn1.*` enabled (160/320 layers; `attn2` zeroed!)
    - `Comp4_CrossAttn2_Only`: ONLY `attn2.*` enabled (160/320 layers; `attn1` zeroed!)

Turns Axis 2 (Component Localization) from observational probe (Exp 42) into direct 2x2 causal proof!
Evaluated on unseen held-out `fc_00..03` x `S3, G2, G3, G4` (64 clips) + 4 parallel Blind CoT Judges vs `M0_Base27B`.
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
import exp40_train_wan22_a14b_moe_lora as e40

COMP_SPLITS = [
    "Comp1_QK_Only",
    "Comp2_VO_Only",
    "Comp3_SelfAttn1_Only",
    "Comp4_CrossAttn2_Only",
]

TARGET_REGIMES = [0, 2, 3, 4]  # S3, G2, G3, G4


def _match_component(name: str, cmode: str) -> bool:
    if cmode == "Comp1_QK_Only":
        return name.endswith(".to_q") or name.endswith(".to_k")
    if cmode == "Comp2_VO_Only":
        return name.endswith(".to_v") or name.endswith(".to_out.0")
    if cmode == "Comp3_SelfAttn1_Only":
        return ".attn1." in name
    if cmode == "Comp4_CrossAttn2_Only":
        return ".attn2." in name
    raise ValueError(cmode)


def _apply_component_mask(peft_model, cmode: str, orig_scales: dict) -> tuple[int, int]:
    n_on, n_off = 0, 0
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict) and "default" in mod.scaling:
            if name not in orig_scales:
                orig_scales[name] = float(mod.scaling["default"])
            if _match_component(name, cmode):
                mod.scaling["default"] = orig_scales[name]
                n_on += 1
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
    for sub in ("meta", "strips", "keyframes"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    tasks = []
    for pid in range(4):
        p = pairs[pid]
        img_path, cid = Path(p["anime_path"]), p["anime_id"]
        for ri in TARGET_REGIMES:
            rname, rprompt, dom = e40.EVAL_REGIMES[ri]
            seed = 9400 + pid * 43 + ri
            for cmode in COMP_SPLITS:
                tasks.append((pid, dom, img_path, cid, rname, rprompt, cmode, seed))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (out_root / "meta" / f"{t[0]:02d}_{t[4]}_{t[6]}.json").exists()]
    if not todo:
        print(f"[exp58 s{args.shard}] all complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_v2_dir)
    pipe.enable_model_cpu_offload()

    orig_scales: dict[str, float] = {}
    logged_masks: set[str] = set()

    for pid, dom, img_path, cid, rname, rprompt, cmode, seed in todo:
        n_on, n_off = _apply_component_mask(pipe.transformer, cmode, orig_scales)
        if cmode not in logged_masks:
            print(f"[exp58 s{args.shard}] mask {cmode} -> {n_on} LoRA modules ON, {n_off} OFF", flush=True)
            logged_masks.add(cmode)

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

        key = f"{pid:02d}_{rname}_{cmode}"
        e31b._save_strip(
            pil_frames, out_root / "strips" / f"{key}.png",
            f"{cmode} | {pid:02d}_{rname} | dL(g={st['global_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=160,
        )
        kf_dir = out_root / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        rec = {"pair_idx": pid, "idx": pid, "domain": dom, "char_id": cid, "regime": rname, "mode": cmode, "seed": seed, "stats": st}
        (out_root / "meta" / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[exp58 s{args.shard}] {key:42s} | dL_up={st['upper_face_luma_delta']:+.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp40-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp40_wan22_a14b_moe_lora"))
    ap.add_argument("--lora-v2-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp44_high_lora_v2/lora_high_r16"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    args = ap.parse_args()
    return stage_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
