#!/usr/bin/env python3
"""
Exp 43 (Generalization Boundary & Control Curve on Flagship 27B MoE `Wan2.2-I2V-A14B`):

Tests TWO critical top-venue scientific questions on UNSEEN held-out characters (`fc_00`, `fc_02`):
  1. **Zero-Shot Unseen 2D Anime Grammar Categories** (NEVER seen in Exp 39 training!):
     - `U1_BlushLines_Pop`: Classic 2D anime diagonal red/pink blush hatching strokes (`///`) appearing cleanly on cheeks
     - `U2_DepressionLines_Aura`: Classic 2D anime vertical blue-gray gloom/depression lines descending behind the head
     - `U3_ShockSpeedLines`: Classic 2D manga radial black-and-white shock/impact lines bursting in the background
     Tests honestly whether our 14B MoE LoRA generalizes zero-shot to novel 2D graphic signifiers OR strictly requires
     per-symbol supervision (critical honest boundary analysis!).
  2. **Continuous Inference-Time Control (`alpha` sweep)** on `S3_NonBioAffect` & `G3_AngerVein_Pop`:
     - `alpha in {0.0, 0.35, 0.70, 1.0}` on High-Noise LoRA (`M1_LoRA_High`) — shows smooth continuous grammar intensity
       dialing from physical literalism (`alpha=0`) to clean 2D anime grammar (`alpha=1.0`)!
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


UNSEEN_GRAMMAR_REGIMES = [
    (
        "U1_BlushLines_Pop",
        "In bright flat 2D Japanese anime style, the character looks slightly shy with a gentle smile as classic stylized 2D pink anime diagonal blush hatching lines appear cleanly across both cheeks, bright hair, clothes, and white studio background stay completely unchanged.",
    ),
    (
        "U2_DepressionLines_Aura",
        "In bright flat 2D Japanese anime style, the character looks awkwardly embarrassed as classic 2D vertical blue-gray anime depression gloom lines appear behind their head and shoulders while their bright hair, face, and clothes stay completely unchanged.",
    ),
    (
        "U3_ShockSpeedLines",
        "In bright flat 2D Japanese anime style, the character widens their eyes in comedic surprise as classic 2D manga radial black action shock speed lines burst cleanly across the white background behind them, preserving bright hair color and crisp flat cel shading.",
    ),
]

ALPHA_REGIMES = [
    (
        "S3_NonBioAffect",
        "The character's bright eye highlights suddenly vanish leaving hollow dark lifeless eyes while a sharp ominous dark shadow falls across their upper forehead, yet their mouth stays frozen in a calm eerie smile, 2D Japanese TV anime cel shading.",
    ),
    (
        "G3_AngerVein_Pop",
        "In bright flat 2D Japanese anime style, the character holds a polite closed-mouth smile as a classic stylized red cross-shaped anime anger vein symbol pops onto their upper forehead temple and pulses gently, everything else stays bright, clean, and consistent.",
    ),
]


def _set_lora_scale(peft_model, scale: float):
    if scale <= 0.0:
        peft_model.disable_adapter_layers()
        return
    peft_model.enable_adapter_layers()
    for mod in peft_model.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            if not hasattr(mod, "_orig_scaling"):
                mod._orig_scaling = dict(mod.scaling)
            for k, v in mod._orig_scaling.items():
                mod.scaling[k] = v * scale


@torch.no_grad()
def stage_run(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from peft import PeftModel

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    sel_pids = [0, 2]

    tasks = []
    # Part 1: Unseen symbols (Base27B vs LoRA_High vs LoRA_Dual) = 2 * 3 * 3 = 18 clips
    for pid in sel_pids:
        p = pairs[pid]
        for rname, rprompt in UNSEEN_GRAMMAR_REGIMES:
            for tag, hi_s, lo_s in [("M0_Base27B", 0.0, 0.0), ("M1_LoRA_High", 1.0, 0.0), ("M3_LoRA_Dual", 1.0, 1.0)]:
                tasks.append(("unseen", pid, Path(p["anime_path"]), p["anime_id"], rname, rprompt, tag, hi_s, lo_s))

    # Part 2: Continuous alpha sweep on High-Noise LoRA (char 00) = 2 * 4 = 8 clips
    p0 = pairs[0]
    for rname, rprompt in ALPHA_REGIMES:
        for alpha in [0.0, 0.35, 0.70, 1.0]:
            tasks.append(("alpha", 0, Path(p0["anime_path"]), p0["anime_id"], rname, rprompt, f"alpha_{alpha:.2f}", alpha, 0.0))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    meta_dir, strips_dir, mp4_dir = args.out_dir / "meta", args.out_dir / "strips", args.out_dir / "mp4"
    for d in [meta_dir, strips_dir, mp4_dir]:
        d.mkdir(parents=True, exist_ok=True)

    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]}_{t[1]:02d}_{t[4]}_{t[6]}.json").exists()]
    if not todo:
        print(f"[exp43 s{args.shard}] all tasks complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_dir / "lora_high_r16")
    pipe.transformer_2 = PeftModel.from_pretrained(pipe.transformer_2, args.lora_dir / "lora_low_r16")
    pipe.enable_model_cpu_offload()

    for part, pid, img_path, cid, rname, rprompt, tag, hi_s, lo_s in todo:
        _set_lora_scale(pipe.transformer, hi_s)
        _set_lora_scale(pipe.transformer_2, lo_s)

        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        seed = 9600 + pid * 37 + sum(ord(c) for c in rname) % 97
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

        key = f"{part}_{pid:02d}_{rname}_{tag}"
        e36._save_single_mp4(pil_frames, mp4_dir / f"{key}.mp4", fps=16)
        e31b._save_strip(
            pil_frames, strips_dir / f"{key}.png",
            f"Exp43 {key} | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=170,
        )
        (meta_dir / f"{key}.json").write_text(json.dumps({
            "part": part, "pair_idx": pid, "char_id": cid, "regime": rname, "tag": tag,
            "hi_scale": hi_s, "lo_scale": lo_s, "seed": seed, "stats": st,
        }, indent=2), encoding="utf-8")
        print(f"[exp43 s{args.shard}] {key:44s} | dL_glob={st['global_luma_delta']:+.3f} dL_low={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}", flush=True)
    return 0


def stage_summarize(args) -> int:
    strips_dir = args.out_dir / "strips"
    # Stack unseen symbol comparisons
    for pid in [0, 2]:
        for rname, _ in UNSEEN_GRAMMAR_REGIMES:
            ps = [strips_dir / f"unseen_{pid:02d}_{rname}_{t}.png" for t in ["M0_Base27B", "M1_LoRA_High", "M3_LoRA_Dual"]]
            ps = [p for p in ps if p.exists()]
            if ps:
                e31b._stack_strips(ps, args.out_dir / f"qc43_unseen_{pid:02d}_{rname}.png")
    # Stack alpha sweeps
    for rname, _ in ALPHA_REGIMES:
        ps = [strips_dir / f"alpha_00_{rname}_alpha_{a:.2f}.png" for a in [0.0, 0.35, 0.70, 1.0]]
        ps = [p for p in ps if p.exists()]
        if ps:
            e31b._stack_strips(ps, args.out_dir / f"qc43_alpha_00_{rname}.png")
    print(f"Saved qc43_*.png in {args.out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "summarize"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--lora-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp40_wan22_a14b_moe_lora"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp43_generalization_boundary"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_run(args) if args.stage == "run" else stage_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
