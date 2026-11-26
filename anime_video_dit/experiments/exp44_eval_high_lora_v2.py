#!/usr/bin/env python3
"""
Exp 44 (Evaluation of Clean Foreground-Masked Specular-Only `High-LoRA v2` — 14B High-Noise Expert ONLY):

Evaluates our clean v2 High-Noise-Only MoE LoRA (`lora_high_r16` trained on `exp39v2_clean_train`,
keeping the 14B Low-Noise Expert 100% FROZEN so native studio 2D cel lineart sharpness is preserved!) across:
  1. `--suite unseen8`: ALL 8 held-out UNSEEN characters (`fc_00..fc_07`) x 9 regimes (`72` clips),
     saving directly into `exp40_wan22_a14b_moe_lora/eval_2x2` as mode `M4_HighLoRA_v2` for instant
     apples-to-apples comparison against `M0_Base27B`, `M1_LoRA_High` (v1), `M2_LoRA_Low`, and `M3_LoRA_Dual` (v1)!
  2. `--suite ood6`: ALL 6 Cross-Style & Cross-Generator OOD characters (`retro90s`, `chibi_sd`,
     `watercolor`, `bw_manga`, `illustrious_blonde`, `illustrious_black`) x 4 regimes (`24` clips),
     saving directly into `exp41b_cross_style_ood` as mode `M4_HighLoRA_v2` for direct `Base` vs `v1` vs `v2`!
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
import exp41b_cross_style_ood_eval as e41b


@torch.no_grad()
def stage_eval(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from peft import PeftModel

    mode = args.mode
    tasks = []

    if args.suite in ("unseen8", "both"):
        pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        out_u = args.exp40_dir / "eval_2x2"
        for pid in range(args.n_test_chars):
            p = pairs[pid]
            for ri, (rname, rprompt, dom) in enumerate(e40.EVAL_REGIMES):
                img_path = Path(p["anime_path"]) if dom == "anime" else Path(p["real_path"])
                cid = p["anime_id"] if dom == "anime" else p["real_id"]
                seed = 9400 + pid * 43 + ri
                tasks.append(("unseen8", out_u, pid, dom, img_path, cid, rname, rprompt, seed))

    if args.suite in ("ood6", "both"):
        out_o = args.exp41b_dir
        for idx, (tag, img_path) in enumerate(e41b.OOD_PORTRAITS):
            for ri, (rname, rprompt) in enumerate(e41b.OOD_REGIMES):
                seed = 9600 + idx * 53 + ri
                tasks.append(("ood6", out_o, idx, tag, Path(img_path), tag, rname, rprompt, seed))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = []
    for t in my_tasks:
        out_root, pid, rname = t[1], t[2], t[6]
        for sub in ("meta", "strips", "mp4", "keyframes"):
            (out_root / sub).mkdir(parents=True, exist_ok=True)
        if not (out_root / "meta" / f"{pid:02d}_{rname}_{mode}.json").exists():
            todo.append(t)

    if not todo:
        print(f"[exp44 {mode} s{args.shard}] all complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_v2_dir)
    if mode == "M5_DualLoRA_v2" and args.lora_low_v2_dir.exists():
        pipe.transformer_2 = PeftModel.from_pretrained(pipe.transformer_2, args.lora_low_v2_dir)
    pipe.enable_model_cpu_offload()

    for suite_tag, out_root, pid, dom, img_path, cid, rname, rprompt, seed in todo:
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

        key = f"{pid:02d}_{rname}_{mode}"
        e36._save_single_mp4(pil_frames, out_root / "mp4" / f"{key}.mp4", fps=16)
        e31b._save_strip(
            pil_frames, out_root / "strips" / f"{key}.png",
            f"{mode} | {pid:02d}_{rname} ({cid}) | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=170,
        )
        kf_dir = out_root / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        rec = {"pair_idx": pid, "idx": pid, "domain": dom, "char_id": cid, "regime": rname, "mode": mode, "seed": seed, "stats": st}
        (out_root / "meta" / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[exp44 {suite_tag} s{args.shard}] {key:40s} ({cid:18s}) | dL_glob={st['global_luma_delta']:+.3f} dL_up={st['upper_face_luma_delta']:+.3f} dL_low={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}",
            flush=True,
        )
    return 0


def stage_stack(args) -> int:
    """Stack Base27B vs v1 High-LoRA vs v2 Clean High-LoRA visual comparison sheets!"""
    out_u = args.exp40_dir / "eval_2x2"
    v2_cmp = args.out_dir / "v1_vs_v2_sheets"
    v2_cmp.mkdir(parents=True, exist_ok=True)
    for rname in ["G1_HighlightOff_Only", "G2_ForeheadShadow_Clean", "S3_NonBioAffect", "C1_Comp_HeadTurn_Shadow"]:
        for pid in [0, 3]:
            ps = [out_u / "strips" / f"{pid:02d}_{rname}_{m}.png" for m in ["M0_Base27B", "M1_LoRA_High", "M4_HighLoRA_v2", "M5_DualLoRA_v2"]]
            ps = [p for p in ps if p.exists()]
            if len(ps) >= 2:
                e31b._stack_strips(ps, v2_cmp / f"cmp_v1_v2_{pid:02d}_{rname}.png")

    out_o = args.exp41b_dir
    for rname in ["G1_HighlightOff_Only", "S3_NonBioAffect", "G3_AngerVein_Pop", "G4_SweatDrop_Slide"]:
        ps = []
        for idx in range(6):
            for m in ["M0_Base27B", "M3_LoRA_Dual", "M4_HighLoRA_v2", "M5_DualLoRA_v2"]:
                p = out_o / "strips" / f"{idx:02d}_{rname}_{m}.png"
                if p.exists():
                    ps.append(p)
        if ps:
            e31b._stack_strips(ps, v2_cmp / f"ood_v1_v2_{rname}.png")
    print(f"Stacked v1 vs v2 sheets in {v2_cmp}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["eval", "stack"])
    ap.add_argument("--suite", default="both", choices=["unseen8", "ood6", "both"])
    ap.add_argument("--mode", default="M4_HighLoRA_v2")
    ap.add_argument("--n-test-chars", type=int, default=8)
    ap.add_argument("--lora-v2-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp44_high_lora_v2/lora_high_r16"))
    ap.add_argument("--lora-low-v2-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp44_high_lora_v2/lora_low_r16"))
    ap.add_argument("--exp40-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp40_wan22_a14b_moe_lora"))
    ap.add_argument("--exp41b-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp41b_cross_style_ood"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp44_high_lora_v2"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_eval(args) if args.stage == "eval" else stage_stack(args)


if __name__ == "__main__":
    raise SystemExit(main())
