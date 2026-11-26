#!/usr/bin/env python3
"""
Exp 49: Training-Free Baselines Benchmark on Flagship Wan2.2-I2V-A14B (27B MoE)

Directly answers the #1 Reviewer-2 objection:
  "Could Physical World-Model Literalism on 27B MoE be solved WITHOUT any weight-level
   LoRA adaptation, simply via targeted Negative Prompting, Guidance Scale tuning, or
   Hyper-Explicit 2D-Overlay Prompt Engineering?"

Evaluates 4 best-effort training-free baselines on the EXACT same unseen held-out portraits
(`fc_00..03`), 5 core 2D grammar regimes (`S3, G1, G2, G3, G4`), and exact seeds (`9400 + pid*43 + ri`)
as Exp 40/44 (`80` clips total -> ~13 min across 6 H100 GPUs):
  - `B1_StrongNegPrompt`:     Targeted anti-literalism negative prompt (no blackout/closed eyes/hairclip/tears)
  - `B2_HighCFG_6.0`:         High classifier-free guidance (`g1=g2=6.0`) to maximize text adherence
  - `B3_LowCFG_2.0`:          Low classifier-free guidance (`g1=g2=2.0`) to curb global lighting blackout
  - `B4_HyperExplicitPrompt`: Best-effort 2D cel-overlay prompt engineering + StrongNegPrompt combo

Saves directly into `exp40_wan22_a14b_moe_lora/eval_2x2` so Blind High-Res CoT `Qwen3-VL-32B`
Judge (`exp40d_highres_cot_blind_judge.py`) evaluates `B1..B4` head-to-head against `M5_DualLoRA_v2`!
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

DEFAULT_NEG = "blurry, low quality, distorted, extra limbs, watermark, text"
STRONG_NEG = (
    "blurry, low quality, distorted, extra limbs, watermark, text, "
    "dark room, room blackout, dim lighting, global darkness, shadowed lower face, "
    "darkened background, dark chin, dark clothes, darkened hair, hair dye, "
    "closed eyes, blinking, eyes shutting, squinting, sleepy eyes, "
    "3D plastic hairclip, hairpin, accessory clip, blood, red wound, scratch, "
    "dripping tears, crying, wet cheeks, water splatter, puddles, "
    "photorealistic lighting, volumetric fog, vignette, sepia, monochrome"
)

HYPER_EXPLICIT_PROMPTS = {
    "S3_NonBioAffect": (
        "Flat 2D cel-shaded Japanese TV anime animation overlay (NOT 3D room lighting). "
        "Keep the pure bright white background, bright hair color, bright cheeks, chin, clothes, and calm smile 100% unchanged and brightly lit. "
        "Keep both eyes WIDE OPEN without blinking. Strictly two localized 2D graphic effects appear: "
        "(1) ONLY the tiny white specular highlight dots inside both open eyes fade away leaving flat matte irises, and "
        "(2) a translucent slate-blue 2D anime gloom shadow band with fine vertical ink hatching lines overlays strictly across the upper forehead and brow."
    ),
    "G1_HighlightOff_Only": (
        "Flat 2D cel-shaded Japanese TV anime animation overlay (NOT blinking, NOT closing eyes). "
        "The character stays completely still with both eyes WIDE OPEN in bright even studio daylight; "
        "hair, skin, clothes, smile, and pure white background remain 100% unchanged. "
        "ONLY the small white specular reflection catchlight dots inside the open colored irises smoothly vanish, leaving flat matte dull open eyes."
    ),
    "G2_ForeheadShadow_Clean": (
        "Flat 2D cel-shaded Japanese TV anime graphic overlay (NOT room dimming, NOT hair dye, NOT an opaque board). "
        "Keep the pure bright white background, lower face, cheeks, gentle smile, and uniform brightly lit and completely unchanged. "
        "A translucent 2D anime indigo-slate gloom gradient band with fine vertical black ink lines fades in strictly across the upper forehead and brow while hair texture remains visible underneath."
    ),
    "G3_AngerVein_Pop": (
        "Flat 2D cel-shaded Japanese TV anime comic symbol overlay (NOT a 3D hairclip, NOT a hair ornament, NOT blood or a scar). "
        "The character holds a calm polite smile with eyes open in bright even lighting. "
        "A flat hand-drawn red 2D manga anger vein mark (four small curved red ink brackets forming a pop-mark) pops onto the upper forehead temple with crisp 2D cel outlines and pulses gently."
    ),
    "G4_SweatDrop_Slide": (
        "Flat 2D cel-shaded Japanese TV anime comic symbol overlay (NOT crying, NOT tears from the eyes, NOT dripping water). "
        "The character holds an awkward polite smile in bright flat daylight. "
        "A single large stylized flat 2D cel-shaded pale blue anime sweat drop with a crisp dark ink outline and white shine mark appears beside the temple and slides smoothly downward."
    ),
}

BASELINES = [
    ("B1_StrongNegPrompt", 3.5, STRONG_NEG, False),
    ("B2_HighCFG_6.0", 6.0, DEFAULT_NEG, False),
    ("B3_LowCFG_2.0", 2.0, DEFAULT_NEG, False),
    ("B4_HyperExplicitPrompt", 3.5, STRONG_NEG, True),
]

TARGET_REGIMES = [0, 1, 2, 3, 4]  # S3, G1, G2, G3, G4


@torch.no_grad()
def stage_eval(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_root = args.exp40_dir / "eval_2x2"
    for sub in ("meta", "strips", "mp4", "keyframes"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    tasks = []
    for pid in range(args.n_test_chars):
        p = pairs[pid]
        img_path = Path(p["anime_path"])
        cid = p["anime_id"]
        for ri in TARGET_REGIMES:
            rname, rprompt, dom = e40.EVAL_REGIMES[ri]
            seed = 9400 + pid * 43 + ri
            for bmode, cfg, neg_p, use_hyper in BASELINES:
                prompt = HYPER_EXPLICIT_PROMPTS[rname] if use_hyper else rprompt
                tasks.append((pid, dom, img_path, cid, rname, prompt, neg_p, cfg, bmode, seed))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (out_root / "meta" / f"{t[0]:02d}_{t[4]}_{t[8]}.json").exists()]
    if not todo:
        print(f"[exp49 s{args.shard}] all {len(my_tasks)} tasks already done!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    for pid, dom, img_path, cid, rname, prompt, neg_p, cfg, bmode, seed in todo:
        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=im, prompt=prompt, negative_prompt=neg_p,
            height=512, width=512, num_frames=49, num_inference_steps=36,
            guidance_scale=cfg, guidance_scale_2=cfg, generator=gen,
        ).frames[0]

        pil_frames = e31b._to_pil_frames(frames)
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))
        st.update(e37a._grammar_diagnostics(np_frames, inp_u8))

        key = f"{pid:02d}_{rname}_{bmode}"
        e36._save_single_mp4(pil_frames, out_root / "mp4" / f"{key}.mp4", fps=16)
        e31b._save_strip(
            pil_frames, out_root / "strips" / f"{key}.png",
            f"{bmode} | {pid:02d}_{rname} ({cid}) | dL(g={st['global_luma_delta']:+.2f} h={st['hair_luma_delta']:+.2f} up={st['upper_face_luma_delta']:+.2f} lo={st['lower_face_luma_delta']:+.2f}) shR={st['sharp_ratio_vs_inp']:.2f}",
            n_show=6, thumb_h=170,
        )
        kf_dir = out_root / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        T = len(pil_frames)
        for k_i in range(6):
            pil_frames[round(k_i * (T - 1) / 5)].save(kf_dir / f"k{k_i}.png")

        rec = {
            "pair_idx": pid, "idx": pid, "domain": dom, "char_id": cid, "regime": rname,
            "mode": bmode, "cfg": cfg, "seed": seed, "stats": st,
        }
        (out_root / "meta" / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[exp49 s{args.shard}] {key:46s} | dL_g={st['global_luma_delta']:+.3f} dL_up={st['upper_face_luma_delta']:+.3f} dL_lo={st['lower_face_luma_delta']:+.3f} shR={st['sharp_ratio_vs_inp']:.2f}",
            flush=True,
        )
    return 0


def stage_summary(args) -> int:
    out_root = args.exp40_dir / "eval_2x2"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = [
        "M0_Base27B",
        "B1_StrongNegPrompt",
        "B2_HighCFG_6.0",
        "B3_LowCFG_2.0",
        "B4_HyperExplicitPrompt",
        "M4_HighLoRA_v2",
        "M5_DualLoRA_v2",
    ]
    rnames = [e40.EVAL_REGIMES[ri][0] for ri in TARGET_REGIMES]
    summary = {}
    for rname in rnames:
        summary[rname] = {}
        for m in modes:
            recs = []
            for pid in range(args.n_test_chars):
                p = out_root / "meta" / f"{pid:02d}_{rname}_{m}.json"
                if p.exists():
                    recs.append(json.loads(p.read_text(encoding="utf-8"))["stats"])
            if recs:
                summary[rname][m] = {
                    "n": len(recs),
                    "dL_glob": round(float(np.mean([r["global_luma_delta"] for r in recs])), 4),
                    "dL_hair": round(float(np.mean([r["hair_luma_delta"] for r in recs])), 4),
                    "dL_up": round(float(np.mean([r["upper_face_luma_delta"] for r in recs])), 4),
                    "dL_low": round(float(np.mean([r["lower_face_luma_delta"] for r in recs])), 4),
                    "shR": round(float(np.mean([r["sharp_ratio_vs_inp"] for r in recs])), 4),
                }
    (args.out_dir / "summary_exp49_baselines.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Stack visual comparison sheets for pid=0 and pid=2 across all 5 regimes
    for pid in [0, 2]:
        for rname in rnames:
            ps = [out_root / "strips" / f"{pid:02d}_{rname}_{m}.png" for m in modes]
            ps = [p for p in ps if p.exists()]
            if len(ps) >= 3:
                e31b._stack_strips(ps, args.out_dir / f"qc49_baselines_{pid:02d}_{rname}.png")
    print(f"[exp49] Saved summary & QC sheets in {args.out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["eval", "summary"])
    ap.add_argument("--n-test-chars", type=int, default=4)
    ap.add_argument("--exp40-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp40_wan22_a14b_moe_lora"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp49_training_free_baselines"))
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    return stage_eval(args) if args.stage == "eval" else stage_summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
