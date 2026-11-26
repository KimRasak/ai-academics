#!/usr/bin/env python3
"""
Exp 42 (Post-LoRA Mechanistic Verification on Flagship 27B MoE `Wan2.2-I2V-A14B`):

Closes the causal & mechanistic loop between Exp 38 (Pre-Fix Diagnosis) and Exp 40 (Dual-MoE LoRA Fix):
Runs the EXACT same `CrossAttnProbeProcessor` (`attn2` spatial localization of trigger tokens),
per-zone CFG velocity (`||v_cond - v_uncond||` on hair / upper_face / lower_face / torso + `leak_ratio`),
and MoE Handoff `x0` latent decoding (right at `s=9`, `t>=900` High->Low boundary) across:
  - `M0_Base27B`   (frozen 27B MoE)
  - `M1_LoRA_High` (trained 14B High-Noise Expert LoRA only)
  - `M3_LoRA_Dual` (trained Dual-MoE LoRA)
on UNSEEN held-out characters (`fc_00`, `fc_02`) across:
  `S3_NonBioAffect`, `G1_HighlightOff_Only`, `G3_AngerVein_Pop`, `G4_SweatDrop_Slide`, `C1_Comp_HeadTurn_Shadow`.

Measures quantitatively & visually HOW the 14B High-Noise LoRA rewired internal 27B MoE attention & CFG velocity!
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp31b_video_dit_anime_vs_real_diagnostic as e31b
import exp32b_wan22_causal_dit_surgery as e32b
import exp37a_flagship_2d_grammar_benchmark as e37a
import exp38_wan22_a14b_moe_mechanistic_probe as e38


POST_PROBE_REGIMES = [
    (
        "S3_NonBioAffect",
        "The character's bright eye highlights suddenly vanish leaving hollow dark lifeless eyes while a sharp ominous dark shadow falls across their upper forehead, yet their mouth stays frozen in a calm eerie smile, 2D Japanese TV anime cel shading.",
        ["shadow", "dark", "hollow", "lifeless", "ominous"],
    ),
    (
        "G1_HighlightOff_Only",
        "The character stays still in the exact same bright even studio lighting with unchanged bright hair, clothes, and gentle smile, while only the small white specular highlight dots inside both eyes smoothly fade away leaving flat matte dull irises, clean 2D Japanese TV anime.",
        ["highlight", "fade", "matte", "dull", "eyes"],
    ),
    (
        "G3_AngerVein_Pop",
        "In bright flat 2D Japanese anime style, the character holds a polite closed-mouth smile as a classic stylized red cross-shaped anime anger vein symbol pops onto their upper forehead temple and pulses gently, everything else stays bright, clean, and consistent.",
        ["red", "cross", "anger", "vein", "symbol"],
    ),
    (
        "G4_SweatDrop_Slide",
        "In bright flat 2D Japanese anime style, the character gives an awkward stiff smile as a single classic translucent light-blue anime sweat drop symbol appears beside their upper forehead temple and slides gently downward, clean 2D cel shading.",
        ["sweat", "drop", "symbol", "temple", "blue"],
    ),
    (
        "C1_Comp_HeadTurn_Shadow",
        "The character smoothly turns their head from front view to a three-quarter side profile while keeping a calm smile as eye highlights vanish leaving dull flat irises and a sharp translucent dark 2D anime gloom shadow falls strictly across the upper forehead, bright unchanged hair and white studio background, clean 2D Japanese TV anime.",
        ["shadow", "gloom", "dark", "forehead", "dull"],
    ),
]


def _get_blocks(model):
    if hasattr(model, "blocks"):
        return model.blocks
    return model.base_model.model.blocks


def _install_probes(pipe, state: dict, probe_layers: list[int]):
    for expert_tag, model in [("high", pipe.transformer), ("low", pipe.transformer_2)]:
        blocks = _get_blocks(model)
        for li in probe_layers:
            if li < len(blocks):
                attn2 = blocks[li].attn2
                if not isinstance(attn2.processor, e38.CrossAttnProbeProcessor):
                    attn2.set_processor(e38.CrossAttnProbeProcessor(attn2.processor, expert_tag, li, state))


@torch.no_grad()
def stage_run(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from peft import PeftModel

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    sel_pids = [0, 2]
    modes = ["M0_Base27B", "M1_LoRA_High", "M3_LoRA_Dual"]

    tasks = []
    for pid in sel_pids:
        p = pairs[pid]
        for rname, rprompt, kws in POST_PROBE_REGIMES:
            for mode in modes:
                tasks.append((pid, Path(p["anime_path"]), p["anime_id"], rname, rprompt, kws, mode))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    meta_dir = args.out_dir / "meta"
    handoff_dir = args.out_dir / "handoff"
    strips_dir = args.out_dir / "strips"
    for d in [meta_dir, handoff_dir, strips_dir]:
        d.mkdir(parents=True, exist_ok=True)

    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[3]}_{t[6]}.json").exists()]
    if not todo:
        print(f"[exp42 s{args.shard}] all tasks complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_dir / "lora_high_r16")
    pipe.transformer_2 = PeftModel.from_pretrained(pipe.transformer_2, args.lora_dir / "lora_low_r16")
    pipe.enable_model_cpu_offload()

    probe_state = {"active": False, "xattn": {}}
    probe_layers = [8, 16, 24, 32]
    _install_probes(pipe, probe_state, probe_layers)
    probe_steps = {0, 3, 6, 9, 12, 18, 26, 34}

    for pid, img_path, cid, rname, rprompt, kws, mode in todo:
        if mode == "M0_Base27B":
            pipe.transformer.disable_adapter_layers()
            pipe.transformer_2.disable_adapter_layers()
        elif mode == "M1_LoRA_High":
            pipe.transformer.enable_adapter_layers()
            pipe.transformer_2.disable_adapter_layers()
        elif mode == "M3_LoRA_Dual":
            pipe.transformer.enable_adapter_layers()
            pipe.transformer_2.enable_adapter_layers()

        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        seed = 9400 + pid * 43 + [r[0] for r in POST_PROBE_REGIMES].index(rname)
        trig_idxs = e38._find_trigger_token_idxs(pipe.tokenizer, rprompt, kws)
        probe_state.clear()
        probe_state.update({"active": False, "trigger_idxs": trig_idxs, "xattn": {}, "save_handoff": True})

        pil_frames, diag = e38._run_custom_wan22_a14b(
            pipe, im, rprompt, rprompt, seed=seed,
            height=512, width=512, num_frames=49, num_steps=36,
            guidance_scale=3.5, mode="standard",
            probe_state=probe_state, probe_steps=probe_steps,
        )

        # Save handoff x0 last-frame snapshot right at High->Low MoE boundary (s=9)
        ho_img = diag.pop("handoff_pil", None)
        key = f"{pid:02d}_{rname}_{mode}"
        if ho_img is not None:
            ho_img.save(handoff_dir / f"{key}_handoff_s9.png")
        pil_frames[-1].save(handoff_dir / f"{key}_final.png")

        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))
        st.update(e37a._grammar_diagnostics(np_frames, inp_u8))

        # Aggregate High-Noise (s<=9) vs Low-Noise (s>9) CFG velocity & xattn localization
        step_logs = diag["steps"]
        high_logs = [s for s in step_logs if s["expert"] == "high"]
        low_logs = [s for s in step_logs if s["expert"] == "low"]

        def _mean_zone(logs, k):
            return round(float(np.mean([l["cfg_zones"][k] for l in logs])), 4) if logs else 0.0

        vel_summary = {
            "high_hair": _mean_zone(high_logs, "hair"),
            "high_upper_face": _mean_zone(high_logs, "upper_face"),
            "high_lower_face": _mean_zone(high_logs, "lower_face"),
            "high_torso": _mean_zone(high_logs, "torso"),
            "high_leak_ratio": _mean_zone(high_logs, "leak_ratio"),
            "low_hair": _mean_zone(low_logs, "hair"),
            "low_upper_face": _mean_zone(low_logs, "upper_face"),
            "low_lower_face": _mean_zone(low_logs, "lower_face"),
            "low_torso": _mean_zone(low_logs, "torso"),
            "low_leak_ratio": _mean_zone(low_logs, "leak_ratio"),
        }

        xattn_raw = probe_state.get("xattn", {})
        xa_high_up, xa_high_hair, xa_high_low = [], [], []
        xa_low_up, xa_low_hair, xa_low_low = [], [], []
        for tag, layers in xattn_raw.items():
            for _li, rec in layers.items():
                if tag.startswith("high_"):
                    xa_high_up.append(rec["upper_face_frac"])
                    xa_high_hair.append(rec["hair_frac"])
                    xa_high_low.append(rec["lower_face_frac"])
                else:
                    xa_low_up.append(rec["upper_face_frac"])
                    xa_low_hair.append(rec["hair_frac"])
                    xa_low_low.append(rec["lower_face_frac"])

        xattn_summary = {
            "high_upper_face_frac": round(float(np.mean(xa_high_up)), 4) if xa_high_up else 0.0,
            "high_hair_frac": round(float(np.mean(xa_high_hair)), 4) if xa_high_hair else 0.0,
            "high_lower_face_frac": round(float(np.mean(xa_high_low)), 4) if xa_high_low else 0.0,
            "low_upper_face_frac": round(float(np.mean(xa_low_up)), 4) if xa_low_up else 0.0,
            "low_hair_frac": round(float(np.mean(xa_low_hair)), 4) if xa_low_hair else 0.0,
            "low_lower_face_frac": round(float(np.mean(xa_low_low)), 4) if xa_low_low else 0.0,
            "n_trigger_tokens": len(trig_idxs),
        }

        e31b._save_strip(
            pil_frames,
            strips_dir / f"{key}.png",
            f"Exp42 {key} | HiVel(up={vel_summary['high_upper_face']:.2f},hair={vel_summary['high_hair']:.2f},lk={vel_summary['high_leak_ratio']:.2f}) | XA_up={xattn_summary['high_upper_face_frac']:.3f}",
            n_show=6, thumb_h=168,
        )

        rec = {
            "pair_idx": pid, "char_id": cid, "regime": rname, "mode": mode, "seed": seed,
            "vel_summary": vel_summary, "xattn_summary": xattn_summary,
            "stats": st, "step_logs": step_logs,
        }
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(
            f"[exp42 s{args.shard}] {key:36s} | HiVel(hair={vel_summary['high_hair']:.2f} up={vel_summary['high_upper_face']:.2f} low={vel_summary['high_lower_face']:.2f} lk={vel_summary['high_leak_ratio']:.2f}) "
            f"| XA_hi(up={xattn_summary['high_upper_face_frac']:.3f}) | dL_low={st['lower_face_luma_delta']:+.2f}",
            flush=True,
        )
    return 0


def stage_summarize(args) -> int:
    meta_dir = args.out_dir / "meta"
    handoff_dir = args.out_dir / "handoff"
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(meta_dir.glob("*.json"))]
    print(f"\n=== EXP 42 POST-LORA MECHANISTIC SUMMARY (n={len(recs)}) ===", flush=True)
    summary = {}
    for rname, _, _ in POST_PROBE_REGIMES:
        print(f"\n--- {rname} ---", flush=True)
        summary[rname] = {}
        for mode in ["M0_Base27B", "M1_LoRA_High", "M3_LoRA_Dual"]:
            sub = [r for r in recs if r["regime"] == rname and r["mode"] == mode]
            if not sub:
                continue
            row = {
                "n": len(sub),
                "high_hair_vel": round(float(np.mean([r["vel_summary"]["high_hair"] for r in sub])), 4),
                "high_up_vel": round(float(np.mean([r["vel_summary"]["high_upper_face"] for r in sub])), 4),
                "high_low_vel": round(float(np.mean([r["vel_summary"]["high_lower_face"] for r in sub])), 4),
                "high_leak_ratio": round(float(np.mean([r["vel_summary"]["high_leak_ratio"] for r in sub])), 4),
                "xa_high_up": round(float(np.mean([r["xattn_summary"]["high_upper_face_frac"] for r in sub])), 4),
                "xa_high_hair": round(float(np.mean([r["xattn_summary"]["high_hair_frac"] for r in sub])), 4),
                "xa_low_up": round(float(np.mean([r["xattn_summary"]["low_upper_face_frac"] for r in sub])), 4),
                "dL_hair": round(float(np.mean([r["stats"]["hair_luma_delta"] for r in sub])), 4),
                "dL_up": round(float(np.mean([r["stats"]["upper_face_luma_delta"] for r in sub])), 4),
                "dL_low": round(float(np.mean([r["stats"]["lower_face_luma_delta"] for r in sub])), 4),
            }
            summary[rname][mode] = row
            print(
                f"  {mode:13s} | HiCFG(hair={row['high_hair_vel']:.3f} up={row['high_up_vel']:.3f} low={row['high_low_vel']:.3f} leak={row['high_leak_ratio']:.3f}) "
                f"| XA_hi_up={row['xa_high_up']:.4f} | dL(hair={row['dL_hair']:+.3f} up={row['dL_up']:+.3f} low={row['dL_low']:+.3f})",
                flush=True,
            )

    # Build visual side-by-side MoE Handoff (s=9) vs Final comparison sheet for each char!
    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    for pid in [0, 2]:
        inp = Image.open(pairs[pid]["anime_path"]).convert("RGB").resize((224, 224), Image.LANCZOS)
        cols = [
            "Input (t=0)",
            "Base27B s=9 Handoff", "Base27B Final",
            "LoRA_High s=9 Handoff", "LoRA_High Final",
            "LoRA_Dual s=9 Handoff", "LoRA_Dual Final",
        ]
        sz = 224
        W = len(cols) * sz
        H = 34 + len(POST_PROBE_REGIMES) * (sz + 24)
        canvas = Image.new("RGB", (W, H), (18, 18, 22))
        dr = ImageDraw.Draw(canvas)
        for ci, lbl in enumerate(cols):
            dr.text((ci * sz + 8, 10), lbl, fill=(245, 245, 245))
        for ri, (rname, _, _) in enumerate(POST_PROBE_REGIMES):
            y0 = 34 + ri * (sz + 24)
            dr.text((8, y0 + 4), f"Unseen Char {pid:02d} | {rname}", fill=(255, 220, 120))
            canvas.paste(inp, (0, y0 + 22))
            for mi, mode in enumerate(["M0_Base27B", "M1_LoRA_High", "M3_LoRA_Dual"]):
                ho_p = handoff_dir / f"{pid:02d}_{rname}_{mode}_handoff_s9.png"
                fn_p = handoff_dir / f"{pid:02d}_{rname}_{mode}_final.png"
                if ho_p.exists():
                    canvas.paste(Image.open(ho_p).convert("RGB").resize((sz, sz), Image.LANCZOS), ((1 + mi * 2) * sz, y0 + 22))
                if fn_p.exists():
                    canvas.paste(Image.open(fn_p).convert("RGB").resize((sz, sz), Image.LANCZOS), ((2 + mi * 2) * sz, y0 + 22))
        canvas.save(args.out_dir / f"qc42_post_lora_handoff_{pid:02d}.png")

    (args.out_dir / "summary_exp42.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary_exp42.json and qc42_post_lora_handoff_*.png in {args.out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "summarize"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--lora-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp40_wan22_a14b_moe_lora"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp42_post_lora_mechanistic"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_run(args) if args.stage == "run" else stage_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
