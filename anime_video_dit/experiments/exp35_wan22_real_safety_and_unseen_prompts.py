#!/usr/bin/env python3
"""
Exp 35 (Video DiT Track — Real-Human Safety Control + Unseen Anime Prompt Generalization on `Wan2.2-TI2V-5B`!):

Answers the final two Reviewer-#2 questions for our Zero-Training Video DiT Surgery (`OURS_EarlyDual`):
  1. **Real-Human Safety Control** (`n = 8` matched Real Humans x 2 regimes `R1_HeadTurn`, `R2_DynamicWind` = 16 pairs):
     Does `OURS_EarlyDual` degrade Photorealistic Human videos at all, or is it 100% safe / even beneficial
     on real humans (requiring ZERO domain classifier)?
  2. **Zero-Shot Unseen Anime Motion Generalization** (`n = 8` Anime Chars x 3 BRAND-NEW unseen prompts = 24 pairs):
     - `U1_CameraOrbit`:      3D camera orbit around the character
     - `U2_HairTossLaugh`:    Dynamic head-tilt laugh with bouncing hair
     - `U3_TurnWaveWink`:     Side turn + waving hand + winking smile

Evaluates BOTH quantitative metrics (`Luma-CLIP ID`, `Sharpness`, `Edge Boiling`, `SatOvershoot`, `Motion`)
AND Order-Balanced Blind `Qwen3-VL-32B-Instruct` Pairwise Visual Judge across ALL 40 pairs (80 A/B queries)!
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp31b_video_dit_anime_vs_real_diagnostic as e31b
import exp32a_wan22_dit_mechanistic_probe as e32a
import exp32b_wan22_causal_dit_surgery as e32b
import exp33_wan22_decoupled_flow_anchor_ablation as e33
import exp34b_wan22_early_winner_and_layer_localization as e34b


UNSEEN_PROMPTS = [
    (
        "U1_CameraOrbit",
        "The camera smoothly orbits horizontally around the character while they blink softly and hold a warm smile, crisp consistent 2D anime art style.",
    ),
    (
        "U2_HairTossLaugh",
        "The character tilts their head joyfully and laughs brightly as their hair bounces and flows dynamically in the breeze, lively 2D anime animation.",
    ),
    (
        "U3_TurnWaveWink",
        "The character turns gently to the side, raises one hand to wave warmly at the viewer, and winks with a cheerful smile, fluid 2D anime motion.",
    ),
]


def _build_tasks(manifest: Path):
    pairs = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()][:8]
    tasks = []
    # Part A: Real-Human Safety Control (8 real chars x 2 regimes x 2 methods = 32 clips)
    for p in pairs:
        for rname, rprompt in e32a.REGIMES_32:
            for spec in e34b.SCALE16_METHODS:
                tasks.append(("real_ctrl", p["pair_idx"], Path(p["real_path"]), p["real_id"], rname, rprompt, spec))
    # Part B: Unseen Anime Prompts (8 anime chars x 3 unseen prompts x 2 methods = 48 clips)
    for p in pairs:
        for rname, rprompt in UNSEEN_PROMPTS:
            for spec in e34b.SCALE16_METHODS:
                tasks.append(("unseen_anime", p["pair_idx"], Path(p["anime_path"]), p["anime_id"], rname, rprompt, spec))
    return tasks


@torch.no_grad()
def stage_run(args) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    tasks = _build_tasks(args.paired_manifest)
    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    meta_dir = args.out_dir / "meta"
    strips_dir = args.out_dir / "strips"
    meta_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)

    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]}_{t[1]:02d}_{t[4]}_{t[6][0]}.json").exists()]
    if not todo:
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    attn_ctrl = {"beta": 0.0, "l_min": 0, "l_max": 30, "T_lat": 13, "step_active": False}
    for li, blk in enumerate(pipe.transformer.blocks):
        blk.attn1.set_processor(e32b.AnchorBoostWanAttnProcessor(blk.attn1.processor, li, attn_ctrl))

    for group, pair_idx, img_path, cid, rname, rprompt, spec in todo:
        mname = spec[0]
        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        seed = 7300 + pair_idx * 17 + abs(hash(rname)) % 97

        pil_frames = e33._run_decoupled_wan22(pipe, im, rprompt, seed, spec, attn_ctrl)
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))

        key = f"{group}_{pair_idx:02d}_{rname}_{mname}"
        e31b._save_strip(
            pil_frames, strips_dir / f"{key}.png",
            f"{mname:15s} | {key} | mot={st['motion_l1']:.3f} shR={st['sharp_ratio_vs_inp']:.2f} boil={st['edge_boiling']:.3f} satOv={st['sat_overshoot']:+.3f}",
        )
        kf_dir = args.out_dir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        for fi in [0, 12, 24, 36, len(pil_frames) - 1]:
            pil_frames[fi].save(kf_dir / f"f{fi:02d}.png")

        rec = {"group": group, "pair_idx": pair_idx, "char_id": cid, "regime": rname, "prompt": rprompt, "method": mname, "stats": st}
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[exp35 s{args.shard}] {key:45s} | mot={st['motion_l1']:.4f} shR={st['sharp_ratio_vs_inp']:.3f} boil={st['edge_boiling']:.4f} satOv={st['sat_overshoot']:+.3f}", flush=True)
    return 0


@torch.no_grad()
def stage_eval_and_judge(args) -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor, CLIPModel, CLIPProcessor

    meta_dir = args.out_dir / "meta"
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(meta_dir.glob("*.json"))]

    # 1. Quantitative Luma-CLIP ID
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.bfloat16).to("cuda").eval()
    for r in recs:
        key = f"{r['group']}_{r['pair_idx']:02d}_{r['regime']}_{r['method']}"
        kf_dir = args.out_dir / "keyframes" / key
        ims = [ImageOps.autocontrast(ImageOps.grayscale(Image.open(kf_dir / fn).convert("RGB")), cutoff=1).convert("RGB") for fn in ["input.png", "f12.png", "f24.png", "f36.png", "f48.png"]]
        pv = cp(images=ims, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)
        f = cm.get_image_features(pixel_values=pv)
        if not isinstance(f, torch.Tensor): f = f.pooler_output
        e = F.normalize(f.float(), dim=-1)
        r["luma_id"] = round(float((e[1:] @ e[0]).mean().item()), 4)
        r["luma_min"] = round(float((e[1:] @ e[0]).min().item()), 4)
    del cm; torch.cuda.empty_cache()

    print("\n=== EXP 35 QUANTITATIVE SUMMARY (Real-Human Safety Control + 3 Unseen Anime Prompts) ===", flush=True)
    quant = {}
    for group in ["real_ctrl", "unseen_anime"]:
        quant[group] = {}
        regimes = sorted({r["regime"] for r in recs if r["group"] == group})
        for rname in regimes:
            quant[group][rname] = {}
            for mname in ["Base", "OURS_EarlyDual"]:
                sub = [r for r in recs if r["group"] == group and r["regime"] == rname and r["method"] == mname]
                if not sub: continue
                m = {
                    "n": len(sub),
                    "mot": round(float(np.mean([r["stats"]["motion_l1"] for r in sub])), 4),
                    "luma_id": round(float(np.mean([r["luma_id"] for r in sub])), 4),
                    "luma_min": round(float(np.mean([r["luma_min"] for r in sub])), 4),
                    "shR": round(float(np.mean([r["stats"]["sharp_ratio_vs_inp"] for r in sub])), 4),
                    "boil": round(float(np.mean([r["stats"]["edge_boiling"] for r in sub])), 4),
                    "satOv": round(float(np.mean([r["stats"]["sat_overshoot"] for r in sub])), 4),
                }
                quant[group][rname][mname] = m
                print(f"  {group:12s} | {rname:16s} | {mname:14s} | mot={m['mot']:.4f} LumaID={m['luma_id']:.4f}(min={m['luma_min']:.4f}) shR={m['shR']:.3f} boil={m['boil']:.4f} satOv={m['satOv']:+.3f}", flush=True)

    # Stack unseen anime strips (char 00..03) for visual QC
    for rname, _ in UNSEEN_PROMPTS:
        paths = []
        for pid in range(4):
            paths.append(args.out_dir / "strips" / f"unseen_anime_{pid:02d}_{rname}_Base.png")
            paths.append(args.out_dir / "strips" / f"unseen_anime_{pid:02d}_{rname}_OURS_EarlyDual.png")
        e31b._stack_strips(paths, args.out_dir / f"qc35_{rname}.png")

    (args.out_dir / "exp35_quant_summary.json").write_text(json.dumps(quant, indent=2), encoding="utf-8")
    return 0


@torch.no_grad()
def stage_judge(args) -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.judge_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.judge_model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    ab_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in ["A", "B"]]

    def make_strip(kf_dir: Path) -> Image.Image:
        ims = [Image.open(kf_dir / fn).convert("RGB").resize((224, 224), Image.LANCZOS) for fn in ["f12.png", "f24.png", "f36.png", "f48.png"]]
        c = Image.new("RGB", (896, 224))
        for i, im in enumerate(ims): c.paste(im, (i * 224, 0))
        return c

    def judge_ab(ref, sA, sB, txt):
        m = [{"role": "user", "content": [{"type": "image", "image": ref}, {"type": "image", "image": sA}, {"type": "image", "image": sB}, {"type": "text", "text": txt}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[ref, sA, sB], return_tensors="pt").to(model.device)
        return int(np.argmax(model(**inp).logits[0, -1, ab_ids].float().cpu().numpy()))

    groups = [
        ("real_ctrl", e32a.REGIMES_32),
        ("unseen_anime", UNSEEN_PROMPTS),
    ]
    out_j = {}
    print(f"\n=== EXP 35 BLIND ORDER-BALANCED VLM JUDGE ({Path(args.judge_model).name}) ===", flush=True)
    for group, rlist in groups:
        out_j[group] = {}
        tot_w = tot_l = tot_t = q_o = q_n = 0
        for rname, rprompt in rlist:
            ptxt = (
                f"Image 1 is the reference starting frame. Image 2 is 4 frames (t=12,24,36,48) of Video A, and Image 3 is 4 frames of Video B for prompt: '{rprompt}'.\n"
                "Which video (A or B) is BETTER OVERALL in preserving exact identity and clean art/photo style WITHOUT distortion, color explosions, or weird artifacts, while still showing natural motion?\n"
                "Answer ONLY a single letter: A or B."
            )
            w = l = t = qo = qn = 0
            for pid in range(8):
                kb = args.out_dir / "keyframes" / f"{group}_{pid:02d}_{rname}_Base"
                ko = args.out_dir / "keyframes" / f"{group}_{pid:02d}_{rname}_OURS_EarlyDual"
                if not kb.exists() or not ko.exists(): continue
                ref = Image.open(kb / "input.png").convert("RGB").resize((336, 336), Image.LANCZOS)
                sb, so = make_strip(kb), make_strip(ko)
                p1 = judge_ab(ref, so, sb, ptxt)
                p2 = judge_ab(ref, sb, so, ptxt)
                qo += int(p1 == 0) + int(p2 == 1); qn += 2
                if p1 == 0 and p2 == 1: w += 1
                elif p1 == 1 and p2 == 0: l += 1
                else: t += 1
            tot_w += w; tot_l += l; tot_t += t; q_o += qo; q_n += qn
            out_j[group][rname] = {"win_rate": round(qo / max(1, qn), 4), "W_L_T": [w, l, t], "p": round(e34b._sign_test_p(w, l), 5)}
            print(f"  {group:12s} | {rname:16s} | OURS WinRate={qo/max(1,qn)*100:5.1f}% (W/L/T={w}/{l}/{t}, p={e34b._sign_test_p(w,l):.5f})", flush=True)
        out_j[group]["OVERALL"] = {"win_rate": round(q_o / max(1, q_n), 4), "W_L_T": [tot_w, tot_l, tot_t], "p": round(e34b._sign_test_p(tot_w, tot_l), 5)}
        print(f"  >>> {group:12s} OVERALL | OURS WinRate={q_o/max(1,q_n)*100:5.1f}% (W/L/T={tot_w}/{tot_l}/{tot_t}, p={e34b._sign_test_p(tot_w, tot_l):.5f})", flush=True)

    (args.out_dir / "exp35_vlm_judge_summary.json").write_text(json.dumps(out_j, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "eval", "judge"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp35_wan22_safety_unseen"))
    ap.add_argument("--judge-model", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen3-VL-32B-Instruct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        return stage_run(args)
    if args.stage == "eval":
        return stage_eval_and_judge(args)
    return stage_judge(args)


if __name__ == "__main__":
    raise SystemExit(main())
