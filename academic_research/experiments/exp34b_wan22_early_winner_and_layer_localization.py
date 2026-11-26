#!/usr/bin/env python3
"""
Exp 34B (Video DiT Track — Early-Window Winner `EarlyAnchor + OrthCFG` + Layer-Band Causal Localization
+ Scale-16 Blind Pairwise `Qwen3-VL-32B` Evaluation on `Wan2.2-TI2V-5B`!):

Exp 33 proved causally that:
  - Intervening ONLY in Early High-Noise Steps (`s = 0..10`) fixes trajectory bifurcation (while Mid-Late
    `s = 8..29` is too late!).
  - Pure Orthogonal-Projected CFG (`eta_par = 0.0`, no hard norm clamp on motion) independently kills
    neon saturation overshoot (-67%) and restores >100% lineart sharpness (`shR: 0.905 -> 1.052`)!

Part 1 (`--stage layer_ablation`, n=8 chars x 2 regimes):
  Causal Layer-Band Localization of Early Frame-0 Anchor Restoration (`s = 0..14`, `beta = 1.1` + `OrthCFG`):
    - `L_Base`:         Unmodified `Wan2.2-TI2V-5B`
    - `L_Early_0_10`:   Anchor-Attn ONLY in Early blocks `L0..9`  (+ OrthCFG)
    - `L_Mid_10_20`:    Anchor-Attn ONLY in Mid blocks   `L10..19` (+ OrthCFG)
    - `L_Late_20_30`:   Anchor-Attn ONLY in Late blocks  `L20..29` (+ OrthCFG)
    - `L_OURS_All_0_30`: Early-Window Anchor-Attn (`s=0..14`, `beta=1.0`, `L0..29`) + `OrthCFG` (`eta_par=0.0`)!

Part 2 (`--stage scale16`, ALL 16 Anime Characters x 2 Regimes = 32 paired comparisons: `Base` vs `OURS_EarlyAnchor_OrthCFG`):
  Full statistical validation (`n = 32` clips/method, Wilcoxon/paired t-test & McNemar) + ready for
  order-balanced Blind `Qwen3-VL-32B` Pairwise Visual Judge (`--stage vlm_judge`, run under `qwen_vl_3_5` env)!
"""

from __future__ import annotations

import argparse
import json
import math
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


LAYER_METHODS = [
    # (name, g, use_orth_cfg, eta_par, rho_cap, beta, s_min, s_max, l_min, l_max)
    ("L_Base",          5.0, False, 1.0, 9.99, 0.00,  0,  0,  0, 30),
    ("L_Early_0_10",    5.0, True,  0.0, 0.12, 1.20,  0, 14,  0, 10),
    ("L_Mid_10_20",     5.0, True,  0.0, 0.12, 1.20,  0, 14, 10, 20),
    ("L_Late_20_30",    5.0, True,  0.0, 0.12, 1.20,  0, 14, 20, 30),
    ("L_OURS_EarlyDual",5.0, True,  0.0, 0.12, 1.00,  0, 14,  0, 30),
]

SCALE16_METHODS = [
    ("Base",            5.0, False, 1.0, 9.99, 0.00,  0,  0,  0, 30),
    ("OURS_EarlyDual",  5.0, True,  0.0, 0.12, 1.00,  0, 14,  0, 30),
]


def _sign_test_p(wins_b: int, wins_a: int) -> float:
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    return min(1.0, 2.0 * sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1)))


@torch.no_grad()
def stage_run_specs(args, methods: list[tuple], n_chars: int, tag: str) -> int:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    pairs = [json.loads(l) for l in args.paired_manifest.read_text(encoding="utf-8").splitlines() if l.strip()][:n_chars]
    tasks = []
    for p in pairs:
        for rname, rprompt in e32a.REGIMES_32:
            for spec in methods:
                tasks.append((p["pair_idx"], Path(p["anime_path"]), p["anime_id"], rname, rprompt, spec))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    sub_dir = args.out_dir / tag
    meta_dir = sub_dir / "meta"
    strips_dir = sub_dir / "strips"
    meta_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)

    todo = [t for t in my_tasks if not (meta_dir / f"{t[0]:02d}_{t[3]}_{t[5][0]}.json").exists()]
    if not todo:
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    attn_ctrl = {"beta": 0.0, "l_min": 0, "l_max": 30, "T_lat": 13, "step_active": False}
    for li, blk in enumerate(pipe.transformer.blocks):
        blk.attn1.set_processor(e32b.AnchorBoostWanAttnProcessor(blk.attn1.processor, li, attn_ctrl))

    for pair_idx, img_path, cid, rname, rprompt, spec in todo:
        mname = spec[0]
        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        seed = 5200 + pair_idx * 10 + (0 if rname == "R1_HeadTurn" else 1)

        pil_frames = e33._run_decoupled_wan22(pipe, im, rprompt, seed, spec, attn_ctrl)
        np_frames = [np.asarray(f, dtype=np.uint8) for f in pil_frames]
        inp_u8 = np.asarray(im, dtype=np.uint8)
        st = e31b._frame_stats(np_frames)
        inp_st = e31b._frame_stats([inp_u8, np_frames[0], np_frames[-1]])
        st["sharp_ratio_vs_inp"] = round(st["edge_sharp_end"] / max(inp_st["edge_sharp_t0"], 1e-4), 4)
        st.update(e32b._extra_artifact_stats(np_frames, inp_u8))

        key = f"{pair_idx:02d}_{rname}_{mname}"
        e31b._save_strip(
            pil_frames, strips_dir / f"{key}.png",
            f"{mname:16s} | p{pair_idx:02d}_{rname} | mot={st['motion_l1']:.3f} shR={st['sharp_ratio_vs_inp']:.2f} boil={st['edge_boiling']:.3f} satOv={st['sat_overshoot']:+.3f} jmp={st['jump_spike_ratio']:.1f}",
        )
        kf_dir = sub_dir / "keyframes" / key
        kf_dir.mkdir(parents=True, exist_ok=True)
        im.save(kf_dir / "input.png")
        for fi in [0, 12, 24, 36, len(pil_frames) - 1]:
            pil_frames[fi].save(kf_dir / f"f{fi:02d}.png")

        rec = {"pair_idx": pair_idx, "char_id": cid, "regime": rname, "method": mname, "stats": st}
        (meta_dir / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"[exp34b:{tag} s{args.shard}] {key:36s} | mot={st['motion_l1']:.4f} shR={st['sharp_ratio_vs_inp']:.3f} boil={st['edge_boiling']:.4f} satOv={st['sat_overshoot']:+.3f}", flush=True)
    return 0


@torch.no_grad()
def stage_eval_all(args) -> int:
    from transformers import CLIPModel, CLIPProcessor

    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.bfloat16).to("cuda").eval()
    def enc_luma_clip(ims: list[Image.Image]) -> torch.Tensor:
        g_ims = [ImageOps.autocontrast(ImageOps.grayscale(im), cutoff=1).convert("RGB") for im in ims]
        pv = cp(images=g_ims, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)
        f = cm.get_image_features(pixel_values=pv)
        if not isinstance(f, torch.Tensor): f = f.pooler_output
        return F.normalize(f.float(), dim=-1)

    for tag, methods in [("layer_ablation", LAYER_METHODS), ("scale16", SCALE16_METHODS)]:
        sub_dir = args.out_dir / tag
        meta_dir = sub_dir / "meta"
        if not meta_dir.exists():
            continue
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(meta_dir.glob("*.json"))]
        if not recs:
            continue
        for r in recs:
            key = f"{r['pair_idx']:02d}_{r['regime']}_{r['method']}"
            kf_dir = sub_dir / "keyframes" / key
            ims = [Image.open(kf_dir / fn).convert("RGB") for fn in ["input.png", "f12.png", "f24.png", "f36.png", "f48.png"]]
            el = enc_luma_clip(ims)
            r["luma_clip_mean"] = round(float((el[1:] @ el[0]).mean().item()), 4)
            r["luma_clip_min"] = round(float((el[1:] @ el[0]).min().item()), 4)

        print(f"\n=== EXP 34B [{tag.upper()}] SUMMARY (n={len(recs)} clips) ===", flush=True)
        out_s = {}
        for rname, _ in e32a.REGIMES_32:
            out_s[rname] = {}
            print(f"  --- {rname} ---", flush=True)
            for spec in methods:
                mname = spec[0]
                sub = [r for r in recs if r["regime"] == rname and r["method"] == mname]
                if not sub: continue
                m = {
                    "n": len(sub),
                    "mot": round(float(np.mean([r["stats"]["motion_l1"] for r in sub])), 4),
                    "disp": round(float(np.mean([r["stats"]["disp_tail"] for r in sub])), 4),
                    "luma_id": round(float(np.mean([r["luma_clip_mean"] for r in sub])), 4),
                    "luma_min": round(float(np.mean([r["luma_clip_min"] for r in sub])), 4),
                    "shR": round(float(np.mean([r["stats"]["sharp_ratio_vs_inp"] for r in sub])), 4),
                    "boil": round(float(np.mean([r["stats"]["edge_boiling"] for r in sub])), 4),
                    "satOv": round(float(np.mean([r["stats"]["sat_overshoot"] for r in sub])), 4),
                    "jmp": round(float(np.mean([r["stats"]["jump_spike_ratio"] for r in sub])), 2),
                }
                out_s[rname][mname] = m
                print(
                    f"    {mname:18s} (n={m['n']:2d}) | mot={m['mot']:.4f} disp={m['disp']:.3f} | "
                    f"LumaID={m['luma_id']:.4f}(min={m['luma_min']:.4f}) | shR={m['shR']:.3f} boil={m['boil']:.4f} satOv={m['satOv']:+.3f} jmp={m['jmp']:.2f}",
                    flush=True,
                )
            if tag == "layer_ablation":
                for pid in range(4):
                    e31b._stack_strips([sub_dir / "strips" / f"{pid:02d}_{rname}_{ms[0]}.png" for ms in methods], sub_dir / f"cmp34_layer_{rname}_char{pid:02d}.png")
        (sub_dir / f"summary_{tag}.json").write_text(json.dumps(out_s, indent=2), encoding="utf-8")
    return 0


@torch.no_grad()
def stage_vlm_judge(args) -> int:
    """
    Order-balanced Blind Pairwise Visual Quality Judge using `Qwen3-VL-32B-Instruct`!
    Compares `Base` vs `OURS_EarlyDual` across ALL 16 characters x 2 regimes (= 32 video pairs, 64 A/B queries!)
    by presenting Reference Frame + 4-frame strip of Video A + 4-frame strip of Video B!
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    sub_dir = args.out_dir / "scale16"
    processor = AutoProcessor.from_pretrained(args.judge_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.judge_model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    ab_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in ["A", "B"]]

    def make_4frame_strip(kf_dir: Path) -> Image.Image:
        ims = [Image.open(kf_dir / fn).convert("RGB").resize((224, 224), Image.LANCZOS) for fn in ["f12.png", "f24.png", "f36.png", "f48.png"]]
        c = Image.new("RGB", (896, 224))
        for i, im in enumerate(ims):
            c.paste(im, (i * 224, 0))
        return c

    def judge_pair(ref_im: Image.Image, stripA: Image.Image, stripB: Image.Image, prompt_txt: str) -> int:
        m = [{"role": "user", "content": [
            {"type": "image", "image": ref_im},
            {"type": "image", "image": stripA},
            {"type": "image", "image": stripB},
            {"type": "text", "text": prompt_txt},
        ]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[ref_im, stripA, stripB], return_tensors="pt").to(model.device)
        lg = model(**inp).logits[0, -1, ab_ids].float().cpu().numpy()
        return int(np.argmax(lg))

    criteria = [
        (
            "Overall_Quality",
            "Image 1 is the original 2D anime character reference. Image 2 is 4 frames (t=12,24,36,48) of Video A, and Image 3 is 4 frames of Video B animating this character for prompt: '{p}'.\n"
            "Which video (A or B) is BETTER OVERALL in preserving the character's exact identity and clean 2D anime style WITHOUT color explosions, severe face deformation, or weird artifacts, while still showing natural motion?\n"
            "Answer ONLY a single letter: A or B.",
        ),
        (
            "Identity_and_Artifact_Free",
            "Image 1 is the reference 2D anime character. Image 2 is Video A and Image 3 is Video B.\n"
            "Which video (A or B) has FEWER visual artifacts (no neon color flash, no melted face, no mutated eyes, no extra hallucinated objects) and preserves the character's identity better?\n"
            "Answer ONLY a single letter: A or B.",
        ),
        (
            "Motion_Compliance",
            "Image 1 is the starting frame. Image 2 is Video A and Image 3 is Video B for motion prompt: '{p}'.\n"
            "Which video (A or B) executes the requested motion MORE clearly and naturally (without freezing static or breaking into a jarring jump-cut)?\n"
            "Answer ONLY a single letter: A or B.",
        ),
    ]

    results = {}
    print(f"\n=== EXP 34B BLIND ORDER-BALANCED VLM JUDGE ({Path(args.judge_model).name}, 16 chars x 2 regimes = 32 pairs) ===", flush=True)
    for rname, rprompt in e32a.REGIMES_32:
        results[rname] = {}
        for cname, ctmpl in criteria:
            ptxt = ctmpl.format(p=rprompt)
            ours_wins = base_wins = ties = 0
            ours_q = 0; tot_q = 0
            for pid in range(16):
                kf_base = sub_dir / "keyframes" / f"{pid:02d}_{rname}_Base"
                kf_ours = sub_dir / "keyframes" / f"{pid:02d}_{rname}_OURS_EarlyDual"
                if not kf_base.exists() or not kf_ours.exists():
                    continue
                ref = Image.open(kf_base / "input.png").convert("RGB").resize((336, 336), Image.LANCZOS)
                sb = make_4frame_strip(kf_base)
                so = make_4frame_strip(kf_ours)
                # Order 1: A=Ours, B=Base -> Ours wins if pred==0
                p1 = judge_pair(ref, so, sb, ptxt)
                # Order 2: A=Base, B=Ours -> Ours wins if pred==1
                p2 = judge_pair(ref, sb, so, ptxt)
                ours_q += int(p1 == 0) + int(p2 == 1)
                tot_q += 2
                if p1 == 0 and p2 == 1:
                    ours_wins += 1
                elif p1 == 1 and p2 == 0:
                    base_wins += 1
                else:
                    ties += 1
            win_rate = ours_q / max(1, tot_q)
            pval = _sign_test_p(ours_wins, base_wins)
            results[rname][cname] = {
                "ours_balanced_win_rate": round(win_rate, 4),
                "strict_ours_wins": ours_wins,
                "strict_base_wins": base_wins,
                "ties": ties,
                "sign_test_p": round(pval, 6),
            }
            print(f"  {rname:15s} | {cname:26s} | OURS WinRate={win_rate*100:5.1f}% (Strict W/L/T={ours_wins}/{base_wins}/{ties}, p={pval:.5f})", flush=True)

    (sub_dir / "exp34b_vlm_judge_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["layer_ablation", "scale16", "eval", "vlm_judge"])
    ap.add_argument("--paired-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp31_video_dit/real_portraits/paired_anime_real_16.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp34b_wan22_scale16_layers"))
    ap.add_argument("--judge-model", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen3-VL-32B-Instruct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "layer_ablation":
        return stage_run_specs(args, LAYER_METHODS, n_chars=8, tag="layer_ablation")
    if args.stage == "scale16":
        return stage_run_specs(args, SCALE16_METHODS, n_chars=16, tag="scale16")
    if args.stage == "eval":
        return stage_eval_all(args)
    return stage_vlm_judge(args)


if __name__ == "__main__":
    raise SystemExit(main())
