#!/usr/bin/env python3
"""
Exp 24 (P3 Downstream Generative Application — Best-of-N Anime Affect Generation Reranking!):

Closes the entire scientific loop from Mechanistic Discovery -> Downstream Generative Utility:
  Task: Generate tricky non-biological anime affect portraits with an open T2I diffusion model
        (`Animagine-XL-4.0`) across 15 diverse prompts (5 `empty_eyes + smile`, 5 `sweatdrop + smile`,
        5 `anger_vein + smile`) x N=8 seeds = 120 native candidate images!
        (Note: `empty_eyes` and `anger_vein` are COMPLETELY UNSEEN by our mid-layer anomaly gate!)

  Stage 1 (`--stage gen`): Generates 120 candidate images across 4 H100 GPUs (~20s).
  Stage 2 (`--stage judge`): Independent strong-verifier ground-truth scoring of all 120 candidates
        using `Qwen3-VL-32B` explicit visual-checklist audit (verifies BOTH: (a) target non-biological
        anime trope clearly rendered, AND (b) smiling mouth preserved!).
  Stage 3 (`--stage rerank`): Compares Best-of-8 Selection on lightweight 7B/8B rerankers
        (`InternVL3-8B`, `Qwen2.5-VL-7B`, `LLaVA-OneVision-7B`) at 1.0x inference cost:
          1. `Random / Pass@1` (mean success across all 8 seeds)
          2. `VLM-Direct Reranker` (picks candidate with lowest Direct `P(A: Cheerful)` or highest `P(target)`)
          3. `OURS 1-Pass Mid-Layer-Gated Surgical Reranker` (ranks candidates by mid-layer anomaly score
             `s_mid(h)` + 1-pass surgical target probability — ZERO weight updates, ZERO unseen symbols!)
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

PROMPTS_15 = [
    # 5 x UNSEEN empty_eyes + smile (target: B)
    ("p00_empty_blue_girl", "empty_eyes", "B", "1girl, solo, short blue hair, school uniform, happy smile, empty eyes, lifeless dull eyes, no eye highlights"),
    ("p01_empty_pink_girl", "empty_eyes", "B", "1girl, solo, pastel pink twin tails, white blouse, cheerful smile, empty eyes, blank hollow eyes, no eye highlights"),
    ("p02_empty_silver_boy", "empty_eyes", "B", "1boy, solo, silver hair, dark jacket, gentle smile, empty eyes, dull dead eyes, no eye highlights"),
    ("p03_empty_blonde_girl", "empty_eyes", "B", "1girl, solo, long blonde hair, green cardigan, bright smile, empty eyes, hollow eyes, no eye highlights"),
    ("p04_empty_black_girl", "empty_eyes", "B", "1girl, solo, long straight black hair, cream sweater, sweet smile, empty eyes, lifeless eyes, no eye highlights"),
    # 5 x sweatdrop + smile (target: D)
    ("p05_sweat_orange_girl", "sweatdrop", "D", "1girl, solo, short orange hair, yellow hoodie, awkward smile, sweatdrop on temple, nervous expression"),
    ("p06_sweat_brown_boy", "sweatdrop", "D", "1boy, solo, brown hair, collared shirt, strained cheerful smile, large blue sweatdrop, uneasy"),
    ("p07_sweat_violet_girl", "sweatdrop", "D", "1girl, solo, wavy lavender hair, lilac blouse, polite smile, sweatdrop, awkward nervous"),
    ("p08_sweat_mint_girl", "sweatdrop", "D", "1girl, solo, mint green hair, sailor uniform, smiling, anime sweatdrop on cheek, embarrassed awkward"),
    ("p09_sweat_red_girl", "sweatdrop", "D", "1girl, solo, crimson ponytail, dark turtleneck, smiling mouth, blue sweatdrop, tense awkward"),
    # 5 x UNSEEN anger_vein (💢) + smile (target: E)
    ("p10_vein_white_girl", "anger_vein", "E", "1girl, solo, short white hair, red scarf, polite cheerful smile, anger vein on forehead, anime vein mark, irritated"),
    ("p11_vein_teal_boy", "anger_vein", "E", "1boy, solo, dark teal hair, varsity jacket, smiling mouth, red anger vein symbol on temple, annoyed"),
    ("p12_vein_auburn_girl", "anger_vein", "E", "1girl, solo, auburn hair, denim jacket, sweet smile, anime anger vein on forehead, suppressing anger"),
    ("p13_vein_cobalt_boy", "anger_vein", "E", "1boy, solo, cobalt blue hair, black hoodie, happy grin, anger vein mark on head, irritated"),
    ("p14_vein_plum_girl", "anger_vein", "E", "1girl, solo, plum purple hair, beige blazer, gentle smile, anime red anger vein on temple, annoyed"),
]

COMMON_POS = "anime coloring, tv anime screenshot, vibrant full color, upper body portrait, front view, looking at viewer, clean cel shading, simple light background, masterpiece, high score"
COMMON_NEG = "3d, photorealistic, monochrome, sketch, lineart, bad anatomy, lowres, text, watermark, closed eyes"

CHECKLIST_BY_TROPE = {
    "empty_eyes": (
        "Inspect this 2D anime portrait strictly for TWO visual criteria:\n"
        "1. Do the character's eyes appear visibly EMPTY / HOLLOW / DEAD (missing bright white specular catchlights or dull/darkened lifeless irises)?\n"
        "2. Is the character's mouth smiling or neutral-smiling (NOT frowning/crying)?\n"
        "Does this image clearly satisfy BOTH criteria?\n"
        "Answer ONLY with YES or NO."
    ),
    "sweatdrop": (
        "Inspect this 2D anime portrait strictly for TWO visual criteria:\n"
        "1. Is there a clearly visible anime SWEATDROP (teardrop-shaped blue/white sweat symbol) on the character's face, temple, or hair?\n"
        "2. Is the character's mouth smiling or awkwardly smiling?\n"
        "Does this image clearly satisfy BOTH criteria?\n"
        "Answer ONLY with YES or NO."
    ),
    "anger_vein": (
        "Inspect this 2D anime portrait strictly for TWO visual criteria:\n"
        "1. Is there a clearly visible anime ANGER VEIN symbol (classic red/dark cross-popping '💢' vein mark) on the character's forehead, temple, or hair?\n"
        "2. Is the character's mouth smiling?\n"
        "Does this image clearly satisfy BOTH criteria?\n"
        "Answer ONLY with YES or NO."
    ),
}


def stage_gen(args) -> int:
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    cdir = args.out_dir / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    pipe = DiffusionPipeline.from_pretrained(args.sdxl_dir, torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    items = [p for i, p in enumerate(PROMPTS_15) if i % args.num_shards == args.shard]
    for pid, trope, tgt, desc in items:
        p_idx = int(pid.split("_")[0][1:])
        for k in range(args.n_seeds):
            out_p = cdir / f"{pid}_s{k}.png"
            if out_p.exists():
                continue
            gen = torch.Generator(device="cuda").manual_seed(9100 + p_idx * 100 + k * 13)
            img = pipe(
                prompt=f"{desc}, {COMMON_POS}",
                negative_prompt=COMMON_NEG,
                width=768,
                height=768,
                guidance_scale=5.0,
                num_inference_steps=24,
                generator=gen,
            ).images[0]
            img.save(out_p)
        print(f"[gen shard {args.shard}] finished 8 seeds for {pid}", flush=True)
    return 0


@torch.no_grad()
def stage_judge(args) -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cdir = args.out_dir / "candidates"
    processor = AutoProcessor.from_pretrained(args.judge_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.judge_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    yes_id = processor.tokenizer.encode("YES", add_special_tokens=False)[-1]
    no_id = processor.tokenizer.encode("NO", add_special_tokens=False)[-1]

    records = []
    for pid, trope, tgt, desc in PROMPTS_15:
        q = CHECKLIST_BY_TROPE[trope]
        for k in range(args.n_seeds):
            img_p = cdir / f"{pid}_s{k}.png"
            img = Image.open(img_p).convert("RGB")
            m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
            t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            inp = processor(text=[t], images=[img], return_tensors="pt").to(model.device)
            lg = model(**inp).logits[0, -1, [yes_id, no_id]].float().cpu().numpy()
            pr = np.exp(lg - lg.max()); pr /= pr.sum()
            p_yes = float(pr[0])
            records.append({
                "prompt_id": pid,
                "trope": trope,
                "target_label": tgt,
                "seed_idx": k,
                "image_path": str(img_p),
                "oracle_p_yes": round(p_yes, 4),
                "oracle_pass": bool(p_yes >= 0.5),
            })
        n_ok = sum(r["oracle_pass"] for r in records[-args.n_seeds:])
        print(f"[judge 32B] {pid:22s}: {n_ok}/{args.n_seeds} oracle pass", flush=True)

    mp = args.out_dir / "manifest_bon_judged.jsonl"
    mp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    by_t = {}
    for t in ("empty_eyes", "sweatdrop", "anger_vein"):
        sub = [r for r in records if r["trope"] == t]
        by_t[t] = round(float(np.mean([r["oracle_pass"] for r in sub])), 4)
    print(f"ORACLE PASS@1 RATES: {by_t} | overall={np.mean([r['oracle_pass'] for r in records]):.4f}", flush=True)
    return 0


@torch.no_grad()
def stage_rerank(args) -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import exp17c_scale48_5fold_and_unified_1pass as e17c

    mp = args.out_dir / "manifest_bon_judged.jsonl"
    cand_recs = [json.loads(l) for l in mp.read_text(encoding="utf-8").splitlines() if l.strip()]
    tr_all = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    chars24 = set(sorted({r["char_id"] for r in tr_all})[:24])
    tr_recs = [r for r in tr_all if r["char_id"] in chars24 and r["variant"] in e17c.SEEN_TRAIN_VARS]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ids5 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS5]

    layers, final_norm, lm_head = e17c._get_lm(model)
    nL = len(layers)
    l_mid, l_late_gate = int(nL * 0.55), int(nL * 0.72)
    l_surg = l_late_gate + 1

    W5 = lm_head.weight[ids5].detach().float()
    wb5 = W5[0] - W5[1:].mean(dim=0)
    if final_norm is not None and hasattr(final_norm, "weight"):
        wb5 = wb5 * final_norm.weight.detach().float()
    w_bio5 = (wb5 / (wb5.norm() + 1e-8)).to(model.device)

    cap: dict[int, np.ndarray] = {}
    surg_on = {"on": False}
    hooks = []
    for li in (l_mid, l_late_gate):
        def _mk_c(idx):
            def _fn(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                cap[idx] = h[0, -1, :].detach().float().cpu().numpy()
            return _fn
        hooks.append(layers[li].register_forward_hook(_mk_c(li)))
    for li in range(l_surg, nL):
        def _mk_s():
            def _fn(_m, _i, out):
                if not surg_on["on"]:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                v = h[0, -1, :].float()
                proj = torch.dot(v, w_bio5)
                if proj.item() > 0.0:
                    h = h.clone()
                    h[0, -1, :] = (v - 1.5 * proj * w_bio5).to(h.dtype)
                    return (h,) + out[1:] if isinstance(out, tuple) else h
                return out
            return _fn
        hooks.append(layers[li].self_attn.register_forward_hook(_mk_s()))
        hooks.append(layers[li].mlp.register_forward_hook(_mk_s()))

    def encode(img: Image.Image, prompt: str):
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        return processor(text=[t], images=[img], return_tensors="pt").to(model.device)

    # 1. Fit frozen mid-layer anomaly gate ONLY on 24 synthetic chars x {orig, placebo} vs {forehead_shadow, sweat_drop}
    #    (empty_eyes & anger_vein NEVER SEEN!)
    X_tr, y_tr, clean_surg = [], [], []
    for r in tr_recs:
        img = Image.open(r["image_path"]).convert("RGB")
        inp = encode(img, e17c.DIRECT5_PROMPT)
        cap.clear(); surg_on["on"] = False
        model(**inp)
        X_tr.append(np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0))
        is_c = r["variant"] not in {"orig", "placebo_hairpin"}
        y_tr.append(1.0 if is_c else 0.0)
        if not is_c:
            surg_on["on"] = True
            clean_surg.append(model(**inp).logits[0, -1, ids5].float().cpu().numpy())
    X_tr = np.stack(X_tr, axis=0)
    y_tr = np.array(y_tr, dtype=np.float32)
    w_g, b_g = e17c._fit_logreg(X_tr, y_tr)
    mu_surg = np.mean(clean_surg, axis=0)

    # 2. Score all 120 native candidates with Direct vs OURS 1-Pass Surgical + Mid-Layer Gate (1.0x pass!)
    scored = []
    for r in cand_recs:
        img = Image.open(r["image_path"]).convert("RGB")
        inp = encode(img, e17c.DIRECT5_PROMPT)
        cap.clear(); surg_on["on"] = False
        lg_d = model(**inp).logits[0, -1, ids5].float().cpu().numpy()
        pr_d = np.exp(lg_d - lg_d.max()); pr_d /= pr_d.sum()
        h = np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0)[None, :]
        hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        s_mid = float((hn @ w_g + b_g)[0])

        surg_on["on"] = True
        lg_s = model(**inp).logits[0, -1, ids5].float().cpu().numpy()
        adj_s = lg_s - mu_surg
        pr_s = np.exp(adj_s - adj_s.max()); pr_s /= pr_s.sum()

        t_idx = e17c.LABELS5.index(r["target_label"])
        row = dict(r)
        row["score_direct_target"] = float(pr_d[t_idx])
        row["score_direct_nonA"] = float(1.0 - pr_d[0])
        row["score_ours_mid_gate"] = s_mid
        row["score_ours_1pass_surg_target"] = float(pr_s[t_idx])
        # Combined 1-pass score: standardized mid-layer anomaly + surgical target logit (zero hyperparam tuning!)
        row["score_ours_1pass_combined"] = float(adj_s[t_idx] - adj_s[0]) + 5.0 * s_mid
        scored.append(row)

    for hk in hooks:
        hk.remove()

    # 3. Evaluate Best-of-8 Selection Accuracy & Oracle P(YES) across the 15 prompts!
    by_prompt: dict[str, list[dict]] = {}
    for r in scored:
        by_prompt.setdefault(r["prompt_id"], []).append(r)

    rankers = [
        ("pass1_random", None),
        ("direct_p_target", "score_direct_target"),
        ("direct_nonA", "score_direct_nonA"),
        ("ours_mid_gate_only", "score_ours_mid_gate"),
        ("ours_1pass_surg_target", "score_ours_1pass_surg_target"),
        ("ours_1pass_combined", "score_ours_1pass_combined"),
        ("oracle_upper_bound", "oracle_p_yes"),
    ]
    summary = {}
    print(f"\n=== EXP 24 BEST-OF-8 ANIME GENERATION RERANKING: {Path(args.model_dir).name} (15 prompts x 8 seeds = 120 imgs) ===", flush=True)
    for rname, key in rankers:
        passes_all, pyes_all = [], []
        by_t = {t: [] for t in ("empty_eyes", "sweatdrop", "anger_vein")}
        for pid, group in by_prompt.items():
            trope = group[0]["trope"]
            if key is None:
                p_rate = float(np.mean([g["oracle_pass"] for g in group]))
                p_y = float(np.mean([g["oracle_p_yes"] for g in group]))
            else:
                best = max(group, key=lambda g: g[key])
                p_rate = float(best["oracle_pass"])
                p_y = float(best["oracle_p_yes"])
            passes_all.append(p_rate)
            pyes_all.append(p_y)
            by_t[trope].append(p_rate)
        summary[rname] = {
            "pass_rate": round(float(np.mean(passes_all)), 4),
            "mean_oracle_pyes": round(float(np.mean(pyes_all)), 4),
            "by_trope": {t: round(float(np.mean(xs)), 4) for t, xs in by_t.items()},
        }
        bt = summary[rname]["by_trope"]
        print(f"  {rname:24s} | Pass@Bo8={summary[rname]['pass_rate']:.3f} (pYES={summary[rname]['mean_oracle_pyes']:.3f}) | empty={bt['empty_eyes']:.2f} sweat={bt['sweatdrop']:.2f} vein={bt['anger_vein']:.2f}", flush=True)

    out_p = args.out_dir / f"exp24_bon_rerank_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "summary": summary}, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "judge", "rerank"], required=True)
    ap.add_argument("--sdxl-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/animagine-xl-4.0")
    ap.add_argument("--judge-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen3-VL-32B-Instruct")
    ap.add_argument("--model-dir", type=str, default="OpenGVLab/InternVL3-8B-hf")
    ap.add_argument("--manifest48", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    args = ap.parse_args()
    if args.stage == "gen":
        return stage_gen(args)
    if args.stage == "judge":
        return stage_judge(args)
    return stage_rerank(args)


if __name__ == "__main__":
    raise SystemExit(main())
