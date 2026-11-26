#!/usr/bin/env python3
"""
Exp 19B (P3 Cross-Generator Architecture OOD Stress Test — ZERO Animagine-XL Overlap!):

Eliminates any concern that Biological-Override or our mid-layer anomaly gate is specific
to Animagine-XL-4.0's texture/style:
  Generates TWO independent native anime trope test suites using completely different
  open-weight generative backbones:
    1. `Illustrious-XL-v2.0-diffusers` (distinct anime model family, 16 native trope imgs)
    2. `stable-diffusion-3.5-medium`   (distinct MMDiT flow-matching architecture — ZERO UNet/SDXL!, 16 native trope imgs)

  Each suite has 4 tropes x 4 diverse characters = 16 images (32 total cross-generator imgs):
    - `cheerful`         -> A (clean smiling anime portrait)
    - `empty_eyes_smile` -> B (hollow lifeless dead eyes + smiling mouth)
    - `shaded_face_smile`-> B (dark ominous forehead gloom shadow + smiling mouth)
    - `sweatdrop_smile`  -> D (awkward floating anime sweat drop + smiling mouth)

  Then trains our frozen mid-layer anomaly gate ONLY on `Animagine-XL-4.0` synthetic counterfactuals
  (`forehead_shadow_on`, `sweat_drop_on` vs `orig`, `placebo_hairpin`) and evaluates ZERO-SHOT
  cross-generator transfer (`Animagine-XL -> Illustrious-XL-v2.0` & `Animagine-XL -> SD3.5-Medium`)
  with BOTH 1-Pass Unified Surgery (1.0x pass) and 2-Pass Debiased Binder!
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

CHARS4 = [
    ("c0_blue_girl", "anime girl with short blue hair, school uniform"),
    ("c1_red_boy", "anime boy with messy red hair, dark jacket"),
    ("c2_blonde_girl", "anime girl with long golden blonde hair, green cardigan"),
    ("c3_black_girl", "anime girl with straight black hair, cream sweater"),
]

TROPES4 = [
    (
        "native_cheerful",
        "A",
        "bright sparkling detailed eyes with white eye highlights, happy cheerful smile, bright even lighting",
        "empty eyes, dead eyes, shaded face, dark forehead shadow, sweatdrop, horror, eerie, sad, angry",
    ),
    (
        "native_empty_eyes_smile",
        "B",
        "empty eyes, blank lifeless dull dark eyes with NO white eye highlights, yandere eerie unsettling smile, cheerful smiling mouth",
        "sparkling eyes, white eye highlights, bright catchlights, crying, frown, angry",
    ),
    (
        "native_shaded_face_smile",
        "B",
        "gloomy shaded upper face, heavy dark shadow cast over forehead and eyes, vertical gloom lines on forehead, ominous menacing smile, smiling mouth",
        "bright forehead, even lighting, cheerful, blush, sweatdrop",
    ),
    (
        "native_sweatdrop_smile",
        "D",
        "large blue anime sweatdrop on temple, awkward nervous strained smile, uneasy embarrassed expression",
        "shaded face, dark shadow, horror, angry, empty eyes",
    ),
]


def stage_gen(args) -> int:
    out_gen = args.out_dir / args.gen_name
    out_gen.mkdir(parents=True, exist_ok=True)
    sheets_dir = args.out_dir / "qc_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    if args.gen_name == "illustrious_v2":
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        pipe = DiffusionPipeline.from_pretrained(args.gen_dir, torch_dtype=torch.float16, use_safetensors=True)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        style_prefix = "masterpiece, best quality, anime coloring, 2d tv anime portrait, upper body, front view, looking at viewer, simple light background, "
        steps, cfg = 26, 5.5
    else:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(args.gen_dir, torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
        style_prefix = "2D Japanese TV anime screenshot, vibrant cel-shaded anime character portrait, front view looking at viewer, clean ink lineart, simple light background, "
        steps, cfg = 28, 5.0

    records = []
    tiles_by_char: dict[str, list[Image.Image]] = {}
    for ci, (cid, cdesc) in enumerate(CHARS4):
        tiles_by_char[cid] = []
        for ti, (tname, exp_l, tpos, tneg) in enumerate(TROPES4):
            p = out_gen / f"{args.gen_name}_{cid}_{tname}.png"
            if not p.exists():
                gen = torch.Generator(device="cuda").manual_seed(7700 + ci * 97 + ti * 19 + (0 if args.gen_name == "illustrious_v2" else 500))
                prompt = f"{style_prefix}{cdesc}, {tpos}"
                neg = f"3d, photorealistic, realistic, photo, monochrome, sketch, lowres, bad anatomy, text, watermark, {tneg}"
                img = pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    width=1024,
                    height=1024,
                    guidance_scale=cfg,
                    num_inference_steps=steps,
                    generator=gen,
                ).images[0]
                img.save(p)
                print(f"[{args.gen_name}] saved {p.name}", flush=True)
            else:
                img = Image.open(p).convert("RGB")
            tiles_by_char[cid].append(img)
            records.append({
                "generator": args.gen_name,
                "char_id": f"{args.gen_name}_{cid}",
                "variant": tname,
                "expected": exp_l,
                "image_path": str(p),
            })

    # Save 4x4 contact sheet for visual QC
    tile = 320
    sheet = Image.new("RGB", (tile * 4, tile * 4), (255, 255, 255))
    for ci, (cid, _) in enumerate(CHARS4):
        for ti, im in enumerate(tiles_by_char[cid]):
            sheet.paste(im.resize((tile, tile), Image.LANCZOS), (ti * tile, ci * tile))
    sheet.save(sheets_dir / f"{args.gen_name}_4x4_sheet.png")

    mp = args.out_dir / f"manifest_{args.gen_name}.jsonl"
    mp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"[{args.gen_name}] DONE: {len(records)} cross-generator native anime images -> {mp}", flush=True)
    return 0


@torch.no_grad()
def stage_eval(args) -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import exp17c_scale48_5fold_and_unified_1pass as e17c

    recs_tr = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs_tr_seen = [r for r in recs_tr if r["variant"] in e17c.SEEN_TRAIN_VARS]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ids4 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS4]

    layers, final_norm, lm_head = e17c._get_lm(model)
    nL = len(layers)
    l_mid = int(nL * 0.55)
    l_late_gate = int(nL * 0.72)
    l_surg = l_late_gate + 1

    W4 = lm_head.weight[ids4].detach().float()
    wb4 = W4[0] - W4[1:].mean(dim=0)
    if final_norm is not None and hasattr(final_norm, "weight"):
        wb4 = wb4 * final_norm.weight.detach().float()
    w_bio4 = (wb4 / (wb4.norm() + 1e-8)).to(model.device)

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
                proj = torch.dot(v, w_bio4)
                if proj.item() > 0.0:
                    h = h.clone()
                    h[0, -1, :] = (v - 1.5 * proj * w_bio4).to(h.dtype)
                    return (h,) + out[1:] if isinstance(out, tuple) else h
                return out
            return _fn
        hooks.append(layers[li].self_attn.register_forward_hook(_mk_s()))
        hooks.append(layers[li].mlp.register_forward_hook(_mk_s()))

    def encode(img: Image.Image, prompt: str):
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        return processor(text=[t], images=[img], return_tensors="pt").to(model.device)

    # 1. Fit Animagine-only gate + clean-prior means on first 24 Animagine chars (zero cross-gen data!)
    chars24 = set(sorted({r["char_id"] for r in recs_tr_seen})[:24])
    X_tr, y_tr = [], []
    clean_dir, clean_surg, clean_aud = [], [], []
    for r in recs_tr_seen:
        if r["char_id"] not in chars24:
            continue
        img = Image.open(r["image_path"]).convert("RGB")
        inp4 = encode(img, e17c.DIRECT4_PROMPT)
        cap.clear(); surg_on["on"] = False
        lg_d = model(**inp4).logits[0, -1, ids4].float().cpu().numpy()
        X_tr.append(np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0))
        is_c = r["variant"] not in {"orig", "placebo_hairpin"}
        y_tr.append(1.0 if is_c else 0.0)
        if not is_c:
            clean_dir.append(lg_d)
            surg_on["on"] = True
            clean_surg.append(model(**inp4).logits[0, -1, ids4].float().cpu().numpy())
            surg_on["on"] = False
            clean_aud.append(model(**encode(img, e17c.AUDIT4_PROMPT)).logits[0, -1, ids4].float().cpu().numpy())

    X_tr = np.stack(X_tr, axis=0)
    y_tr = np.array(y_tr, dtype=np.float32)
    w, b = e17c._fit_logreg(X_tr, y_tr)
    tau = e17c._cal_thresh(e17c._score(X_tr, w, b), y_tr)
    mu_d, mu_s, mu_a = np.mean(clean_dir, axis=0), np.mean(clean_surg, axis=0), np.mean(clean_aud, axis=0)

    # 2. Evaluate ZERO-SHOT on each cross-generator manifest!
    all_gen_res = {}
    for gname in ["illustrious_v2", "sd35_medium"]:
        mp = args.out_dir / f"manifest_{gname}.jsonl"
        if not mp.exists():
            continue
        grecs = [json.loads(l) for l in mp.read_text(encoding="utf-8").splitlines() if l.strip()]
        res_m = {m: [] for m in ["direct", "audit", "ours_1pass_unified", "ours_2pass_debiased"]}
        for r in grecs:
            img = Image.open(r["image_path"]).convert("RGB")
            inp4 = encode(img, e17c.DIRECT4_PROMPT)
            cap.clear(); surg_on["on"] = False
            lg_d = model(**inp4).logits[0, -1, ids4].float().cpu().numpy()
            h = np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0)[None, :]
            fired = bool(e17c._score(h, w, b)[0] >= tau)
            surg_on["on"] = True
            lg_s = model(**inp4).logits[0, -1, ids4].float().cpu().numpy()
            surg_on["on"] = False
            lg_a = model(**encode(img, e17c.AUDIT4_PROMPT)).logits[0, -1, ids4].float().cpu().numpy()

            exp_l = r["expected"]
            res_m["direct"].append((r["variant"], e17c.LABELS4[int(np.argmax(lg_d))] == exp_l))
            res_m["audit"].append((r["variant"], e17c.LABELS4[int(np.argmax(lg_a))] == exp_l))
            p1 = "A" if not fired else e17c.LABELS4[1 + int(np.argmax((lg_s - mu_s)[1:]))]
            p2 = "A" if not fired else e17c.LABELS4[1 + int(np.argmax((0.5 * (lg_a - mu_a) + 0.5 * (lg_d - mu_d))[1:]))]
            res_m["ours_1pass_unified"].append((r["variant"], p1 == exp_l))
            res_m["ours_2pass_debiased"].append((r["variant"], p2 == exp_l))

        gs = {}
        for m, pairs in res_m.items():
            by_v = {v: round(float(np.mean([c for vv, c in pairs if vv == v])), 4) for v in e17c.EXPECTED_NAT}
            gs[m] = {"macro4": round(float(np.mean(list(by_v.values()))), 4), "clean": by_v["native_cheerful"], "by_variant": by_v}
        all_gen_res[gname] = gs
        print(
            f"[{Path(args.model_dir).name}] {gname:15s} | Direct={gs['direct']['macro4']:.3f}(cl={gs['direct']['clean']:.2f}) "
            f"| Audit={gs['audit']['macro4']:.3f}(cl={gs['audit']['clean']:.2f}) "
            f"| OURS-1Pass={gs['ours_1pass_unified']['macro4']:.3f}(cl={gs['ours_1pass_unified']['clean']:.2f}) "
            f"| OURS-2Pass={gs['ours_2pass_debiased']['macro4']:.3f}(cl={gs['ours_2pass_debiased']['clean']:.2f})",
            flush=True,
        )

    for hk in hooks:
        hk.remove()
    out_p = args.out_dir / f"exp19b_xgen_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "results": all_gen_res}, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "eval"], required=True)
    ap.add_argument("--gen-name", choices=["illustrious_v2", "sd35_medium"], default="illustrious_v2")
    ap.add_argument("--gen-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/illustrious/Illustrious-XL-v2.0-diffusers")
    ap.add_argument("--manifest48", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--model-dir", type=str, default="OpenGVLab/InternVL3-8B-hf")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.stage == "gen":
        return stage_gen(args)
    return stage_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
