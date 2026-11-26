#!/usr/bin/env python3
"""
Exp 17B (P3 BIDIRECTIONAL SYMMETRY PROOF — Positive Non-Biological Anime Symbols
         Overridden by a Neutral / Deadpan Biological Mouth!):

Falsifies the rival hypothesis ("VLMs just have a positivity / cheerful bias"):
  Generates 16 clean NEUTRAL / DEADPAN-MOUTH TV-anime portraits (nm_00..nm_15) and injects
  THREE canonical POSITIVE non-biological anime affect symbols with ZERO mouth change (mouth_l1=0.0):
    1. `orig_neutral`    -> A: Calm, neutral, stoic, or unimpressed
    2. `placebo_hairpin` -> A: Calm, neutral, stoic, or unimpressed (pixel-matched neutral control)
    3. `star_eyes_on`    -> B: Ecstatically amazed, thrilled, or starry-eyed with awe (`✦_✦`)
    4. `heart_eyes_on`   -> C: Infatuated, smitten, or head-over-heels in love (`♥_♥`)
    5. `sparkle_aura_on` -> B: Radiant joy / starry admiration (`✨` shoujo manga sparkles)

Tests across 4 diverse SOTA VLMs (InternVL3-8B, Qwen2.5-VL-7B, LLaVA-OneVision-7B, Qwen3-VL-32B):
  - Gate 2 Text Knowledge (proves models KNOW `✦_✦`, `♥_♥`, `✨` override a neutral mouth!)
  - Direct 1-Token vs Audit 1-Token (proves Neutral Biological Mouth STILL overrides POSITIVE symbols!)
  - Zero-Shot Leave-Positive-Symbols-Out Hidden-State Probe (trained ONLY on `orig_neutral` vs `sparkle_aura_on`,
    tested ZERO-SHOT on UNSEEN `star_eyes_on` & `heart_eyes_on`!)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))

NEUTRAL_CHARACTERS = [
    ("nm_00_navy_girl", "1girl, solo, short navy blue bob hair, bright cyan eyes, neutral expression, closed straight mouth, school uniform"),
    ("nm_01_pink_girl", "1girl, solo, pastel pink twin tails hair, emerald green eyes, deadpan neutral face, closed straight mouth, white collar blouse"),
    ("nm_02_silver_boy", "1boy, solo, messy silver hair, vivid amber eyes, stoic calm expression, closed flat mouth, dark blue jacket"),
    ("nm_03_blonde_girl", "1girl, solo, long golden blonde hair, sapphire blue eyes, expressionless calm face, closed straight mouth, green cardigan"),
    ("nm_04_black_girl", "1girl, solo, long straight black hair, ruby red eyes, stoic neutral expression, closed flat mouth, cream sweater"),
    ("nm_05_orange_girl", "1girl, solo, short bright orange hair, teal eyes, calm neutral face, closed straight mouth, yellow hoodie"),
    ("nm_06_violet_girl", "1girl, solo, wavy lavender hair, violet eyes, expressionless neutral face, closed straight mouth, lilac blouse"),
    ("nm_07_brown_boy", "1boy, solo, neat chestnut brown hair, warm golden eyes, calm stoic face, closed flat mouth, white collared shirt"),
    ("nm_08_mint_girl", "1girl, solo, mint green hair, golden yellow eyes, neutral expression, closed straight mouth, sailor uniform"),
    ("nm_09_crimson_girl", "1girl, solo, crimson red ponytail hair, bright blue eyes, stoic deadpan face, closed flat mouth, dark turtleneck"),
    ("nm_10_white_girl", "1girl, solo, snowy white short hair, rose pink eyes, calm expressionless face, closed straight mouth, red scarf"),
    ("nm_11_teal_boy", "1boy, solo, dark teal hair, turquoise eyes, neutral stoic face, closed flat mouth, varsity jacket"),
    ("nm_12_auburn_girl", "1girl, solo, warm auburn hair, bright green eyes, calm neutral face, closed straight mouth, denim jacket"),
    ("nm_13_cobalt_boy", "1boy, solo, cobalt blue hair, bright orange eyes, expressionless stoic face, closed flat mouth, black hoodie"),
    ("nm_14_peach_girl", "1girl, solo, peach blonde bob hair, sky blue eyes, calm neutral expression, closed straight mouth, striped shirt"),
    ("nm_15_plum_girl", "1girl, solo, deep plum hair, bright gold eyes, deadpan neutral face, closed straight mouth, beige blazer"),
]

COMMON_POS = (
    "anime coloring, tv anime screenshot, vibrant full color, upper body portrait, "
    "front view, looking at viewer, open eyes, bright detailed eyes, white eye highlights, "
    "neutral flat mouth, no smile, clean cel shading, even studio lighting, simple light cream background, "
    "masterpiece, high score, great score, absurdres"
)
COMMON_NEG = (
    "smile, smiling, grin, happy, laughing, blush, open mouth, teeth, frown, angry, crying, "
    "sparkles, stars, hearts, star-shaped pupils, heart-shaped pupils, "
    "sketch, lineart, monochrome, greyscale, 3d, photorealistic, heavy shadow, dark background, side view, closed eyes, text, watermark"
)

LABELS_POS = ["A", "B", "C"]
EXPECTED_POS = {
    "orig_neutral": "A",
    "placebo_hairpin": "A",
    "star_eyes_on": "B",
    "sparkle_aura_on": "B",
    "heart_eyes_on": "C",
}

DIRECT_POS_PROMPT = (
    "Look at this 2D anime character portrait and judge their TRUE emotional / psychological state.\n"
    "Choose the SINGLE best label:\n"
    "A) Calm, neutral, stoic, or unimpressed\n"
    "B) Ecstatically amazed, thrilled, or starry-eyed with radiant admiration\n"
    "C) Infatuated, smitten, or head-over-heels in romantic adoration\n"
    "Answer with ONLY a single letter: A, B, or C."
)

AUDIT_POS_PROMPT = (
    "Inspect this 2D anime portrait carefully for ANY positive non-biological anime graphic affect symbols "
    "(such as bright four-point star-shaped pupils, glowing pink heart-shaped pupils, or floating shoujo manga sparkles around the face), "
    "ignoring ordinary hairpins.\n"
    "In 2D anime, graphic affect symbols OVERRIDE a neutral or flat mouth.\n"
    "Choose the SINGLE best label:\n"
    "A) Calm, neutral, stoic, or unimpressed\n"
    "B) Ecstatically amazed, thrilled, or starry-eyed with radiant admiration\n"
    "C) Infatuated, smitten, or head-over-heels in romantic adoration\n"
    "Answer with ONLY a single letter: A, B, or C."
)

GATE2_TEXT_QUESTIONS = [
    (
        "star_eyes_text",
        "In 2D anime/manga visual grammar, if a character has a calm closed mouth BUT bright golden four-point star-shaped pupils (`✦_✦`) glowing inside both eyes, what is their TRUE emotional state?\n"
        "A) Calm, neutral, stoic, or unimpressed\n"
        "B) Ecstatically amazed, thrilled, or starry-eyed with radiant admiration\n"
        "C) Infatuated, smitten, or head-over-heels in romantic adoration\n"
        "Answer with ONLY a single letter: A, B, or C.",
        "B",
    ),
    (
        "heart_eyes_text",
        "In 2D anime/manga visual grammar, if a character has a calm closed mouth BUT vivid pink heart-shaped pupils (`♥_♥`) glowing inside both eyes, what is their TRUE emotional state?\n"
        "A) Calm, neutral, stoic, or unimpressed\n"
        "B) Ecstatically amazed, thrilled, or starry-eyed with radiant admiration\n"
        "C) Infatuated, smitten, or head-over-heels in romantic adoration\n"
        "Answer with ONLY a single letter: A, B, or C.",
        "C",
    ),
    (
        "sparkle_aura_text",
        "In 2D anime/manga visual grammar, if a character has a calm closed mouth BUT floating golden shoujo manga four-point sparkles (`✨`) glittering around their face, what is their TRUE emotional state?\n"
        "A) Calm, neutral, stoic, or unimpressed\n"
        "B) Ecstatically amazed, thrilled, or starry-eyed with radiant admiration\n"
        "C) Infatuated, smitten, or head-over-heels in romantic adoration\n"
        "Answer with ONLY a single letter: A, B, or C.",
        "B",
    ),
]


def _draw_4point_star(draw: ImageDraw.ImageDraw, cx: float, cy: float, r_out: float, r_in: float, fill, outline=None):
    pts = []
    for k in range(8):
        ang = k * math.pi / 4.0
        r = r_out if k % 2 == 0 else r_in
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    draw.polygon(pts, fill=fill, outline=outline)


def _draw_heart(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, fill, outline=None):
    pts = []
    for i in range(48):
        t = 2.0 * math.pi * i / 48.0
        hx = 16.0 * (math.sin(t) ** 3)
        hy = -(13.0 * math.cos(t) - 5.0 * math.cos(2 * t) - 2.0 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((cx + hx * size / 18.0, cy + hy * size / 18.0))
    draw.polygon(pts, fill=fill, outline=outline)


def _to_px(box_1000: list[int], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, int(round(box_1000[0] * w / 1000.0))))
    y1 = max(0, min(h - 1, int(round(box_1000[1] * h / 1000.0))))
    x2 = max(x1 + 4, min(w, int(round(box_1000[2] * w / 1000.0))))
    y2 = max(y1 + 4, min(h, int(round(box_1000[3] * h / 1000.0))))
    return x1, y1, x2, y2


def _locate_iris_centers(img: Image.Image, boxes: dict[str, list[int]]) -> list[tuple[float, float, float]]:
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    res = []
    for key in ["left_eye", "right_eye"]:
        x1, y1, x2, y2 = _to_px(boxes[key], w, h)
        bw, bh = x2 - x1, y2 - y1
        patch = arr[y1:y2, x1:x2]
        cmax = patch.max(axis=2) / 255.0
        cmin = patch.min(axis=2) / 255.0
        sat = np.where(cmax > 1e-5, (cmax - cmin) / np.maximum(cmax, 1e-5), 0.0)
        colored = (sat > 0.22) & (cmax > 0.14) & (cmax < 0.96)
        if colored.sum() >= 30:
            ys, xs = np.where(colored)
            cx = x1 + float(np.median(xs))
            cy = y1 + float(np.median(ys))
            r = max(10.0, float(np.percentile(xs, 92) - np.percentile(xs, 8)) * 0.42)
        else:
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            r = max(10.0, min(bw, bh) * 0.32)
        res.append((cx, cy, r))
    return res


def _add_star_eyes(img: Image.Image, boxes: dict[str, list[int]]) -> Image.Image:
    """Draw canonical anime golden/white 4-point starry-eyed pupils (`✦_✦`) strictly inside both irises."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dg = ImageDraw.Draw(glow)
    d = ImageDraw.Draw(overlay)
    for cx, cy, r in _locate_iris_centers(img, boxes):
        _draw_4point_star(dg, cx, cy, r * 1.12, r * 0.36, fill=(255, 224, 64, 210))
        _draw_4point_star(d, cx, cy, r * 1.00, r * 0.30, fill=(255, 232, 82, 255), outline=(210, 130, 10, 255))
        _draw_4point_star(d, cx, cy, r * 0.62, r * 0.18, fill=(255, 255, 245, 255))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=5))
    out = Image.alpha_composite(img.convert("RGBA"), glow)
    return Image.alpha_composite(out, overlay).convert("RGB")


def _add_heart_eyes(img: Image.Image, boxes: dict[str, list[int]]) -> Image.Image:
    """Draw canonical anime vivid pink glowing heart-shaped pupils (`♥_♥`) strictly inside both irises."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dg = ImageDraw.Draw(glow)
    d = ImageDraw.Draw(overlay)
    for cx, cy, r in _locate_iris_centers(img, boxes):
        _draw_heart(dg, cx, cy + r * 0.05, r * 0.92, fill=(255, 70, 145, 200))
        _draw_heart(d, cx, cy + r * 0.05, r * 0.80, fill=(255, 42, 128, 255), outline=(175, 12, 78, 255))
        _draw_heart(d, cx - r * 0.14, cy - r * 0.08, r * 0.28, fill=(255, 220, 238, 235))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=5))
    out = Image.alpha_composite(img.convert("RGBA"), glow)
    return Image.alpha_composite(out, overlay).convert("RGB")


def _add_sparkle_aura(img: Image.Image, boxes: dict[str, list[int]]) -> Image.Image:
    """Draw canonical shoujo-manga golden floating 4-point sparkles (`✨`) framing the cheeks/temples (strictly away from mouth)."""
    w, h = img.size
    fx1, fy1, fx2, fy2 = _to_px(boxes["face"], w, h)
    fw = fx2 - fx1
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dg = ImageDraw.Draw(glow)
    d = ImageDraw.Draw(overlay)

    spots = [
        (fx1 - 0.12 * fw, fy1 + 0.18 * fw, 0.11 * fw),
        (fx1 - 0.18 * fw, fy1 + 0.42 * fw, 0.075 * fw),
        (fx1 - 0.06 * fw, fy1 + 0.56 * fw, 0.065 * fw),
        (fx2 + 0.12 * fw, fy1 + 0.16 * fw, 0.115 * fw),
        (fx2 + 0.18 * fw, fy1 + 0.40 * fw, 0.08 * fw),
        (fx2 + 0.08 * fw, fy1 + 0.55 * fw, 0.065 * fw),
    ]
    for sx, sy, sr in spots:
        sx = max(sr + 4, min(w - sr - 4, sx))
        sy = max(sr + 4, min(h - sr - 4, sy))
        _draw_4point_star(dg, sx, sy, sr * 1.25, sr * 0.34, fill=(255, 218, 65, 195))
        _draw_4point_star(d, sx, sy, sr, sr * 0.24, fill=(255, 238, 110, 255), outline=(228, 155, 20, 255))
        _draw_4point_star(d, sx, sy, sr * 0.52, sr * 0.14, fill=(255, 255, 255, 255))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=6))
    out = Image.alpha_composite(img.convert("RGBA"), glow)
    return Image.alpha_composite(out, overlay).convert("RGB")


def stage_gen(args) -> int:
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    pdir = args.out_dir / "portraits_neutral16"
    pdir.mkdir(parents=True, exist_ok=True)
    pipe = DiffusionPipeline.from_pretrained(args.sdxl_dir, torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    items = [c for i, c in enumerate(NEUTRAL_CHARACTERS) if i % args.num_shards == args.shard]
    for cid, desc in items:
        idx = int(cid.split("_")[1])
        p = pdir / f"{cid}.png"
        if p.exists():
            continue
        gen = torch.Generator(device="cuda").manual_seed(6200 + idx * 41)
        img = pipe(
            prompt=f"{desc}, {COMMON_POS}",
            negative_prompt=COMMON_NEG,
            width=1024,
            height=1024,
            guidance_scale=5.0,
            num_inference_steps=28,
            generator=gen,
        ).images[0]
        img.save(p)
        print(f"[neutral shard {args.shard}] saved {p.name}", flush=True)
    return 0


def stage_build(args) -> int:
    import exp2b_p3_counterfactual_suite as exp2b

    pdir = args.out_dir / "portraits_neutral16"
    paths = sorted(pdir.glob("nm_*.png"))
    assert len(paths) == 16, f"Expected 16 neutral portraits, got {len(paths)}"

    boxes_file = args.out_dir / "face_boxes_neutral16.json"
    if boxes_file.exists():
        boxes_map = json.loads(boxes_file.read_text(encoding="utf-8"))
    else:
        boxes_map = exp2b._detect_face_boxes_vlm(paths, args.qwen_dir)
        boxes_file.write_text(json.dumps(boxes_map, indent=2), encoding="utf-8")

    vdir = args.out_dir / "variants_pos5"
    sdir = args.out_dir / "qc_sheets"
    vdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)

    records = []
    max_mouth_l1 = 0.0
    for idx, p in enumerate(paths):
        cid = p.stem
        boxes = boxes_map[cid]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        mx1, my1, mx2, my2 = exp2b._to_px(boxes["mouth"], w, h)
        orig_arr = np.asarray(img, dtype=np.float32)
        orig_mouth = orig_arr[my1:my2, mx1:mx2]

        img_pl, _ = exp2b._add_placebo_hairpin(img, boxes)
        img_star = _add_star_eyes(img, boxes)
        img_heart = _add_heart_eyes(img, boxes)
        img_sp = _add_sparkle_aura(img, boxes)

        variants = [
            ("orig_neutral", img),
            ("placebo_hairpin", img_pl),
            ("star_eyes_on", img_star),
            ("heart_eyes_on", img_heart),
            ("sparkle_aura_on", img_sp),
        ]
        cdir = vdir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        for vname, vimg in variants:
            out_p = cdir / f"{vname}.png"
            vimg.save(out_p)
            varr = np.asarray(vimg, dtype=np.float32)
            mouth_l1 = float(np.mean(np.abs(varr[my1:my2, mx1:mx2] - orig_mouth)))
            max_mouth_l1 = max(max_mouth_l1, mouth_l1)
            records.append({
                "char_id": cid,
                "variant": vname,
                "expected": EXPECTED_POS[vname],
                "image_path": str(out_p),
                "mouth_l1": round(mouth_l1, 5),
            })

        if idx < 4:
            tile = 340
            sheet = Image.new("RGB", (tile * 5, tile), (255, 255, 255))
            for vi, (_vn, vimg) in enumerate(variants):
                sheet.paste(vimg.resize((tile, tile), Image.LANCZOS), (vi * tile, 0))
            sheet.save(sdir / f"{cid}_pos5_sheet.png")

    mp = args.out_dir / "manifest_pos5.jsonl"
    mp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"DONE: 16 neutral chars x 5 variants = {len(records)} imgs | max_mouth_l1={max_mouth_l1:.5f} -> {mp}")
    return 0


def _get_lm_layers(model):
    for path in [("model", "language_model"), ("language_model", "model"), ("language_model",), ("model",)]:
        cur = model
        ok = True
        for a in path:
            if hasattr(cur, a):
                cur = getattr(cur, a)
            else:
                ok = False
                break
        if ok and hasattr(cur, "layers"):
            return cur.layers
    raise AttributeError("Cannot locate LM layers")


def _fit_logreg(X: np.ndarray, y: np.ndarray, reg: float = 5.0):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    w = np.zeros(Xn.shape[1], dtype=np.float32)
    b = 0.0
    for _ in range(400):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
        err = p - y
        w -= 0.05 * ((Xn.T @ err) / len(y) + reg * w / len(y))
        b -= 0.05 * float(np.mean(err))
    return w, b


def _score(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -30, 30)))


@torch.no_grad()
def stage_eval(args) -> int:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    mp = args.out_dir / "manifest_pos5.jsonl"
    recs = [json.loads(l) for l in mp.read_text(encoding="utf-8").splitlines() if l.strip()]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    abc_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in LABELS_POS]
    layers = _get_lm_layers(model)
    nL = len(layers)
    probe_layers = [int(nL * 0.55), int(nL * 0.75)]

    # 1. Gate 2 Text Knowledge check (text-only)
    gate2 = {}
    for qname, qtext, exp_lbl in GATE2_TEXT_QUESTIONS:
        m = [{"role": "user", "content": [{"type": "text", "text": qtext}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], return_tensors="pt").to(model.device)
        out = model(**inp)
        lg = out.logits[0, -1, abc_ids].float().cpu().numpy()
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        pred = LABELS_POS[int(np.argmax(lg))]
        gate2[qname] = {"expected": exp_lbl, "pred": pred, "correct": pred == exp_lbl, "probs": {LABELS_POS[i]: round(float(pr[i]), 4) for i in range(3)}}

    # 2. Direct + Audit + Hidden-state extraction at probe_layers
    cap: dict[int, np.ndarray] = {}
    hooks = []
    for li in probe_layers:
        def _mk(idx):
            def _fn(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                cap[idx] = h[0, -1, :].detach().float().cpu().numpy()
            return _fn
        hooks.append(layers[li].register_forward_hook(_mk(li)))

    def eval_img(img: Image.Image, prompt: str):
        cap.clear()
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[img], return_tensors="pt").to(model.device)
        out = model(**inp)
        lg = out.logits[0, -1, abc_ids].float().cpu().numpy()
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        h = np.concatenate([cap[li] for li in probe_layers], axis=0)
        return lg, pr, h

    dir_rows, aud_rows = [], []
    feats = []
    for r in recs:
        img = Image.open(r["image_path"]).convert("RGB")
        lg_d, pr_d, h_d = eval_img(img, DIRECT_POS_PROMPT)
        lg_a, pr_a, _ = eval_img(img, AUDIT_POS_PROMPT)
        dir_rows.append({"char_id": r["char_id"], "variant": r["variant"], "expected": r["expected"], "logits": lg_d, "probs": pr_d, "pred": LABELS_POS[int(np.argmax(lg_d))]})
        aud_rows.append({"char_id": r["char_id"], "variant": r["variant"], "expected": r["expected"], "logits": lg_a, "probs": pr_a, "pred": LABELS_POS[int(np.argmax(lg_a))]})
        feats.append(h_d)

    for hk in hooks:
        hk.remove()
    feats = np.stack(feats, axis=0)

    def summarize(rows):
        by_v = {}
        for v in EXPECTED_POS:
            sub = [x for x in rows if x["variant"] == v]
            by_v[v] = {
                "acc": round(float(np.mean([x["pred"] == x["expected"] for x in sub])), 4),
                "mean_pA_neutral_override": round(float(np.mean([float(x["probs"][0]) for x in sub])), 4),
            }
        clean = float(np.mean([by_v["orig_neutral"]["acc"], by_v["placebo_hairpin"]["acc"]]))
        pos3 = float(np.mean([by_v["star_eyes_on"]["acc"], by_v["heart_eyes_on"]["acc"], by_v["sparkle_aura_on"]["acc"]]))
        return {"macro5": round(float(np.mean([x["acc"] for x in by_v.values()])), 4), "clean": round(clean, 4), "pos3": round(pos3, 4), "by_variant": by_v}

    # 3. Leave-Positive-Symbols-Out Probe (train ONLY on 10 chars x {orig_neutral, placebo_hairpin} vs {sparkle_aura_on},
    #    test ZERO-SHOT on 6 held-out chars including UNSEEN `star_eyes_on` & `heart_eyes_on`!)
    chars = sorted({r["char_id"] for r in recs})
    tr_c, te_c = set(chars[:10]), set(chars[10:])
    tr_idx = [i for i, r in enumerate(recs) if r["char_id"] in tr_c and r["variant"] in {"orig_neutral", "placebo_hairpin", "sparkle_aura_on"}]
    y_tr = np.array([0.0 if recs[i]["variant"] in {"orig_neutral", "placebo_hairpin"} else 1.0 for i in tr_idx], dtype=np.float32)
    w, b = _fit_logreg(feats[tr_idx], y_tr)
    sc_tr = _score(feats[tr_idx], w, b)
    best_t, best_bal = 0.5, -1.0
    for t in np.linspace(float(sc_tr.min()) - 0.01, float(sc_tr.max()) + 0.01, 101):
        p = (sc_tr >= t).astype(np.float32)
        bal = 0.5 * float(np.mean(p[y_tr == 0] == 0)) + 0.5 * float(np.mean(p[y_tr == 1] == 1))
        if bal > best_bal:
            best_bal, best_t = bal, float(t)

    # Clean-prior debiasing mean from clean train rows only
    clean_tr_idx = [i for i in tr_idx if recs[i]["variant"] in {"orig_neutral", "placebo_hairpin"}]
    mean_clean_aud = np.mean([aud_rows[i]["logits"] for i in clean_tr_idx], axis=0)

    te_idx = [i for i, r in enumerate(recs) if r["char_id"] in te_c]
    sc_te = _score(feats[te_idx], w, b)
    gated_te_rows = []
    fire_by_v: dict[str, list[int]] = {v: [] for v in EXPECTED_POS}
    for j, i in enumerate(te_idx):
        fired = bool(sc_te[j] >= best_t)
        fire_by_v[recs[i]["variant"]].append(int(fired))
        if not fired:
            pred = "A"
        else:
            adj = aud_rows[i]["logits"] - mean_clean_aud
            # Restrict to non-A positive classes {B, C} when gate fires
            pred = LABELS_POS[1 + int(np.argmax(adj[1:]))]
        gated_te_rows.append({"char_id": recs[i]["char_id"], "variant": recs[i]["variant"], "expected": recs[i]["expected"], "probs": aud_rows[i]["probs"], "pred": pred})

    dir_te = summarize([dir_rows[i] for i in te_idx])
    aud_te = summarize([aud_rows[i] for i in te_idx])
    gat_te = summarize(gated_te_rows)

    res = {
        "model": Path(args.model_dir).name,
        "gate2_text_knowledge": gate2,
        "gate2_acc": round(float(np.mean([v["correct"] for v in gate2.values()])), 4),
        "all16_direct": summarize(dir_rows),
        "all16_audit": summarize(aud_rows),
        "test6_direct": dir_te,
        "test6_audit": aud_te,
        "test6_gated_lso_probe": gat_te,
        "test6_gate_fire_rate": {v: round(float(np.mean(xs)), 4) for v, xs in fire_by_v.items()},
    }
    out_p = args.out_dir / f"exp17b_pos_sym_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n=== EXP 17B BIDIRECTIONAL POSITIVE SYMBOLS: {res['model']} (Gate2={res['gate2_acc']:.2f}) ===", flush=True)
    print(f"  All16 Direct: macro5={res['all16_direct']['macro5']:.3f} clean={res['all16_direct']['clean']:.3f} pos3={res['all16_direct']['pos3']:.3f} | "
          f"star={res['all16_direct']['by_variant']['star_eyes_on']['acc']:.2f}(pA={res['all16_direct']['by_variant']['star_eyes_on']['mean_pA_neutral_override']:.2f}) "
          f"heart={res['all16_direct']['by_variant']['heart_eyes_on']['acc']:.2f}(pA={res['all16_direct']['by_variant']['heart_eyes_on']['mean_pA_neutral_override']:.2f}) "
          f"sparkle={res['all16_direct']['by_variant']['sparkle_aura_on']['acc']:.2f}(pA={res['all16_direct']['by_variant']['sparkle_aura_on']['mean_pA_neutral_override']:.2f})", flush=True)
    print(f"  Test6 Direct={dir_te['macro5']:.3f} | Audit={aud_te['macro5']:.3f} | OURS Gated-LSO={gat_te['macro5']:.3f} (fire={res['test6_gate_fire_rate']})", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "build", "eval"], required=True)
    ap.add_argument("--sdxl-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/animagine-xl-4.0")
    ap.add_argument("--qwen-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=4)
    args = ap.parse_args()
    if args.stage == "gen":
        return stage_gen(args)
    if args.stage == "build":
        return stage_build(args)
    return stage_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
