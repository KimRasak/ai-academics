#!/usr/bin/env python3
"""
Exp 2B (P3: Beyond-FACS Non-Biological Anime Affect Symbols):
  Stage 1 (--build-pairs):
    Uses Qwen2.5-VL-7B-Instruct to localize exact 2D bounding boxes (left_eye, right_eye, mouth, face)
    on clean Animagine-XL-4.0 portraits, then programmatically builds 7 PIXEL-EXACT counterfactual
    variants per character (mouth stays 100% pixel-identical across all variants!):
      1. orig                 (cheerful smile + bright eye catchlights)
      2. eye_highlight_off    (ONLY white specular catchlights inside irises removed -> anime 'dead eyes')
      3. forehead_shadow_on   (anime upper-face vertical dread shadow + fine vertical gloom lines)
      4. dead_eyes_and_shadow (eye_highlight_off + forehead_shadow_on, mouth STILL smiling!)
      5. blush_lines_on       (canonical anime diagonal blush hatch lines '///' + soft cheek tint)
      6. sweat_drop_on        (canonical anime cel-shaded temple sweat drop)
      7. placebo_hairpin      (equal-pixel neutral geometric hairclip -> MUST NOT shift affect!)

  Stage 2 (--eval-vlm <model_dir>):
    Runs the strict 4-Gate VLM Ladder:
      - Gate 1 (Perception): Can the VLM accurately detect each graphic symbol ON vs OFF?
      - Gate 2 (Text-Only Rule Knowledge): Without any image, does the VLM know the anime meaning?
      - Gate 3 (Conflict Binding — Direct): On the counterfactual image (smiling mouth + non-bio symbol),
        does predicted emotion shift from Cheerful/Happy -> Ominous/Hollow/Embarrassed/Anxious,
        or does the biological smiling mouth override the non-biological cue?
      - Gate 3-CoT (Observe-then-Bind Intervention): Explicitly instruct the VLM to FIRST list
        all graphic/non-biological anime cues (eye highlights, forehead shadow, blush lines, sweat drop),
        and ONLY THEN decide the character's true emotional state! Tests whether inference-time
        decomposition fixes Biological-Override Binding Failure.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _parse_json_obj(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"No JSON object found in: {text!r}")
    return json.loads(m.group(0))


def _detect_face_boxes_vlm(portrait_paths: list[Path], vlm_path: str) -> dict[str, dict[str, list[int]]]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        vlm_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    model.eval()

    prompt = (
        "Locate the character's facial parts in normalized coordinates [x1, y1, x2, y2] where each integer is in 0..1000.\n"
        "Return ONLY valid JSON with keys: \"left_eye\" (viewer's left eye iris+pupil tightly), "
        "\"right_eye\" (viewer's right eye iris+pupil tightly), \"mouth\", \"face\".\n"
        "Example: {\"left_eye\": [330, 410, 445, 510], \"right_eye\": [555, 410, 670, 510], \"mouth\": [455, 620, 545, 680], \"face\": [260, 220, 740, 770]}"
    )

    boxes_map: dict[str, dict[str, list[int]]] = {}
    for p in portrait_paths:
        img = Image.open(p).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[txt], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        raw = processor.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        try:
            b = _parse_json_obj(raw)
        except Exception as e:
            print(f"[WARN] Box parse failed for {p.name}: {raw!r} ({e}); using portrait defaults")
            b = {
                "left_eye": [330, 420, 455, 535],
                "right_eye": [545, 420, 670, 535],
                "mouth": [445, 610, 555, 685],
                "face": [250, 200, 750, 780],
            }
        boxes_map[p.stem] = b
        print(f"Detected boxes for {p.stem}: {b}", flush=True)
    return boxes_map


def _to_px(box_1000: list[int], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, int(round(box_1000[0] * w / 1000.0))))
    y1 = max(0, min(h - 1, int(round(box_1000[1] * h / 1000.0))))
    x2 = max(x1 + 4, min(w, int(round(box_1000[2] * w / 1000.0))))
    y2 = max(y1 + 4, min(h, int(round(box_1000[3] * h / 1000.0))))
    return x1, y1, x2, y2


def _remove_eye_highlights(img: Image.Image, boxes: dict[str, list[int]]) -> tuple[Image.Image, np.ndarray]:
    """
    Remove ONLY specular white catchlights strictly inside the colored iris of both eyes,
    by first localizing the saturated colored iris pixels (zero sclera leakage!) and
    replacing bright specular pixels inside the iris hull with matte darker iris tone.
    Leaves sclera, eyelashes, skin, hair, and mouth 100% pixel-untouched.
    """
    arr = np.asarray(img, dtype=np.float32).copy()
    h, w = arr.shape[:2]
    edit_mask = np.zeros((h, w), dtype=bool)

    for key in ["left_eye", "right_eye"]:
        x1, y1, x2, y2 = _to_px(boxes[key], w, h)
        bw, bh = x2 - x1, y2 - y1
        patch = arr[y1:y2, x1:x2]
        cmax = patch.max(axis=2) / 255.0
        cmin = patch.min(axis=2) / 255.0
        sat = np.where(cmax > 1e-5, (cmax - cmin) / np.maximum(cmax, 1e-5), 0.0)
        val = cmax

        # Step 1: Locate saturated colored iris pixels inside eye box (excludes white sclera & black lashes)
        colored_iris = (sat > 0.24) & (val > 0.14) & (val < 0.96)
        if colored_iris.sum() >= 30:
            ys_c, xs_c = np.where(colored_iris)
            # Robust centroid & tight iris radii from quantiles of colored iris pixels
            icx = float(np.median(xs_c))
            icy = float(np.median(ys_c))
            irx = max(6.0, float(np.percentile(xs_c, 96) - np.percentile(xs_c, 4)) * 0.54)
            iry = max(6.0, float(np.percentile(ys_c, 96) - np.percentile(ys_c, 4)) * 0.56)
        else:
            icx, icy = bw / 2.0, bh / 2.0
            irx, iry = bw * 0.34, bh * 0.40

        yy, xx = np.ogrid[0:bh, 0:bw]
        iris_ell = ((xx - icx) / irx) ** 2 + ((yy - icy) / iry) ** 2 <= 1.0

        # Step 2: Specular catchlights inside the iris ellipse (bright low-sat OR ultra-bright)
        hl_mask = iris_ell & (((val > 0.76) & (sat < 0.42)) | (val > 0.91))
        # Dilate by 2px inside iris_ell to eliminate white anti-aliased catchlight halos
        hl_img = Image.fromarray((hl_mask.astype(np.uint8) * 255), mode="L").filter(ImageFilter.MaxFilter(5))
        hl_mask = (np.asarray(hl_img) > 127) & iris_ell

        # Step 3: Estimate rich matte iris base color from non-highlight colored iris pixels
        iris_body = iris_ell & (~hl_mask) & (val > 0.15) & (val < 0.85)
        if (iris_body & colored_iris).sum() >= 20:
            fill_rgb = np.percentile(patch[iris_body & colored_iris], 30, axis=0)
        elif iris_body.sum() >= 20:
            fill_rgb = np.percentile(patch[iris_body], 30, axis=0)
        else:
            fill_rgb = np.array([58.0, 54.0, 72.0], dtype=np.float32)

        # Slightly desaturate & darken matte dead-eye tone (classic anime ハイライトオフ look)
        gray_val = float(fill_rgb.mean())
        dead_rgb = fill_rgb * 0.68 + gray_val * 0.12

        if hl_mask.sum() > 0:
            alpha = (
                np.asarray(
                    Image.fromarray((hl_mask.astype(np.uint8) * 255), mode="L").filter(ImageFilter.GaussianBlur(radius=1.8)),
                    dtype=np.float32,
                )
                / 255.0
            )
            for c in range(3):
                patch[:, :, c] = patch[:, :, c] * (1.0 - alpha) + dead_rgb[c] * alpha

            # Gently flatten remaining iris gloss inside iris_body
            matte_alpha = (
                np.asarray(
                    Image.fromarray((iris_body.astype(np.uint8) * 255), mode="L").filter(ImageFilter.GaussianBlur(radius=2.0)),
                    dtype=np.float32,
                )
                / 255.0
            ) * 0.34
            for c in range(3):
                patch[:, :, c] = patch[:, :, c] * (1.0 - matte_alpha) + dead_rgb[c] * matte_alpha

            arr[y1:y2, x1:x2] = patch
            edit_mask[y1:y2, x1:x2] |= hl_mask | iris_body

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB"), edit_mask


def _add_forehead_shadow(img: Image.Image, boxes: dict[str, list[int]]) -> tuple[Image.Image, np.ndarray]:
    """
    Add classic anime ominous forehead/upper-eye dread shadow (ガーン / 影) with
    fine vertical dark hatching lines across the upper face, stopping strictly ABOVE
    the nose tip and leaving the smiling mouth 100% pixel-identical!
    """
    w, h = img.size
    lx1, ly1, lx2, ly2 = _to_px(boxes["left_eye"], w, h)
    rx1, ry1, rx2, ry2 = _to_px(boxes["right_eye"], w, h)
    fx1, fy1, fx2, fy2 = _to_px(boxes["face"], w, h)
    mx1, my1, mx2, my2 = _to_px(boxes["mouth"], w, h)

    eye_bot = max(ly2, ry2)
    eye_top = min(ly1, ry1)
    shadow_top = max(0, int(fy1 - 0.04 * h))
    # Stop shadow cleanly right below the eyes (strictly well above the mouth!)
    shadow_bot = min(my1 - int(0.05 * h), int(eye_bot + 0.35 * (eye_bot - eye_top)))
    if shadow_bot <= shadow_top + 10:
        shadow_bot = eye_bot + 10

    arr = np.asarray(img, dtype=np.float32).copy()
    yy, xx = np.mgrid[0:h, 0:w]

    # Horizontal ellipse/face bounds so shadow stays on the character's head/face (not background!)
    fcx = (fx1 + fx2) / 2.0
    frx = max(1.0, (fx2 - fx1) * 0.56)
    horiz = np.clip(1.0 - ((xx - fcx) / frx) ** 4, 0.0, 1.0)

    # Vertical profile: strong dark indigo dread shadow over forehead & eyes, smoothly fading out by shadow_bot
    v_norm = (yy - shadow_top) / max(1.0, float(shadow_bot - shadow_top))
    vert = np.where((v_norm >= 0.0) & (v_norm <= 1.0), np.clip(1.0 - (v_norm ** 2.2), 0.0, 1.0), 0.0)
    # Soft top fade-in so no sharp horizontal bar at top of hair
    top_fade = np.clip((yy - shadow_top) / max(1.0, 0.06 * h), 0.0, 1.0)
    mask = horiz * vert * top_fade

    # Classic anime vertical dread lines ('||||') inside the upper shadow zone
    stripe = (np.sin(xx * (2.0 * math.pi / 9.0)) > 0.72).astype(np.float32)
    alpha = np.clip(mask * 0.64 + mask * stripe * 0.20, 0.0, 0.84)

    gloom_tint = np.array([26.0, 22.0, 48.0], dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1.0 - alpha) + gloom_tint[c] * alpha

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB"), (alpha > 0.02)


def _add_blush_lines(img: Image.Image, boxes: dict[str, list[int]]) -> tuple[Image.Image, np.ndarray]:
    """Add canonical anime shy blush (soft pink cheek oval + crisp diagonal '///' lines) below both eyes."""
    w, h = img.size
    lx1, ly1, lx2, ly2 = _to_px(boxes["left_eye"], w, h)
    rx1, ry1, rx2, ry2 = _to_px(boxes["right_eye"], w, h)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    cheek_boxes = []
    for (ex1, ey1, ex2, ey2) in [(lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2)]:
        ew = ex2 - ex1
        eh = ey2 - ey1
        cx = (ex1 + ex2) * 0.5
        cy = ey2 + eh * 0.36
        rx = ew * 0.52
        ry = eh * 0.26
        cheek_boxes.append((cx, cy, rx, ry))
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(255, 118, 142, 110))

    # Soften the pink cheek glow
    soft_blush = overlay.filter(ImageFilter.GaussianBlur(radius=14))

    # Draw crisp diagonal anime blush strokes ('///') on top
    strokes = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ds = ImageDraw.Draw(strokes)
    for (cx, cy, rx, ry) in cheek_boxes:
        for k in (-1.0, -0.33, 0.33, 1.0):
            sx = cx + k * rx * 0.55
            ds.line(
                [(sx - rx * 0.14, cy + ry * 0.65), (sx + rx * 0.14, cy - ry * 0.65)],
                fill=(228, 72, 102, 205),
                width=max(2, int(round(w * 0.0045))),
            )

    out = Image.alpha_composite(img.convert("RGBA"), soft_blush)
    out = Image.alpha_composite(out, strokes).convert("RGB")
    diff = np.abs(np.asarray(out, dtype=np.float32) - np.asarray(img, dtype=np.float32)).mean(axis=2) > 1.0
    return out, diff


def _add_sweat_drop(img: Image.Image, boxes: dict[str, list[int]]) -> tuple[Image.Image, np.ndarray]:
    """Composite a crisp cel-shaded anime temple sweat drop (cyan-white gradient + dark ink outline)."""
    w, h = img.size
    rx1, ry1, rx2, ry2 = _to_px(boxes["right_eye"], w, h)
    ew = rx2 - rx1
    # Place cleanly at the right temple (above/outer of viewer's right eye)
    cx = min(w - int(ew * 0.55), int(rx2 + ew * 0.25))
    cy = max(int(ew * 0.6), int(ry1 - ew * 0.22))
    r = max(14, int(ew * 0.28))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # Teardrop polygon (pointed top, rounded bottom)
    pts = []
    n = 48
    for i in range(n):
        t = 2.0 * math.pi * i / n
        # Parametric teardrop
        x = cx + r * 0.72 * math.sin(t) * (0.68 - 0.32 * math.cos(t))
        y = cy - r * 1.15 * math.cos(t)
        pts.append((x, y))
    d.polygon(pts, fill=(175, 232, 255, 248), outline=(38, 56, 88, 255), width=max(3, int(r * 0.12)))
    # Crisp white anime highlight inside bottom-left of drop
    d.ellipse([cx - r * 0.36, cy + r * 0.15, cx - r * 0.08, cy + r * 0.62], fill=(255, 255, 255, 245))

    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    diff = np.abs(np.asarray(out, dtype=np.float32) - np.asarray(img, dtype=np.float32)).mean(axis=2) > 1.0
    return out, diff


def _add_placebo_hairpin(img: Image.Image, boxes: dict[str, list[int]]) -> tuple[Image.Image, np.ndarray]:
    """Equal-pixel neutral geometric hairclip on upper-left hair (non-affect placebo control)."""
    w, h = img.size
    lx1, ly1, lx2, ly2 = _to_px(boxes["left_eye"], w, h)
    ew = lx2 - lx1
    cx = max(int(ew * 0.6), int(lx1 - ew * 0.25))
    cy = max(int(ew * 0.6), int(ly1 - ew * 0.48))
    pw = max(24, int(ew * 0.45))
    ph = max(9, int(ew * 0.14))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle([cx - pw // 2, cy - ph // 2, cx + pw // 2, cy + ph // 2], radius=ph // 2, fill=(235, 212, 112, 255), outline=(45, 38, 48, 255), width=3)
    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    diff = np.abs(np.asarray(out, dtype=np.float32) - np.asarray(img, dtype=np.float32)).mean(axis=2) > 1.0
    return out, diff


def build_counterfactual_pairs(portraits_dir: Path, out_dir: Path, locator_vlm: str) -> list[dict[str, Any]]:
    portrait_paths = sorted(portraits_dir.glob("*.png"))
    if not portrait_paths:
        raise FileNotFoundError(f"No *.png portraits in {portraits_dir}")

    boxes_path = out_dir / "face_boxes.json"
    if boxes_path.exists():
        boxes_map = json.loads(boxes_path.read_text(encoding="utf-8"))
    else:
        boxes_map = _detect_face_boxes_vlm(portrait_paths, locator_vlm)
        boxes_path.write_text(json.dumps(boxes_map, indent=2), encoding="utf-8")

    img_dir = out_dir / "images"
    sheet_dir = out_dir / "sheets"
    img_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for p in portrait_paths:
        cid = p.stem
        orig = Image.open(p).convert("RGB")
        b = boxes_map[cid]

        img_hl_off, m_hl = _remove_eye_highlights(orig, b)
        img_shadow, m_sh = _add_forehead_shadow(orig, b)
        img_dead_sh, _ = _add_forehead_shadow(img_hl_off, b)
        img_blush, m_bl = _add_blush_lines(orig, b)
        img_sweat, m_sw = _add_sweat_drop(orig, b)
        img_placebo, m_pl = _add_placebo_hairpin(orig, b)

        variants = [
            ("orig", orig, "cheerful"),
            ("eye_highlight_off", img_hl_off, "hollow_or_ominous"),
            ("forehead_shadow_on", img_shadow, "hollow_or_ominous"),
            ("dead_eyes_and_shadow", img_dead_sh, "hollow_or_ominous"),
            ("blush_lines_on", img_blush, "shy_or_embarrassed"),
            ("sweat_drop_on", img_sweat, "anxious_or_awkward"),
            ("placebo_hairpin", img_placebo, "cheerful"),
        ]

        # Verify mouth pixel-exact invariance across all variants
        w, h = orig.size
        mx1, my1, mx2, my2 = _to_px(b["mouth"], w, h)
        orig_mouth = np.asarray(orig)[my1:my2, mx1:mx2].astype(np.float32)

        thumbs = []
        for vname, vimg, expected_affect in variants:
            vpath = img_dir / f"{cid}__{vname}.png"
            vimg.save(vpath)
            mouth_l1 = float(np.abs(np.asarray(vimg)[my1:my2, mx1:mx2].astype(np.float32) - orig_mouth).mean())
            changed_frac = float((np.abs(np.asarray(vimg, dtype=np.float32) - np.asarray(orig, dtype=np.float32)).mean(axis=2) > 2.0).mean())
            records.append(
                {
                    "char_id": cid,
                    "variant": vname,
                    "expected_affect": expected_affect,
                    "image_path": str(vpath),
                    "mouth_l1_diff": round(mouth_l1, 5),
                    "changed_pixel_frac": round(changed_frac, 5),
                }
            )
            t = vimg.resize((256, 256), Image.LANCZOS)
            ImageDraw.Draw(t).rectangle([0, 0, 256, 24], fill=(20, 20, 28))
            ImageDraw.Draw(t).text((6, 5), f"{vname} ({changed_frac*100:.1f}%)", fill=(255, 255, 255))
            thumbs.append(t)

        sheet = Image.new("RGB", (256 * len(thumbs), 256))
        for i, t in enumerate(thumbs):
            sheet.paste(t, (i * 256, 0))
        sheet.save(sheet_dir / f"{cid}_sheet.jpg", quality=92)

    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"Built {len(records)} counterfactual anime portraits -> {manifest}")
    return records


def eval_4gate_vlm(records: list[dict[str, Any]], vlm_path: str, out_dir: Path) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        vlm_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    model.eval()

    def ask(img: Image.Image | None, prompt: str, max_new_tokens: int = 128) -> str:
        content: list[dict[str, Any]] = []
        if img is not None:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if img is not None:
            inputs = processor(text=[txt], images=[img], return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=[txt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    # --- GATE 2: Text-only semantic knowledge (no image) ---
    gate2_qs = {
        "eye_highlight_off": (
            "In 2D anime visual language, if a character has a smiling mouth BUT their white eye highlights/catchlights "
            "are completely removed ('dead eyes' / matte dark irises), what emotional state does this typically signify?\n"
            "Choose ONE letter:\nA) Genuinely cheerful and carefree\nB) Emotionally hollow, shocked, sinister, or ominous (yandere)\n"
            "C) Shy or romantically embarrassed\nD) Awkwardly nervous or sweating\nAnswer with ONLY A, B, C, or D."
        ),
        "forehead_shadow_on": (
            "In 2D anime visual language, if a character has a smiling mouth BUT a heavy dark vertical shadow with gloom lines "
            "covers their forehead and eyes, what emotional state does this typically signify?\n"
            "Choose ONE letter:\nA) Genuinely cheerful and carefree\nB) Ominous, menacing, grim, or suppressing dark intent\n"
            "C) Shy or romantically embarrassed\nD) Awkwardly nervous\nAnswer with ONLY A, B, C, or D."
        ),
        "blush_lines_on": (
            "In 2D anime visual language, if a character has diagonal pink-red hatch lines ('///') and soft pink blush under both eyes, "
            "what emotional state does this typically signify?\n"
            "Choose ONE letter:\nA) Genuinely neutral/cheerful without shyness\nB) Ominous or menacing\n"
            "C) Shy, flustered, or embarrassed\nD) Angry\nAnswer with ONLY A, B, C, or D."
        ),
        "sweat_drop_on": (
            "In 2D anime visual language, if a character has a stylized cyan-white cartoon sweat drop floating beside their temple, "
            "what emotional state does this typically signify?\n"
            "Choose ONE letter:\nA) Genuinely carefree and confident\nB) Ominous or sinister\n"
            "C) Shy or romantically in love\nD) Awkward, flustered, or nervously uneasy\nAnswer with ONLY A, B, C, or D."
        ),
    }
    gate2_gt = {"eye_highlight_off": "B", "forehead_shadow_on": "B", "blush_lines_on": "C", "sweat_drop_on": "D"}
    gate2_res = {}
    for k, q in gate2_qs.items():
        ans = ask(None, q, max_new_tokens=16)
        letter = re.search(r"\b([ABCD])\b", ans.upper())
        pred = letter.group(1) if letter else ans[:2]
        gate2_res[k] = {"raw": ans, "pred": pred, "correct": pred == gate2_gt[k]}

    # --- GATE 1 (Perception) + GATE 3 (Direct Affect) + GATE 3-CoT (Observe-then-Bind) ---
    g1_prompts = {
        "eye_highlight": "Look closely inside the character's colored irises. Are bright white specular eye highlights/catchlights clearly visible inside the eyes? Answer ONLY YES or NO.",
        "forehead_shadow": "Is there a heavy dark ominous shadow cast across the character's forehead and upper eyes? Answer ONLY YES or NO.",
        "blush_lines": "Are there stylized anime pink-red diagonal blush lines ('///') on the character's cheeks below the eyes? Answer ONLY YES or NO.",
        "sweat_drop": "Is there a stylized cartoon anime sweat drop floating near the character's temple/head? Answer ONLY YES or NO.",
    }
    g3_direct_prompt = (
        "Look at this 2D anime character portrait and judge their TRUE emotional / psychological state.\n"
        "Choose the SINGLE best label:\n"
        "A) Genuinely cheerful, bright, and happily carefree\n"
        "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
        "C) Shy, flustered, or bashfully embarrassed\n"
        "D) Awkward, uneasy, or nervously sweating\n"
        "Answer with ONLY a single letter: A, B, C, or D."
    )
    g3_cot_prompt = (
        "Analyze this 2D anime portrait in TWO strict steps:\n"
        "Step 1 (Non-Biological Graphic Cues Audit): Explicitly check four anime visual cues — "
        "(1) Are white eye highlights present or removed ('dead eyes')? "
        "(2) Is there a dark ominous shadow over the forehead/eyes? "
        "(3) Are there anime cheek blush lines? "
        "(4) Is there a temple sweat drop?\n"
        "Step 2 (Anime Affect Binding): In anime visual language, non-biological graphic cues OVERRIDE a smiling mouth.\n"
        "Based on Step 1, output the FINAL single letter on the last line (`FINAL: A`, `FINAL: B`, `FINAL: C`, or `FINAL: D`):\n"
        "A) Genuinely cheerful, bright, and happily carefree\n"
        "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
        "C) Shy, flustered, or bashfully embarrassed\n"
        "D) Awkward, uneasy, or nervously sweating"
    )
    expected_letter = {
        "orig": "A",
        "placebo_hairpin": "A",
        "eye_highlight_off": "B",
        "forehead_shadow_on": "B",
        "dead_eyes_and_shadow": "B",
        "blush_lines_on": "C",
        "sweat_drop_on": "D",
    }

    rows = []
    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        v = rec["variant"]

        # Gate 1 perception check relevant to this variant (and orig)
        g1 = {}
        if v in {"orig", "eye_highlight_off", "dead_eyes_and_shadow"}:
            r = ask(img, g1_prompts["eye_highlight"], max_new_tokens=8)
            g1["eye_highlight_present"] = r.upper().startswith("YES")
        if v in {"orig", "forehead_shadow_on", "dead_eyes_and_shadow"}:
            r = ask(img, g1_prompts["forehead_shadow"], max_new_tokens=8)
            g1["forehead_shadow_present"] = r.upper().startswith("YES")
        if v in {"orig", "blush_lines_on"}:
            r = ask(img, g1_prompts["blush_lines"], max_new_tokens=8)
            g1["blush_lines_present"] = r.upper().startswith("YES")
        if v in {"orig", "sweat_drop_on"}:
            r = ask(img, g1_prompts["sweat_drop"], max_new_tokens=8)
            g1["sweat_drop_present"] = r.upper().startswith("YES")

        # Gate 3 direct
        raw_d = ask(img, g3_direct_prompt, max_new_tokens=16)
        md = re.search(r"\b([ABCD])\b", raw_d.upper())
        pred_d = md.group(1) if md else "UNK"

        # Gate 3 Observe-then-Bind CoT intervention
        raw_c = ask(img, g3_cot_prompt, max_new_tokens=220)
        mc = re.search(r"FINAL\s*:\s*([ABCD])", raw_c.upper())
        if not mc:
            ms = re.findall(r"\b([ABCD])\b", raw_c.upper())
            pred_c = ms[-1] if ms else "UNK"
        else:
            pred_c = mc.group(1)

        gt = expected_letter[v]
        rows.append(
            {
                "char_id": rec["char_id"],
                "variant": v,
                "gt": gt,
                "gate1": g1,
                "direct_raw": raw_d,
                "direct_pred": pred_d,
                "direct_correct": pred_d == gt,
                "cot_raw": raw_c,
                "cot_pred": pred_c,
                "cot_correct": pred_c == gt,
            }
        )

    # Aggregate summary
    by_var: dict[str, Any] = {}
    for v in ["orig", "placebo_hairpin", "eye_highlight_off", "forehead_shadow_on", "dead_eyes_and_shadow", "blush_lines_on", "sweat_drop_on"]:
        sub = [r for r in rows if r["variant"] == v]
        if not sub:
            continue
        # Perception accuracy for this variant
        g1_acc = None
        if v == "orig":
            g1_acc = {
                "eye_hl_yes": round(np.mean([r["gate1"]["eye_highlight_present"] for r in sub]), 4),
                "shadow_no": round(np.mean([not r["gate1"]["forehead_shadow_present"] for r in sub]), 4),
                "blush_no": round(np.mean([not r["gate1"]["blush_lines_present"] for r in sub]), 4),
                "sweat_no": round(np.mean([not r["gate1"]["sweat_drop_present"] for r in sub]), 4),
            }
        elif v == "eye_highlight_off":
            g1_acc = round(float(np.mean([not r["gate1"]["eye_highlight_present"] for r in sub])), 4)
        elif v == "forehead_shadow_on":
            g1_acc = round(float(np.mean([r["gate1"]["forehead_shadow_present"] for r in sub])), 4)
        elif v == "dead_eyes_and_shadow":
            g1_acc = round(
                float(np.mean([(not r["gate1"]["eye_highlight_present"]) and r["gate1"]["forehead_shadow_present"] for r in sub])),
                4,
            )
        elif v == "blush_lines_on":
            g1_acc = round(float(np.mean([r["gate1"]["blush_lines_present"] for r in sub])), 4)
        elif v == "sweat_drop_on":
            g1_acc = round(float(np.mean([r["gate1"]["sweat_drop_present"] for r in sub])), 4)

        by_var[v] = {
            "n": len(sub),
            "gt": expected_letter[v],
            "gate1_perception_acc": g1_acc,
            "gate3_direct_acc": round(float(np.mean([r["direct_correct"] for r in sub])), 4),
            "gate3_direct_bio_override_A_rate": round(float(np.mean([r["direct_pred"] == "A" for r in sub])), 4),
            "gate3_cot_acc": round(float(np.mean([r["cot_correct"] for r in sub])), 4),
            "gate3_cot_bio_override_A_rate": round(float(np.mean([r["cot_pred"] == "A" for r in sub])), 4),
        }

    summary = {"model_path": vlm_path, "gate2_text_knowledge": gate2_res, "variants": by_var}
    safe = Path(vlm_path).name
    out_path = out_dir / f"p3_4gate_{safe}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved P3 4-Gate results -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portraits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--locator-vlm", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--build-pairs", action="store_true")
    ap.add_argument("--eval-vlm", type=str, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.jsonl"
    if args.build_pairs or not manifest.exists():
        records = build_counterfactual_pairs(args.portraits_dir, args.out_dir, args.locator_vlm)
    else:
        records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.eval_vlm:
        eval_4gate_vlm(records, args.eval_vlm, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
