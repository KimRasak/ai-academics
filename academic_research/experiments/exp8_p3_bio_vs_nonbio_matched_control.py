#!/usr/bin/env python3
"""
Exp 8 (P3 Gold-Standard Equal-Pixel Biological vs. Non-Biological Control + Atomic Zero-Training Gate):

Addresses the #1 methodological control in affect/VLM research:
  Same character, SAME smiling mouth (mouth_l1=0.0), SAME pixel budget (~0.3% px),
  SAME ground-truth emotion (`E: Irritated / Suppressing Anger`):
    1. `bio_angry_brows` (Biological FACS AU4: crisp dark V-furrowed anime brows `\ /`)
    2. `anger_vein_on`   (Pure Non-Biological Anime Symbol: classic red temple anger mark `💢`)

  Plus evaluates ALL 9 variants (adding `bio_angry_brows` & `anger_vein_on` to our 7 v2 variants)
  across 5 options {A: Cheerful, B: Hollow/Ominous, C: Shy/Embarrassed, D: Awkward/Nervous, E: Irritated/Angry}
  comparing:
    - Direct 1-token
    - Audit 1-token
    - Zero-Training Atomic Graphic-Cue Gate + Eye-Zoom Binder (ZERO training labels, ZERO weight updates!)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

LABELS5 = ["A", "B", "C", "D", "E"]
EXPECTED9 = {
    "orig": "A",
    "placebo_hairpin": "A",
    "eye_highlight_off": "B",
    "forehead_shadow_on": "B",
    "dead_eyes_and_shadow": "B",
    "blush_lines_on": "C",
    "sweat_drop_on": "D",
    "bio_angry_brows": "E",
    "anger_vein_on": "E",
}

DIRECT5_PROMPT = (
    "Look at this 2D anime character portrait and judge their TRUE emotional / psychological state.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "E) Irritated, annoyed, or suppressing anger behind a smile\n"
    "Answer with ONLY a single letter: A, B, C, D, or E."
)

AUDIT5_PROMPT = (
    "Inspect this 2D anime portrait carefully for ANY facial or non-biological anime graphic affect cues "
    "(such as furrowed angry brows, a red anime cross-vein anger mark, unusual dark forehead shadow, "
    "missing/hollow eye highlights, cheek blush lines, or a floating sweat drop), ignoring ordinary hairpins.\n"
    "In 2D anime, graphic affect symbols and angry brows OVERRIDE a smiling mouth.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "E) Irritated, annoyed, or suppressing anger behind a smile\n"
    "Answer with ONLY a single letter: A, B, C, D, or E."
)


def _to_px(box_1000: list[int], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, int(round(box_1000[0] * w / 1000.0))))
    y1 = max(0, min(h - 1, int(round(box_1000[1] * h / 1000.0))))
    x2 = max(x1 + 4, min(w, int(round(box_1000[2] * w / 1000.0))))
    y2 = max(y1 + 4, min(h, int(round(box_1000[3] * h / 1000.0))))
    return x1, y1, x2, y2


def _add_bio_angry_brows(img: Image.Image, boxes: dict[str, list[int]]) -> Image.Image:
    """Draw crisp anime V-furrowed angry eyebrows (FACS AU4) + brow crease strictly above the eyes (~0.3% px)."""
    w, h = img.size
    lx1, ly1, lx2, ly2 = _to_px(boxes["left_eye"], w, h)
    rx1, ry1, rx2, ry2 = _to_px(boxes["right_eye"], w, h)
    ew = int(0.5 * ((lx2 - lx1) + (rx2 - rx1)))
    stroke_w = max(5, int(round(ew * 0.085)))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # Viewer's left brow: high outer-left -> sharply angled DOWN toward nose bridge
    l_out = (lx1 - int(ew * 0.06), ly1 - int(ew * 0.24))
    l_in = (lx2 + int(ew * 0.04), ly1 + int(ew * 0.05))
    d.line([l_out, l_in], fill=(32, 24, 34, 245), width=stroke_w)

    # Viewer's right brow: high outer-right -> sharply angled DOWN toward nose bridge
    r_out = (rx2 + int(ew * 0.06), ry1 - int(ew * 0.24))
    r_in = (rx1 - int(ew * 0.04), ry1 + int(ew * 0.05))
    d.line([r_out, r_in], fill=(32, 24, 34, 245), width=stroke_w)

    # Small central vertical glabella furrow lines (classic FACS AU4 tension between brows)
    mid_x = int(0.5 * (l_in[0] + r_in[0]))
    mid_y = int(0.5 * (l_in[1] + r_in[1]))
    fw = max(2, stroke_w // 2)
    d.line([(mid_x - int(ew * 0.07), mid_y - int(ew * 0.14)), (mid_x - int(ew * 0.05), mid_y + int(ew * 0.04))], fill=(72, 46, 54, 220), width=fw)
    d.line([(mid_x + int(ew * 0.07), mid_y - int(ew * 0.14)), (mid_x + int(ew * 0.05), mid_y + int(ew * 0.04))], fill=(72, 46, 54, 220), width=fw)

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _add_anger_vein_mark(img: Image.Image, boxes: dict[str, list[int]]) -> Image.Image:
    """Composite a crisp canonical anime red cross-popping anger vein mark ('💢') at temple (~0.3% px)."""
    w, h = img.size
    rx1, ry1, rx2, ry2 = _to_px(boxes["right_eye"], w, h)
    ew = rx2 - rx1
    cx = min(w - int(ew * 0.5), int(rx2 + ew * 0.18))
    cy = max(int(ew * 0.55), int(ry1 - ew * 0.30))
    r = max(20, int(ew * 0.30))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    # 4 curved brackets facing outward from center (classic anime 💢 symbol!)
    sw = max(5, int(r * 0.22))
    gap = int(r * 0.26)
    arm = int(r * 0.85)
    # Draw white outline backing for crisp anime readability against any hair color, then bright crimson stroke
    for col, width in [((255, 255, 255, 240), sw + 6), ((225, 34, 48, 255), sw)]:
        for sx in (-1, 1):
            for sy in (-1, 1):
                p1 = (cx + sx * gap, cy + sy * (gap + arm))
                p2 = (cx + sx * gap, cy + sy * gap)
                p3 = (cx + sx * (gap + arm), cy + sy * gap)
                d.line([p1, p2, p3], fill=col, width=width, joint="curve")

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def build_added_variants(v2_dir: Path, portraits_dir: Path) -> list[dict[str, Any]]:
    boxes_map = json.loads((v2_dir / "face_boxes.json").read_text(encoding="utf-8"))
    base_records = [json.loads(l) for l in (v2_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # Keep only the original 7 variants before appending the 2 new ones (idempotent!)
    records = [r for r in base_records if r["variant"] not in {"bio_angry_brows", "anger_vein_on"}]

    img_dir = v2_dir / "images"
    sheet_dir = v2_dir / "sheets_bio_vs_nonbio"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(portraits_dir.glob("*.png")):
        cid = p.stem
        orig = Image.open(p).convert("RGB")
        b = boxes_map[cid]
        w, h = orig.size
        mx1, my1, mx2, my2 = _to_px(b["mouth"], w, h)
        orig_mouth = np.asarray(orig)[my1:my2, mx1:mx2].astype(np.float32)

        img_brows = _add_bio_angry_brows(orig, b)
        img_vein = _add_anger_vein_mark(orig, b)

        for vname, vimg in [("bio_angry_brows", img_brows), ("anger_vein_on", img_vein)]:
            vpath = img_dir / f"{cid}__{vname}.png"
            vimg.save(vpath)
            mouth_l1 = float(np.abs(np.asarray(vimg)[my1:my2, mx1:mx2].astype(np.float32) - orig_mouth).mean())
            chg = float((np.abs(np.asarray(vimg, dtype=np.float32) - np.asarray(orig, dtype=np.float32)).mean(axis=2) > 2.0).mean())
            records.append(
                {
                    "char_id": cid,
                    "variant": vname,
                    "expected_affect": "irritated_or_angry",
                    "image_path": str(vpath),
                    "mouth_l1_diff": round(mouth_l1, 5),
                    "changed_pixel_frac": round(chg, 5),
                }
            )

        # Save 3-panel QC strip: orig | bio_angry_brows | anger_vein_on
        panels = [("orig", orig), ("bio_angry_brows (FACS AU4)", img_brows), ("anger_vein_on (Non-Bio)", img_vein)]
        sheet = Image.new("RGB", (320 * 3, 320))
        for i, (lab, im) in enumerate(panels):
            t = im.resize((320, 320), Image.LANCZOS)
            ImageDraw.Draw(t).rectangle([0, 0, 320, 26], fill=(20, 20, 28))
            ImageDraw.Draw(t).text((6, 6), lab, fill=(255, 255, 255))
            sheet.paste(t, (i * 320, 0))
        sheet.save(sheet_dir / f"{cid}_bio_vs_nonbio.jpg", quality=92)

    out_man = v2_dir / "manifest_9var.jsonl"
    out_man.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"Built 9-variant manifest ({len(records)} images) -> {out_man}", flush=True)
    return records


@torch.no_grad()
def eval_model_9var(records: list[dict[str, Any]], boxes_map: dict[str, Any], model_dir: str, out_dir: Path) -> None:
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    tok = processor.tokenizer
    abcde_ids = [tok.encode(ch, add_special_tokens=False)[-1] for ch in LABELS5]
    yes_id = tok.encode("YES", add_special_tokens=False)[-1]
    no_id = tok.encode("NO", add_special_tokens=False)[-1]

    def logits5(img: Image.Image, prompt: str) -> np.ndarray:
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[txt], images=[img], return_tensors="pt").to(model.device)
        return model(**inputs).logits[0, -1, abcde_ids].float().cpu().numpy()

    def prob_yes(img: Image.Image, q: str) -> float:
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[txt], images=[img], return_tensors="pt").to(model.device)
        lg = model(**inputs).logits[0, -1, [no_id, yes_id]].float().cpu().numpy()
        ex = np.exp(lg - np.max(lg))
        return float(ex[1] / ex.sum())

    # Atomic ZERO-TRAINING cues (open-vocab, zero weights trained!)
    atomic_qs = [
        ("dead_eyes", "Look closely at the character's colored irises in this zoomed eye crop. Are the eyes completely missing bright white catchlights (dark, hollow, or matte dead eyes)? Answer ONLY YES or NO.", True),
        ("forehead_shadow", "Is there a heavy dark ominous gloom shadow cast over the character's forehead and upper eyes? Answer ONLY YES or NO.", False),
        ("blush_lines", "Are there stylized pink-red diagonal anime blush lines ('///') on the character's cheeks? Answer ONLY YES or NO.", False),
        ("sweat_drop", "Is there a stylized cyan-white cartoon anime sweat drop floating beside the character's temple? Answer ONLY YES or NO.", False),
        ("angry_cue", "Are there sharply furrowed V-shaped angry eyebrows OR a red anime cross-vein anger mark on the character's forehead/temple? Answer ONLY YES or NO.", False),
    ]

    direct_preds, audit_preds, atomic_gated_preds = [], [], []
    gate_stats: dict[str, list[dict[str, float]]] = {}

    for rec in records:
        img = Image.open(rec["image_path"]).convert("RGB")
        w, h = img.size
        b = boxes_map[rec["char_id"]]
        # 2x zoomed eye-band crop for fine-grained specular eye highlight inspection!
        ex1 = max(0, int(round((min(b["left_eye"][0], b["right_eye"][0]) - 50) * w / 1000.0)))
        ex2 = min(w, int(round((max(b["left_eye"][2], b["right_eye"][2]) + 50) * w / 1000.0)))
        ey1 = max(0, int(round((min(b["left_eye"][1], b["right_eye"][1]) - 60) * h / 1000.0)))
        ey2 = min(h, int(round((max(b["left_eye"][3], b["right_eye"][3]) + 60) * h / 1000.0)))
        eye_crop = img.crop((ex1, ey1, max(ex1 + 64, ex2), max(ey1 + 64, ey2))).resize((512, 256), Image.LANCZOS)

        ld = logits5(img, DIRECT5_PROMPT)
        la = logits5(img, AUDIT5_PROMPT)
        direct_preds.append(LABELS5[int(np.argmax(ld))])
        audit_preds.append(LABELS5[int(np.argmax(la))])

        # Zero-training atomic cue verification (with eye-band zoom for dead_eyes!)
        cue_ps = {}
        for cname, q, use_eye_crop in atomic_qs:
            cue_ps[cname] = prob_yes(eye_crop if use_eye_crop else img, q)
        gate_stats.setdefault(rec["variant"], []).append(cue_ps)

        # Deterministic zero-training atomic OR-gate -> Stage-2 non-A binding!
        max_cue = max(cue_ps.values())
        if max_cue < 0.55:
            atomic_gated_preds.append("A")
        else:
            # Combine Direct + Audit conditional probabilities over {B, C, D, E}
            p1 = np.exp(ld[1:] - np.max(ld[1:])); p1 /= p1.sum()
            p2 = np.exp(la[1:] - np.max(la[1:])); p2 /= p2.sum()
            atomic_gated_preds.append(LABELS5[1 + int(np.argmax(p1 + p2))])

    def summarize(preds: list[str]) -> dict[str, Any]:
        by_v: dict[str, list[bool]] = {}
        for r, p in zip(records, preds):
            by_v.setdefault(r["variant"], []).append(p == EXPECTED9[r["variant"]])
        sv = {v: round(float(np.mean(xs)), 4) for v, xs in sorted(by_v.items())}
        clean = float(np.mean([sv["orig"], sv["placebo_hairpin"]]))
        nonbio = float(np.mean([sv["eye_highlight_off"], sv["forehead_shadow_on"], sv["dead_eyes_and_shadow"], sv["sweat_drop_on"], sv["anger_vein_on"]]))
        return {
            "macro_9": round(float(np.mean(list(sv.values()))), 4),
            "clean_acc": round(clean, 4),
            "pure_nonbio_5_acc": round(nonbio, 4),
            "bio_angry_brows": sv["bio_angry_brows"],
            "anger_vein_nonbio": sv["anger_vein_on"],
            "bio_minus_nonbio_gap": round(sv["bio_angry_brows"] - sv["anger_vein_on"], 4),
            "by_variant": sv,
        }

    mean_gate = {
        v: {c: round(float(np.mean([row[c] for row in rows])), 3) for c in rows[0]}
        for v, rows in sorted(gate_stats.items())
    }
    res = {
        "model": Path(model_dir).name,
        "direct_5opt": summarize(direct_preds),
        "audit_5opt": summarize(audit_preds),
        "zero_train_atomic_zoom_gate": summarize(atomic_gated_preds),
        "atomic_cue_mean_probs": mean_gate,
    }
    out_path = out_dir / f"exp8_9var_{Path(model_dir).name}.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"=== EXP 8 (9-VAR BIO vs NON-BIO CONTROL): {res['model']} ===")
    for k in ["direct_5opt", "audit_5opt", "zero_train_atomic_zoom_gate"]:
        s = res[k]
        print(
            f"  {k:28s} macro9={s['macro_9']:.3f} clean={s['clean_acc']:.3f} pure_nonbio={s['pure_nonbio_5_acc']:.3f} "
            f"| BIO_brows={s['bio_angry_brows']:.2f} vs NONBIO_vein={s['anger_vein_nonbio']:.2f} (gap={s['bio_minus_nonbio_gap']:+.2f})"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--portraits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--model-dir", type=str, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    man9 = args.v2_dir / "manifest_9var.jsonl"
    if args.build_only or not man9.exists():
        records = build_added_variants(args.v2_dir, args.portraits_dir)
    else:
        records = [json.loads(l) for l in man9.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.model_dir:
        boxes_map = json.loads((args.v2_dir / "face_boxes.json").read_text(encoding="utf-8"))
        eval_model_9var(records, boxes_map, args.model_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
