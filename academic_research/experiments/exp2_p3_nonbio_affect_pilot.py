#!/usr/bin/env python3
"""Exp 2 (P3 Pilot): Beyond-FACS Minimal-Pair Non-Biological Anime Affect Evaluator.

Evaluates VLMs (Qwen2.5-VL / Qwen3-VL / InternVL) through the 4-Gate Ladder on paired
minimal-pair anime faces (either real layered 2D PNGs or Blender VRM renders):
  - Gate 1 (Low-Level Perception): Can the VLM see the exact visual cue toggled (e.g., white eye highlights present)?
  - Gate 2 (Pure-Text Knowledge): Does the VLM know what the anime cue means in pure text?
  - Gate 3 (A -> B Affect Shift / Conflict Override): Given identical smiling face A vs B (only 1 non-bio symbol changed),
            does the VLM flip its psychological reading, or let the smiling mouth override it?
  - Gate 4 (Control Contrast): Compare against Equal-Pixel Biological FACS Control and Equal-Pixel Placebo Control.

Also evaluates Image Editors (Qwen-Image-Edit / FLUX-Kontext) on the 3-Tier Prompt Decomposition
with strictly objective pixel metrics:
  - Global Luminance Shift (delta_L outside face/eye mask -> measures 'turn off room lights' literalization)
  - Outside-Mask Pixel Leakage (L1 drift outside eye/target bbox -> measures unintended biologization/wrinkles)

Expected input manifest JSONL (one row per minimal pair):
  {
    "pair_id": "char01_highlight",
    "arm": "nonbio_symbol" | "bio_control" | "placebo_control",
    "symbol": "eye_highlight_off" | "forehead_shadow" | "blush_lines" | "brow_furrow" | "hairpin_toggle",
    "img_a": "/path/to/neutral_or_smiling_base.png",
    "img_b": "/path/to/edited_minimal_pair.png",
    "gate1_q": "Are bright white specular highlight dots visible inside both eyes? Answer Yes or No.",
    "gate1_gt_a": "Yes",
    "gate1_gt_b": "No",
    "gate2_text_q": "In 2D anime art, if a smiling character's eyes suddenly lose their white specular highlights, what psychological shift does this usually signal? (A) Calm happiness (B) Shock, emotional hollowness, or ominous darkness. Answer A or B.",
    "gate2_gt": "B",
    "gate3_transition_q": "Compare Image A (left) and Image B (right) of the same anime character. How did her psychological/emotional state shift from A to B? (A) Remained calmly happy (B) Shifted toward shock, hollowness, or ominous darkness. Answer A or B.",
    "gate3_gt": "B"
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def compute_objective_edit_leakage(
    img_src_path: str,
    img_edit_path: str,
    target_bbox_xyxy: tuple[int, int, int, int],
) -> dict[str, float]:
    """Objective metrics for Tier 1/2/3 generative biologization & literalization (zero VLM judge)."""
    src = np.asarray(Image.open(img_src_path).convert("RGB"), dtype=np.float32) / 255.0
    edt = np.asarray(Image.open(img_edit_path).convert("RGB").resize((src.shape[1], src.shape[0])), dtype=np.float32) / 255.0

    h, w, _ = src.shape
    x0, y0, x1, y1 = target_bbox_xyxy
    mask = np.zeros((h, w), dtype=bool)
    mask[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = True
    outside = ~mask

    # Rec.709 luminance
    lum_src = 0.2126 * src[..., 0] + 0.7152 * src[..., 1] + 0.0722 * src[..., 2]
    lum_edt = 0.2126 * edt[..., 0] + 0.7152 * edt[..., 1] + 0.0722 * edt[..., 2]

    inside_l1 = float(np.mean(np.abs(edt[mask] - src[mask]))) if mask.any() else 0.0
    outside_l1 = float(np.mean(np.abs(edt[outside] - src[outside]))) if outside.any() else 0.0
    global_delta_lum = float(np.mean(lum_edt[outside] - lum_src[outside])) if outside.any() else 0.0
    abs_delta_lum = float(np.mean(np.abs(lum_edt[outside] - lum_src[outside]))) if outside.any() else 0.0

    return {
        "inside_target_l1": round(inside_l1, 5),
        "outside_mask_l1_leakage": round(outside_l1, 5),
        "outside_signed_delta_lum": round(global_delta_lum, 5),
        "outside_abs_delta_lum": round(abs_delta_lum, 5),
        "leakage_to_signal_ratio": round(outside_l1 / max(1e-5, inside_l1), 4),
    }


def make_side_by_side(img_a: Image.Image, img_b: Image.Image, gap: int = 16) -> Image.Image:
    h = max(img_a.height, img_b.height)
    a = img_a.resize((int(round(img_a.width * h / img_a.height)), h))
    b = img_b.resize((int(round(img_b.width * h / img_b.height)), h))
    canvas = Image.new("RGB", (a.width + b.width + gap, h), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    return canvas


def eval_qwen_vl_4gate(manifest_path: Path, model_id: str, out_dir: Path) -> None:
    """Run the 4-Gate Ladder on Qwen2.5-VL / Qwen3-VL."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    def ask(images: list[Image.Image], prompt: str) -> str:
        content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images if images else None, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        gen = out_ids[0, inputs["input_ids"].shape[1] :]
        return processor.decode(gen, skip_special_tokens=True).strip()

    rows = [json.loads(ln) for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    results = []
    for row in rows:
        im_a = Image.open(row["img_a"]).convert("RGB")
        im_b = Image.open(row["img_b"]).convert("RGB")
        sbs = make_side_by_side(im_a, im_b)

        g1_a = ask([im_a], row["gate1_q"])
        g1_b = ask([im_b], row["gate1_q"])
        g1_pass = g1_a.lower().startswith(row["gate1_gt_a"].lower()) and g1_b.lower().startswith(row["gate1_gt_b"].lower())

        g2 = ask([], row["gate2_text_q"])
        g2_pass = g2.upper().startswith(row["gate2_gt"].upper())

        g3 = ask([sbs], row["gate3_transition_q"])
        g3_pass = g3.upper().startswith(row["gate3_gt"].upper())

        results.append(
            {
                "pair_id": row["pair_id"],
                "arm": row["arm"],
                "symbol": row["symbol"],
                "gate1_a": g1_a,
                "gate1_b": g1_b,
                "gate1_pass": g1_pass,
                "gate2_ans": g2,
                "gate2_pass": g2_pass,
                "gate3_ans": g3,
                "gate3_pass": g3_pass,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"model_id": model_id, "by_arm": {}, "rows": results}
    for arm in sorted({r["arm"] for r in results}):
        sub = [r for r in results if r["arm"] == arm]
        g1_passed = [r for r in sub if r["gate1_pass"]]
        summary["by_arm"][arm] = {
            "n": len(sub),
            "gate1_perception_acc": round(len(g1_passed) / max(1, len(sub)), 4),
            "gate2_text_knowledge_acc": round(sum(r["gate2_pass"] for r in sub) / max(1, len(sub)), 4),
            "gate3_affect_acc_all": round(sum(r["gate3_pass"] for r in sub) / max(1, len(sub)), 4),
            "gate3_given_gate1_pass": round(sum(r["gate3_pass"] for r in g1_passed) / max(1, len(g1_passed)), 4),
        }

    out_path = out_dir / f"p3_4gate_{model_id.replace('/', '_')}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["by_arm"], indent=2, ensure_ascii=False))
    print(f"Saved 4-Gate eval -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-id", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    eval_qwen_vl_4gate(args.manifest, args.model_id, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
