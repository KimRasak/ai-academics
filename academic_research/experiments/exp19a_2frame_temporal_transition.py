#!/usr/bin/env python3
"""
Exp 19A (P3 Temporal / 2-Frame Anime Animation Transition Stress Test!):

Tests the ultimate comparative / video-context question:
  Even when we hand SOTA VLMs BOTH Frame 1 (`orig`, clean cheerful smiling portrait) AND
  Frame 2 (the exact same character moments later with a pixel-exact non-biological anime
  symbol added and ZERO mouth change, mouth_l1=0.0!) SIDE-BY-SIDE in the SAME prompt:
    "Frame 1 shows an anime character initially. Frame 2 shows the exact same character moments later.
     Did the character's TRUE emotional state change from Frame 1 to Frame 2?"
      A) No change — still genuinely cheerful, bright, and carefree
      B) Yes — turned emotionally hollow, eerie, ominous, or sinister
      C) Yes — became shy, flustered, or bashfully embarrassed
      D) Yes — became awkward, uneasy, or nervously sweating
      E) Yes — became irritated, annoyed, or suppressing anger behind a smile

Evaluates across 24 characters (fc_00..fc_23) x 8 transitions = 192 2-frame transition pairs!
Also compares:
  - Direct 2-Frame Transition Prompt (no hint about graphic symbols)
  - Audit 2-Frame Transition Prompt (explicitly asks to compare graphic cues between Frame 1 and Frame 2)
  - Equal-Pixel Bio vs. Non-Bio 2-Frame Gap (`bio_angry_brows` vs `anger_vein_on` `💢`)!
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

LABELS5 = ["A", "B", "C", "D", "E"]
TRANSITIONS = {
    "placebo_hairpin": "A",
    "eye_highlight_off": "B",
    "forehead_shadow_on": "B",
    "dead_eyes_and_shadow": "B",
    "blush_lines_on": "C",
    "sweat_drop_on": "D",
    "bio_angry_brows": "E",
    "anger_vein_on": "E",
}

DIRECT_2F_PROMPT = (
    "You are given TWO sequential frames of the same 2D anime character: Frame 1 (first image) and Frame 2 (second image).\n"
    "Compare Frame 2 to Frame 1 and judge whether the character's TRUE emotional / psychological state changed.\n"
    "Choose the SINGLE best label:\n"
    "A) No emotional change — still genuinely cheerful, bright, and happily carefree\n"
    "B) Turned emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Became shy, flustered, or bashfully embarrassed\n"
    "D) Became awkward, uneasy, or nervously sweating\n"
    "E) Became irritated, annoyed, or suppressing anger behind a smile\n"
    "Answer with ONLY a single letter: A, B, C, D, or E."
)

AUDIT_2F_PROMPT = (
    "You are given TWO sequential frames of the same 2D anime character: Frame 1 (first image) and Frame 2 (second image).\n"
    "Carefully compare Frame 2 against Frame 1 for ANY added or removed anime graphic affect cues "
    "(such as furrowed angry brows, a red anime cross-vein anger mark, dark forehead shadow, "
    "missing/hollow eye highlights, cheek blush lines, or a floating sweat drop), ignoring ordinary hairpins.\n"
    "In 2D anime, graphic affect symbols OVERRIDE an unchanged smiling mouth.\n"
    "Choose the SINGLE best label:\n"
    "A) No emotional change — still genuinely cheerful, bright, and happily carefree\n"
    "B) Turned emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Became shy, flustered, or bashfully embarrassed\n"
    "D) Became awkward, uneasy, or nervously sweating\n"
    "E) Became irritated, annoyed, or suppressing anger behind a smile\n"
    "Answer with ONLY a single letter: A, B, C, D, or E."
)


def _mcnemar_p(ca: list[int], cb: list[int]) -> float:
    b01 = sum(1 for a, b in zip(ca, cb) if a == 0 and b == 1)
    b10 = sum(1 for a, b in zip(ca, cb) if a == 1 and b == 0)
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * p)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest48", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-chars", type=int, default=24)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    chars = sorted({r["char_id"] for r in recs})[: args.n_chars]
    by_cv = {(r["char_id"], r["variant"]): r["image_path"] for r in recs}

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ids5 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in LABELS5]

    def eval_pair(im1: Image.Image, im2: Image.Image, prompt: str):
        m = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": im1},
                    {"type": "image", "image": im2},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[t], images=[im1, im2], return_tensors="pt").to(model.device)
        out = model(**inp)
        lg = out.logits[0, -1, ids5].float().cpu().numpy()
        pr = np.exp(lg - lg.max()); pr /= pr.sum()
        return LABELS5[int(np.argmax(lg))], float(pr[0])

    out_summary = {}
    for mode, prompt in [("direct_2frame", DIRECT_2F_PROMPT), ("audit_2frame", AUDIT_2F_PROMPT)]:
        by_v_ok = {v: [] for v in TRANSITIONS}
        by_v_pA = {v: [] for v in TRANSITIONS}
        for cid in chars:
            im1 = Image.open(by_cv[(cid, "orig")]).convert("RGB").resize((512, 512), Image.LANCZOS)
            for v, exp_l in TRANSITIONS.items():
                im2 = Image.open(by_cv[(cid, v)]).convert("RGB").resize((512, 512), Image.LANCZOS)
                pred, pA = eval_pair(im1, im2, prompt)
                by_v_ok[v].append(int(pred == exp_l))
                by_v_pA[v].append(pA)
        v_stats = {
            v: {"acc": round(float(np.mean(by_v_ok[v])), 4), "mean_pA_no_change": round(float(np.mean(by_v_pA[v])), 4)}
            for v in TRANSITIONS
        }
        conf7 = float(np.mean([v_stats[v]["acc"] for v in TRANSITIONS if v != "placebo_hairpin"]))
        bio_gap = float(np.mean(by_v_ok["bio_angry_brows"]) - np.mean(by_v_ok["anger_vein_on"]))
        bio_p = _mcnemar_p(by_v_ok["anger_vein_on"], by_v_ok["bio_angry_brows"])
        out_summary[mode] = {
            "macro8": round(float(np.mean([x["acc"] for x in v_stats.values()])), 4),
            "placebo_no_change_acc": v_stats["placebo_hairpin"]["acc"],
            "conf7_acc": round(conf7, 4),
            "bio_vs_vein_gap": round(bio_gap, 4),
            "bio_vs_vein_p": round(bio_p, 5),
            "by_variant": v_stats,
        }
        print(
            f"[{Path(args.model_dir).name}] {mode:14s} | Macro8={out_summary[mode]['macro8']:.3f} Placebo={v_stats['placebo_hairpin']['acc']:.2f} Conf7={conf7:.3f} | "
            f"eye={v_stats['eye_highlight_off']['acc']:.2f}(pA={v_stats['eye_highlight_off']['mean_pA_no_change']:.2f}) "
            f"shd={v_stats['forehead_shadow_on']['acc']:.2f} d+s={v_stats['dead_eyes_and_shadow']['acc']:.2f} "
            f"bls={v_stats['blush_lines_on']['acc']:.2f} swt={v_stats['sweat_drop_on']['acc']:.2f} "
            f"bio={v_stats['bio_angry_brows']['acc']:.2f} vein={v_stats['anger_vein_on']['acc']:.2f}(gap={bio_gap:+.2f},p={bio_p:.4f})",
            flush=True,
        )

    out_p = args.out_dir / f"exp19a_2frame_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "n_chars": len(chars), "results": out_summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
