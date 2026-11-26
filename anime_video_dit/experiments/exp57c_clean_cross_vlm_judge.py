#!/usr/bin/env python3
"""
Exp 57C: Clean Judge-Agnostic Multi-VLM Blind Evaluation (`Qwen2.5-VL-32B-Instruct` + `OpenGVLab/InternVL3-14B-hf`)

Fixes two prompt/format artifacts uncovered when porting `exp40d` to non-Qwen3 VLM families:
  1. Replaces the ambiguous label `GRAMMAR` (misread by Qwen2.5-VL / InternVL3 as "linguistic/animation grammar")
     with `EFFECT` ("which video better renders the requested 2D anime visual effect without physical-literalism errors").
  2. Enforces strict 4-line output (`OBSERVATION` + `EFFECT` + `IDENTITY` + `OVERALL`, no post-rating prose)
     with `max_new_tokens=220` and markdown-tolerant (`**`) parsing so ZERO verdicts are truncated.
  3. Uses the exact same high-res `320x320` 3-frame storyboard (`Input | Mid | Final`) and strict order-balanced
     (`AB` & `BA`) blind protocol as `exp40d`, enabling direct inter-judge agreement & Cohen's kappa!
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp40d_highres_cot_blind_judge as e40d


def _parse_clean_verdict(txt: str) -> dict:
    clean = txt.replace("**", "").replace("__", "")
    out = {"effect": "TIE", "identity": "TIE", "overall": "TIE"}
    for key in ["effect", "identity", "overall"]:
        matches = re.findall(rf"{key}\s*[:=]\s*(1|2|VIDEO\s*1|VIDEO\s*2|TIE)", clean, flags=re.IGNORECASE)
        if matches:
            v = matches[-1].upper().replace(" ", "")
            out[key] = "1" if v in ("1", "VIDEO1") else ("2" if v in ("2", "VIDEO2") else "TIE")
    return out


def _binom_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p_tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * p_tail)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--mode-a", required=True)
    ap.add_argument("--mode-b", required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--judge-tag", default="VLM")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    kf_root = args.eval_dir / "keyframes"
    meta_files = sorted((args.eval_dir / "meta").glob(f"*_{args.mode_b}.json"))
    pairs = []
    for mp in meta_files:
        m = json.loads(mp.read_text(encoding="utf-8"))
        pid = m.get("pair_idx", m.get("idx", 0))
        rname = m["regime"]
        kfa = kf_root / f"{pid:02d}_{rname}_{args.mode_a}"
        kfb = kf_root / f"{pid:02d}_{rname}_{args.mode_b}"
        if (kfa / "k5.png").exists() and (kfb / "k5.png").exists():
            pairs.append((pid, rname, kfa, kfb))
    print(f"[{args.judge_tag}] found {len(pairs)} matched pairs ({args.mode_a} vs {args.mode_b})", flush=True)

    proc = AutoProcessor.from_pretrained(str(args.model_dir))
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir), dtype=torch.bfloat16
    ).to("cuda").eval()

    results = []
    for pid, rname, kfa, kfb in pairs:
        target_desc = e40d.PROMPT_BY_REGIME.get(rname, rname)
        q_text = (
            f"Compare VIDEO 1 (top row) and VIDEO 2 (bottom row). Each row shows 3 frames: Input t=0, Mid frame, Final frame.\n"
            f"Requested 2D anime visual effect: {target_desc}\n\n"
            f"Output ONLY these 4 plain-text lines (no markdown bold, no extra commentary after the ratings):\n"
            f"OBSERVATION: <1 concise sentence comparing how well VIDEO 1 vs VIDEO 2 executes the requested effect and preserves character identity>\n"
            f"EFFECT: <1 or 2 or TIE> (which video better executes the requested 2D visual effect without blackout, hair dye, cardboard box, plastic hairclip, slime, or refusal)\n"
            f"IDENTITY: <1 or 2 or TIE> (which video better preserves the character's original hair color, face, clothes, and bright clean studio lighting)\n"
            f"OVERALL: <1 or 2 or TIE> (overall better animation)"
        )

        verdicts = {}
        for order_tag, pa, pb in [("AB", kfa, kfb), ("BA", kfb, kfa)]:
            panel = e40d._make_highres_pair_panel(pa, pb)
            msgs = [{"role": "user", "content": [{"type": "image", "image": panel}, {"type": "text", "text": q_text}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=[panel], return_tensors="pt").to(model.device)
            gen = model.generate(**inp, max_new_tokens=200, do_sample=False)
            ans = proc.decode(gen[0][inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            v = _parse_clean_verdict(ans)
            mapped = {}
            for k, val in v.items():
                if val == "TIE":
                    mapped[k] = "TIE"
                elif order_tag == "AB":
                    mapped[k] = "A" if val == "1" else "B"
                else:
                    mapped[k] = "B" if val == "1" else "A"
            verdicts[order_tag] = {"raw": ans, "mapped": mapped}

        strict = {}
        for k in ["effect", "identity", "overall"]:
            m1, m2 = verdicts["AB"]["mapped"][k], verdicts["BA"]["mapped"][k]
            if m1 == "B" and m2 == "B":
                strict[k] = "WIN_B"
            elif m1 == "A" and m2 == "A":
                strict[k] = "WIN_A"
            else:
                strict[k] = "TIE"
        rec = {"pair_idx": pid, "regime": rname, "strict": strict, "verdicts": verdicts}
        results.append(rec)
        print(f"  [{pid:02d} {rname:25s}] effect={strict['effect']:5s} identity={strict['identity']:5s} overall={strict['overall']:5s}", flush=True)

    print(f"\n=== EXP 57C MULTI-VLM BLIND JUDGE ({args.judge_tag}): {args.mode_b} vs {args.mode_a} (n={len(results)} pairs) ===", flush=True)
    groups = {
        "ALL_GRAMMAR_5 (S3,G1-G4)": [r for r in results if r["regime"] in ("S3_NonBioAffect", "G1_HighlightOff_Only", "G2_ForeheadShadow_Clean", "G3_AngerVein_Pop", "G4_SweatDrop_Slide")],
        "UNSEEN_COMP_2 (C1,C2)": [r for r in results if r["regime"] in ("C1_Comp_HeadTurn_Shadow", "C2_Comp_HeadTurn_Vein")],
        "SAFETY_2 (S1_Anime,Real)": [r for r in results if r["regime"] in ("S1_Safety_AnimeHeadTurn", "S1_Safety_RealHeadTurn")],
        "GRAND_TOTAL": results,
    }
    summary = {}
    for gname, sub in groups.items():
        if not sub:
            continue
        gs = {}
        parts = []
        for k in ["effect", "identity", "overall"]:
            w = sum(1 for r in sub if r["strict"][k] == "WIN_B")
            l = sum(1 for r in sub if r["strict"][k] == "WIN_A")
            t = sum(1 for r in sub if r["strict"][k] == "TIE")
            wr = 100.0 * (w + 0.5 * t) / len(sub)
            p = _binom_p(w, l)
            gs[k] = {"W": w, "L": l, "T": t, "WR": round(wr, 2), "p": round(p, 6)}
            parts.append(f"{k}: {w}W/{l}L/{t}T ({wr:.1f}%, p={p:.4f})")
        summary[gname] = gs
        print(f"  {gname:26s} (n={len(sub):2d}) | " + " | ".join(parts), flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"judge": args.judge_tag, "mode_a": args.mode_a, "mode_b": args.mode_b, "summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
