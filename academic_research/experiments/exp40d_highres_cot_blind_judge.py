#!/usr/bin/env python3
"""
Exp 40D (High-Res 320px + Chain-of-Thought Order-Balanced Blind `Qwen3-VL-32B` Video Judge):

Upgrades Exp 40B/C's blind judge to eliminate 192px thumbnail legibility loss (where 2px eye specular highlights
and fine anime hatching lines became unreadable):
  - Displays crisp 320x320 keyframes: [Input t=0 | Mid t=24 | Final t=48] for Video 1 (top) vs Video 2 (bottom)
  - Requires a concise 2-sentence visual `OBSERVATION:` before emitting `GRAMMAR / IDENTITY / OVERALL`
  - Strictly blind & order-balanced (`AB` AND `BA`; only order-consistent decisions count as WIN_A / WIN_B, else TIE)
  - Reports exact two-sided Binomial p-values across `ALL_GRAMMAR_5`, `UNSEEN_COMP_2`, `SAFETY_2`, and per-regime!
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


PROMPT_BY_REGIME = {
    "S3_NonBioAffect": "The character's bright white eye highlights vanish leaving dull flat eyes AND a translucent 2D anime dark gloom shadow falls across ONLY their upper forehead/brow, while their mouth holds a calm smile and their hair color, uniform, cheeks, and bright background stay intact (NOT a pitch-black room/silhouette blackout or glowing demon eyes).",
    "G1_HighlightOff_Only": "In bright even lighting, ONLY the small white circular specular highlight dots inside both open eyes fade away leaving flat matte colored irises, while eyes stay WIDE OPEN (not closed, not pitch-black empty voids) and bright hair, smile, and clothes stay unchanged.",
    "G2_ForeheadShadow_Clean": "In bright daylight, a translucent slate-blue/indigo 2D anime gloom shadow with fine vertical hatching lines falls across the upper forehead/brow while keeping hair color, smile, cheeks, and bright studio lighting intact (NOT an opaque gray cardboard box, NOT teal/blue hair dye).",
    "G3_AngerVein_Pop": "A classic stylized 2D Japanese anime red cross-bracket anger vein symbol (💢) pops onto the upper temple while the character holds a polite smile (NOT a 3D plastic X hairclip, ribbon, or bloody cut).",
    "G4_SweatDrop_Slide": "A single classic fixed-shape glossy translucent light-blue 2D anime comic sweat drop (💧) appears beside the temple and slides gently downward with an awkward smile (NOT crying tears, NOT a stretching slime tentacle or floating bubbles).",
    "C1_Comp_HeadTurn_Shadow": "The character smoothly turns their head to a three-quarter profile with a calm smile while eye highlights vanish AND a translucent 2D anime gloom shadow wraps across the upper forehead (hair color and white background remain bright).",
    "C2_Comp_HeadTurn_Vein": "The character smoothly turns their head to a three-quarter profile holding a polite smile while a classic red 2D anime anger vein symbol (💢) stays cleanly attached to the upper temple.",
    "S1_Safety_AnimeHeadTurn": "The anime character smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent identity.",
    "S1_Safety_RealHeadTurn": "The photorealistic human smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent identity.",
}


def _make_highres_pair_panel(kf_a: Path, kf_b: Path) -> Image.Image:
    """High-res 320x320 3-frame storyboard: Input t=0, Mid t=24 (k3), Final t=48 (k5)."""
    sz = 320
    fns = [("input.png", "Input (t=0)"), ("k3.png", "Mid Frame"), ("k5.png", "Final Frame")]
    W = sz * len(fns)
    H = (sz + 28) * 2
    canvas = Image.new("RGB", (W, H), (18, 18, 22))
    dr = ImageDraw.Draw(canvas)
    for row_i, (lbl, kfd) in enumerate([("VIDEO 1 (Top Row)", kf_a), ("VIDEO 2 (Bottom Row)", kf_b)]):
        y0 = row_i * (sz + 28)
        dr.text((10, y0 + 6), lbl, fill=(255, 228, 130))
        for col_i, (fn, col_lbl) in enumerate(fns):
            p = kfd / fn
            if not p.exists():
                p = kfd / "k4.png"
            im = Image.open(p).convert("RGB").resize((sz, sz), Image.LANCZOS)
            canvas.paste(im, (col_i * sz, y0 + 28))
            dr.text((col_i * sz + 180, y0 + 6), col_lbl, fill=(190, 195, 210))
    return canvas


def _parse_verdict(txt: str) -> dict:
    out = {"grammar": "TIE", "identity": "TIE", "overall": "TIE"}
    for key in ["grammar", "identity", "overall"]:
        matches = re.findall(rf"{key}\s*[:=]\s*(1|2|VIDEO\s*1|VIDEO\s*2|TIE)", txt, flags=re.IGNORECASE)
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
    ap.add_argument("--mode-a", default="M0_Base27B")
    ap.add_argument("--mode-b", default="M3_LoRA_Dual")
    ap.add_argument("--model-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/weights/Qwen3-VL-32B-Instruct"))
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
    print(f"[highres-judge] found {len(pairs)} matched pairs ({args.mode_a} vs {args.mode_b})", flush=True)

    proc = AutoProcessor.from_pretrained(str(args.model_dir))
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir), dtype=torch.bfloat16
    ).to("cuda").eval()

    results = []
    for pid, rname, kfa, kfb in pairs:
        target_desc = PROMPT_BY_REGIME.get(rname, rname)
        q_text = (
            f"You are an expert Japanese anime animation director comparing two AI-generated video clips shown in high resolution "
            f"(3 columns: Input t=0, Mid frame, Final frame).\n"
            f"Target instruction: {target_desc}\n\n"
            f"First write a concise 2-sentence OBSERVATION describing the concrete visual difference between VIDEO 1 (top row) and VIDEO 2 (bottom row) "
            f"(explicitly noting physical-literalism errors like full-frame blackout, teal hair dye, opaque cardboard rectangle, plastic hairclip, slime tentacle, "
            f"refusal/no-change, or pitch-black eye voids).\n"
            f"Then give your final ratings on 3 lines (values must be 1, 2, or TIE):\n"
            f"OBSERVATION: <2 concise sentences>\n"
            f"GRAMMAR: <1|2|TIE>\nIDENTITY: <1|2|TIE>\nOVERALL: <1|2|TIE>"
        )

        verdicts = {}
        for order_tag, pa, pb in [("AB", kfa, kfb), ("BA", kfb, kfa)]:
            panel = _make_highres_pair_panel(pa, pb)
            msgs = [{"role": "user", "content": [{"type": "image", "image": panel}, {"type": "text", "text": q_text}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=[panel], return_tensors="pt").to(model.device)
            gen = model.generate(**inp, max_new_tokens=160, do_sample=False)
            ans = proc.decode(gen[0][inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            v = _parse_verdict(ans)
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
        for k in ["grammar", "identity", "overall"]:
            m1, m2 = verdicts["AB"]["mapped"][k], verdicts["BA"]["mapped"][k]
            if m1 == "B" and m2 == "B":
                strict[k] = "WIN_B"
            elif m1 == "A" and m2 == "A":
                strict[k] = "WIN_A"
            else:
                strict[k] = "TIE"
        rec = {"pair_idx": pid, "regime": rname, "strict": strict, "verdicts": verdicts}
        results.append(rec)
        print(f"  [{pid:02d} {rname:25s}] grammar={strict['grammar']:5s} identity={strict['identity']:5s} overall={strict['overall']:5s}", flush=True)

    print(f"\n=== EXP 40D HIGH-RES CoT BLIND Qwen3-VL-32B JUDGE: {args.mode_b} vs {args.mode_a} (n={len(results)} pairs) ===", flush=True)
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
        line = f"  {gname:26s} (n={len(sub):2d}) |"
        for ax in ["grammar", "identity", "overall"]:
            wb = sum(1 for r in sub if r["strict"][ax] == "WIN_B")
            wa = sum(1 for r in sub if r["strict"][ax] == "WIN_A")
            ti = sum(1 for r in sub if r["strict"][ax] == "TIE")
            wr = (wb + 0.5 * ti) / max(len(sub), 1)
            pv = _binom_p(wb, wa)
            gs[ax] = {"win_b": wb, "win_a": wa, "tie": ti, "win_rate": round(wr, 4), "p_binom": round(pv, 5)}
            line += f" {ax}: {wb}W/{wa}L/{ti}T ({100*wr:.1f}%, p={pv:.4f}) |"
        summary[gname] = gs
        print(line, flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"mode_a": args.mode_a, "mode_b": args.mode_b, "summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(f"Saved -> {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
