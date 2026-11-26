#!/usr/bin/env python3
"""
Exp 40B (Order-Balanced Blind `Qwen3-VL-32B` Pairwise Video Judge for Flagship 27B MoE `Base` vs `LoRA_Dual` / `High` vs `Low`):

Runs strictly blind, order-balanced (`(A,B)` AND `(B,A)`) pairwise visual evaluation using `Qwen3-VL-32B-Instruct`
across all UNSEEN held-out characters, 2D Anime Grammar regimes (`S3`, `G1`–`G4`), UNSEEN Compositional Motion+Grammar
prompts (`C1`, `C2`), AND Zero-Forgetting Safety Controls (`S1_Anime`, `S1_Real`)!

Scores 3 axes:
  1. `grammar`:   Faithful execution of the requested prompt/2D anime grammar (without physical literalism)
  2. `identity`:  Preservation of character identity, hair color, clothing, smile, and crisp 2D cel style
  3. `overall`:   Overall preference as a broadcast-quality video clip
Reports strict wins/losses/ties, balanced win rate, and exact two-sided Binomial p-values!
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
    "S3_NonBioAffect": "The character's eye highlights vanish leaving hollow lifeless eyes while a sharp dark shadow falls across their upper forehead, yet their mouth stays frozen in a calm smile (preserving hair color, uniform, and crisp 2D anime style without blacking out the whole character).",
    "G1_HighlightOff_Only": "In bright even lighting, ONLY the small white circular highlight dots inside both open eyes fade away leaving flat matte dull irises, while eyes stay OPEN and bright hair, smile, and clothes stay unchanged.",
    "G2_ForeheadShadow_Clean": "In bright daylight, a translucent slate-blue 2D anime gloom shadow band with fine vertical ink lines appears strictly across the upper forehead/brow while keeping bright hair color, smile, cheeks, and white background unchanged (NOT an opaque hairband or dyed stripe).",
    "G3_AngerVein_Pop": "A classic stylized 2D Japanese anime red cross-bracket anger vein symbol pops onto the upper temple and pulses while the character holds a polite smile (NOT a physical X hairclip, ribbon, or bloody wound).",
    "G4_SweatDrop_Slide": "A single classic fixed-shape stylized translucent blue 2D anime comic sweat drop appears beside the temple and slides gently downward with an awkward smile (NOT a long stretching slime tentacle or tears).",
    "C1_Comp_HeadTurn_Shadow": "The character smoothly turns their head to a three-quarter profile with a calm smile while eye highlights vanish and a translucent 2D anime gloom shadow falls strictly on the upper forehead (hair and background bright).",
    "C2_Comp_HeadTurn_Vein": "The character smoothly turns their head to a three-quarter profile holding a polite smile while a classic red 2D anime anger vein symbol pops on their temple.",
    "S1_Safety_AnimeHeadTurn": "The anime character smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent appearance.",
    "S1_Safety_RealHeadTurn": "The photorealistic human smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent identity.",
}


def _make_pair_panel(kf_a: Path, kf_b: Path) -> Image.Image:
    """Build a blind stacked storyboard: Row 1 = Video 1 (Input + k1..k5), Row 2 = Video 2 (Input + k1..k5)."""
    sz = 192
    fns = ["input.png", "k1.png", "k2.png", "k3.png", "k4.png", "k5.png"]
    W = sz * len(fns)
    H = (sz + 24) * 2
    canvas = Image.new("RGB", (W, H), (20, 20, 24))
    dr = ImageDraw.Draw(canvas)
    for row_i, (lbl, kfd) in enumerate([("VIDEO 1 (Top Row: Input t=0 -> Generated Frames)", kf_a), ("VIDEO 2 (Bottom Row: Input t=0 -> Generated Frames)", kf_b)]):
        y0 = row_i * (sz + 24)
        dr.text((8, y0 + 5), lbl, fill=(240, 240, 240))
        for col_i, fn in enumerate(fns):
            im = Image.open(kfd / fn).convert("RGB").resize((sz, sz), Image.LANCZOS)
            canvas.paste(im, (col_i * sz, y0 + 24))
    return canvas


def _parse_verdict(txt: str) -> dict:
    out = {"grammar": "TIE", "identity": "TIE", "overall": "TIE"}
    for key in ["grammar", "identity", "overall"]:
        m = re.search(rf"{key}\s*[:=]\s*(1|2|VIDEO\s*1|VIDEO\s*2|TIE)", txt, flags=re.IGNORECASE)
        if m:
            v = m.group(1).upper().replace(" ", "")
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
    ap.add_argument("--eval-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp40_wan22_a14b_moe_lora/eval_2x2"))
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
        pid, rname = m["pair_idx"], m["regime"]
        kfa = kf_root / f"{pid:02d}_{rname}_{args.mode_a}"
        kfb = kf_root / f"{pid:02d}_{rname}_{args.mode_b}"
        if (kfa / "k5.png").exists() and (kfb / "k5.png").exists():
            pairs.append((pid, rname, kfa, kfb))
    print(f"[judge] found {len(pairs)} matched pairs ({args.mode_a} vs {args.mode_b})", flush=True)

    proc = AutoProcessor.from_pretrained(str(args.model_dir))
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir), dtype=torch.bfloat16
    ).to("cuda").eval()

    results = []
    for pid, rname, kfa, kfb in pairs:
        target_desc = PROMPT_BY_REGIME.get(rname, rname)
        q_text = (
            f"You are an expert Japanese anime animation director judging two AI-generated video clips shown as 6-frame storyboards "
            f"(leftmost column is the shared input image t=0, followed by 5 generated frames).\n"
            f"Target instruction: {target_desc}\n\n"
            f"Compare VIDEO 1 (top row) vs VIDEO 2 (bottom row) strictly on:\n"
            f"1. GRAMMAR: Which clip more faithfully and naturally executes the target instruction without physical-literalist failures "
            f"(e.g., turning the whole face/hair into a pitch-black silhouette, closing eyes instead of removing highlights, painting an opaque hairband, "
            f"sticking a red plastic X hairclip/wound, or turning a comic sweat drop into a dripping slime tentacle)?\n"
            f"2. IDENTITY: Which clip better preserves the character's original face, hair color, clothes, smile, and crisp 2D style?\n"
            f"3. OVERALL: Which clip is better overall?\n\n"
            f"Reply in EXACTLY 3 lines (values must be 1, 2, or TIE):\n"
            f"GRAMMAR: <1|2|TIE>\nIDENTITY: <1|2|TIE>\nOVERALL: <1|2|TIE>"
        )

        # Order 1: A=1, B=2; Order 2: B=1, A=2
        verdicts = {}
        for order_tag, pa, pb in [("AB", kfa, kfb), ("BA", kfb, kfa)]:
            panel = _make_pair_panel(pa, pb)
            msgs = [{"role": "user", "content": [{"type": "image", "image": panel}, {"type": "text", "text": q_text}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=[panel], return_tensors="pt").to(model.device)
            gen = model.generate(**inp, max_new_tokens=64, do_sample=False)
            ans = proc.decode(gen[0][inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            v = _parse_verdict(ans)
            # Map back to A/B/TIE
            mapped = {}
            for k, val in v.items():
                if val == "TIE":
                    mapped[k] = "TIE"
                elif order_tag == "AB":
                    mapped[k] = "A" if val == "1" else "B"
                else:
                    mapped[k] = "B" if val == "1" else "A"
            verdicts[order_tag] = {"raw": ans, "mapped": mapped}

        # Strict order-consistent verdict
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

    # Aggregate summary
    print(f"\n=== EXP 40B BLIND Qwen3-VL-32B JUDGE: {args.mode_b} vs {args.mode_a} (n={len(results)} pairs) ===", flush=True)
    groups = {
        "ALL_GRAMMAR_5 (S3,G1-G4)": [r for r in results if r["regime"] in ("S3_NonBioAffect", "G1_HighlightOff_Only", "G2_ForeheadShadow_Clean", "G3_AngerVein_Pop", "G4_SweatDrop_Slide")],
        "UNSEEN_COMP_2 (C1,C2)": [r for r in results if r["regime"].startswith("C")],
        "SAFETY_2 (S1_Anime,Real)": [r for r in results if r["regime"].startswith("S1_Safety")],
        "GRAND_TOTAL": results,
    }
    summary = {}
    for gname, sub in groups.items():
        if not sub: continue
        summary[gname] = {}
        line = f"  {gname:26s} (n={len(sub):2d}) |"
        for k in ["grammar", "identity", "overall"]:
            w = sum(1 for r in sub if r["strict"][k] == "WIN_B")
            l = sum(1 for r in sub if r["strict"][k] == "WIN_A")
            t = len(sub) - w - l
            wr = (w + 0.5 * t) / len(sub)
            pv = _binom_p(w, l)
            summary[gname][k] = {"W": w, "L": l, "T": t, "bal_wr": round(wr, 4), "p": round(pv, 6)}
            line += f" {k}: {w}W/{l}L/{t}T ({wr*100:.1f}%, p={pv:.4f}) |"
        print(line, flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"mode_a": args.mode_a, "mode_b": args.mode_b, "summary": summary, "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
