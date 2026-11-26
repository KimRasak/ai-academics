#!/usr/bin/env python3
"""Run Exp 1 (P6 Minimal-Pair T-Junction Flip) on frontier VLMs via OneAPIClient.

Tests Paired Flip Consistency on L3_Flat2D (pure T-junctions) vs L1_Shaded (cast drop shadow),
across identical_color (50/50 prior) and distinct_color regimes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path("/Users/jinzili/code/synclub_expression_animation")
sys.path.insert(0, str(REPO_ROOT / "oneapi"))
from oneapi_util import OneAPIClient  # noqa: E402

PROMPT = (
    "Look carefully at the two overlapping flat ribbons in this image. "
    "One ribbon has a RED dot on it, and the other ribbon has a BLUE dot on it. "
    "Which ribbon is physically IN FRONT OF (on top of / occluding) the other at the crossing? "
    "Answer with ONLY a single word: RED or BLUE."
)


def parse_choice(text: str) -> str:
    up = (text or "").strip().upper()
    if up.startswith("RED"):
        return "A"
    if up.startswith("BLUE"):
        return "B"
    has_red = "RED" in up
    has_blue = "BLUE" in up
    if has_red and not has_blue:
        return "A"
    if has_blue and not has_red:
        return "B"
    return "UNK"


def eval_one(model: str, rec: dict[str, Any]) -> dict[str, Any]:
    client = OneAPIClient(model=model, system="You are a precise visual perception assistant. Follow output format strictly.")
    try:
        raw = client.chat(PROMPT, image_path=rec["dots_image_path"])
        pred = parse_choice(raw)
        err = None
    except Exception as exc:
        raw = ""
        pred = "ERR"
        err = str(exc)
    return {
        "pair_id": rec["pair_id"],
        "regime": rec["regime"],
        "render_level": rec["render_level"],
        "order": rec["order"],
        "front_gt": rec["front_gt"],
        "pred": pred,
        "correct": pred == rec["front_gt"],
        "raw": raw[:120],
        "error": err,
    }


def run_model(manifest_path: Path, model: str, max_pairs: int, workers: int, out_dir: Path) -> dict[str, Any]:
    rows = [json.loads(ln) for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Select balanced subset of pair_ids across identical_color and distinct_color
    id_same = sorted({r["pair_id"] for r in rows if r["regime"] == "identical_color"})[: max_pairs // 2]
    id_diff = sorted({r["pair_id"] for r in rows if r["regime"] == "distinct_color"})[: max_pairs - len(id_same)]
    keep_ids = set(id_same + id_diff)
    sub = [r for r in rows if r["pair_id"] in keep_ids]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(eval_one, model, r) for r in sub]
        for f in as_completed(futs):
            results.append(f.result())

    lookup = {(r["pair_id"], r["render_level"], r["order"]): r for r in results}
    summary: dict[str, Any] = {"model": model, "n_images": len(results), "errors": sum(r["pred"] == "ERR" for r in results), "slices": {}}
    for rl in ["L1_Shaded", "L3_Flat2D"]:
        for reg in ["identical_color", "distinct_color", "ALL"]:
            pids = sorted({
                r["pair_id"]
                for r in sub
                if r["render_level"] == rl and (reg == "ALL" or r["regime"] == reg)
            })
            single_ok = 0
            flip_ok = 0
            for pid in pids:
                p1 = lookup[(pid, rl, "A_over_B")]
                p2 = lookup[(pid, rl, "B_over_A")]
                single_ok += int(p1["correct"]) + int(p2["correct"])
                if p1["correct"] and p2["correct"]:
                    flip_ok += 1
            summary["slices"][f"{rl}__{reg}"] = {
                "n_pairs": len(pids),
                "single_acc": round(single_ok / max(1, 2 * len(pids)), 4),
                "paired_flip_consistency": round(flip_ok / max(1, len(pids)), 4),
            }

    out_path = out_dir / f"vlm_p6_{model.replace(' ', '_')}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== {model} ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["gpt-5.5", "glm-5.1"])
    ap.add_argument("--max-pairs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        run_model(args.manifest, m, args.max_pairs, args.workers, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
