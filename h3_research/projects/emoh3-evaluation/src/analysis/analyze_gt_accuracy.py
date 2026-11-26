#!/usr/bin/env python3
"""Accuracy of the two frame-span protocols against EXTERNAL labels.

Reliability (analyze_basefull.py::reliability_all_pairs) can only show that judges agree more; it
cannot show that a protocol is *right*. This script answers the accuracy question without any
annotation work of our own, using the 14 source clips referenced by the training manifest. Those
clips come from the same production batch as the known-emotion portraits the paper already treats as
ground truth, and their intended expression is in the filename
(e.g. ``sad_1534125952_v1_raw.mp4``, ``pres_scared_1538613676_base.mp4``).

Provenance caveat, stated in the paper: these are the batch's *intended* labels, vetted by the
dataset builder when the clips were selected as training targets, not our own annotation. They are
external to our judges, which is what makes the comparison meaningful.

Scoring: strict = judge word equals the label; lenient additionally accepts the declared synonym
groups below, the same rule used for the portrait sanity check.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGDIR = PROJECT_ROOT / "results" / "judge15-logs"
PROTOCOLS = [("0-1 s", "judge_gt1s_{}.jsonl"), ("full clip", "judge_gtfull_{}.jsonl")]
JUDGES = [("Qwen2-VL-7B", "q2"), ("Qwen2.5-VL-7B", "q25"), ("Qwen3-VL-32B", "q3vl")]

# Judge taxonomy: happy joyful sad angry worried cheerful surprised relieved anxious curious
# neutral disgusted fearful amused other
SYNONYM = {
    "joyful": {"happy", "cheerful", "amused"},
    "angry": {"angry"},
    "sad": {"sad"},
    "surprised": {"surprised"},
    "amused": {"happy", "joyful", "cheerful"},
    "anxious": {"worried", "fearful"},
    "curious": {"curious", "surprised"},
    "excited": {"happy", "joyful", "cheerful", "surprised"},
    "relaxed": {"relieved", "neutral"},
    "scared": {"fearful", "worried"},
}


def label_of(case: str) -> str:
    """`pres_scared_1538613676_base` -> scared; `sad_1534125952_v1_raw` -> sad."""
    parts = case.split("_")
    return parts[1] if parts[0] == "pres" else parts[0]


def load(fn: str) -> dict:
    p = LOGDIR / fn
    if not p.exists():
        return {}
    out = {}
    for line in p.open():
        j = json.loads(line)
        word = str(j.get("verdict", "")).split(":")[0].strip().lower()
        out[j["case"]] = word
    return out


def mcnemar(pairs) -> float:
    """Exact two-sided McNemar over (before, after) hit pairs."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    print(f"log dir: {LOGDIR}")
    print("=== Does full-clip scoring recover the labelled emotion more often? ===")
    any_data = False
    for name, tag in JUDGES:
        short, full = load(PROTOCOLS[0][1].format(tag)), load(PROTOCOLS[1][1].format(tag))
        keys = sorted(set(short) & set(full))
        if not keys:
            print(f"  {name}: no paired logs yet")
            continue
        any_data = True
        for mode in ("strict", "lenient"):
            pairs, rows = [], []
            for k in keys:
                lab = label_of(k)
                ok = SYNONYM.get(lab, set()) | {lab} if mode == "lenient" else {lab}
                hs, hf = short[k] in ok, full[k] in ok
                pairs.append((hs, hf))
                rows.append((k, lab, short[k], full[k], hs, hf))
            p = mcnemar(pairs)
            print(f"  {name} [{mode}]: 0-1 s {sum(a for a, _ in pairs)}/{len(pairs)} -> "
                  f"full clip {sum(b for _, b in pairs)}/{len(pairs)}, exact McNemar p={p:.3f}")
            if mode == "lenient":
                for k, lab, a, b, ha, hb in rows:
                    flag = "  " if ha == hb else ("->" if hb else "<-")
                    print(f"    {flag} {k[:34]:34s} label={lab:10s} 1s={a:10s} full={b}")
    if not any_data:
        print("  (run run_gt.sh on the cluster, then copy judge_gt{1s,full}_*.jsonl here)")
    print("\nNOTE: labels are the source batch's intended expressions (same provenance as the")
    print("known-emotion portraits used for the judge sanity check), not our own annotation. A")
    print("judge that cannot see an emotion at all fails under both protocols, so this measures the")
    print("protocol only within the instrument's competence.")


if __name__ == "__main__":
    main()
