#!/usr/bin/env python3
"""Frame-span control: does judging the WHOLE clip change the verdicts?

Round-7 discovery. The protocol used throughout this project samples the judge's 8 frames at
t = i/7 SECONDS, i.e. over the first ~1.0 s of a 5.17 s (124-frame, 24 fps) clip -- about 19% of
the clip, and precisely the part that is closest to the reference portrait. `extract8_full.py`
re-extracts 8 frames spanning the whole clip; this script compares the two protocols on identical
clips, identical judges and greedy decoding, so the ONLY change is which frames the judge sees.

Inputs (judge15_logs/, override with JUDGE_LOG_DIR)
  judge15_8fr.jsonl        Qwen2-VL,      0-1 s frames  (the paper's protocol)
  judge_all88_q3vl.jsonl   Qwen3-VL-32B,  0-1 s frames
  judge_full_q2.jsonl      Qwen2-VL,      full-span frames
  judge_full_q3vl.jsonl    Qwen3-VL-32B,  full-span frames
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from math import comb
from pathlib import Path

EMOTIONS = ["happy", "joyful", "sad", "angry", "worried", "cheerful", "surprised",
            "relieved", "anxious", "curious", "neutral", "disgusted", "fearful",
            "amused", "other"]
SYN = {"joyful": "happy", "cheerful": "happy"}
RX = re.compile(r"\b(" + "|".join(EMOTIONS) + r")\b\s*:\s*(\d{1,3})", re.I)

# Each entry is (judge, 0-1 s logs, full-span log). The 0-1 s side is a list because the newly
# generated task-matched base matrix was judged into its own log (judge_basefull_*) under the same
# case keys as the full-span pass, so merging the two gives one paired set per judge.
PAIRS = [
    ("Qwen2-VL-7B", ["judge15_8fr.jsonl", "judge_basefull_q2.jsonl"], "judge_full_q2.jsonl"),
    ("Qwen2.5-VL-7B", ["judge_all88_q25.jsonl", "judge_basefull_q25.jsonl"],
     "judge_full_q25.jsonl"),
    ("Qwen3-VL-32B", ["judge_all88_q3vl.jsonl", "judge_basefull_q3vl.jsonl"],
     "judge_full_q3vl.jsonl"),
]


def load_many(names) -> dict:
    out: dict[str, str] = {}
    for nm in ([names] if isinstance(names, str) else names):
        out.update(load(nm))
    return out

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = [os.environ.get("JUDGE_LOG_DIR"),
              str(PROJECT_ROOT / "results" / "judge15-logs"),
              str(PROJECT_ROOT / "release" / "judge_logs"),
              str(Path(__file__).resolve().parent)]
LOGDIR = next((Path(c) for c in CANDIDATES if c and Path(c).is_dir()
               and any(Path(c).glob("judge*.jsonl"))), None)


def collapse(w: str) -> str:
    return SYN.get(w, w)


def last_emotion(v: str | None) -> str | None:
    hits = RX.findall(v or "")
    return hits[-1][0].lower() if hits else None


def load(name: str) -> dict[str, str]:
    p = (LOGDIR / name) if LOGDIR else Path(name)
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.open():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        w = last_emotion(rec.get("verdict"))
        if w:
            out[rec["case"]] = w
    return out


def target_of(case: str) -> str | None:
    m = re.search(r"_to_([a-z]+?)(?:_base|_lora|$)", case)
    return m.group(1) if m else None


def kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def arm(case: str) -> str:
    p = case.split("__")[0]
    if "basefull" in p or "base16" in p or "sweep_base" in p or "refswap_base" in p or p == "frames_x02ctrl":
        return "base"
    return "v3/LoRA"


def main() -> None:
    print(f"log dir: {LOGDIR}")
    for name, short_fn, full_fn in PAIRS:
        s, f = load_many(short_fn), load(full_fn)
        keys = sorted(set(s) & set(f))
        if not keys:
            print(f"\n{name}: no overlap ({short_fn} / {full_fn})")
            continue
        a = [collapse(s[k]) for k in keys]
        b = [collapse(f[k]) for k in keys]
        flips = [k for k, x, y in zip(keys, a, b) if x != y]
        print(f"\n=== {name}: 0-1 s frames vs full-span frames, n={len(keys)} clips ===")
        print(f"  self-agreement {1 - len(flips) / len(keys):.1%}  "
              f"Cohen kappa={kappa(a, b):.3f}  flips={len(flips)}/{len(keys)}")
        # direction of the flips
        to_committed = sum(1 for k in flips if collapse(s[k]) == "neutral"
                           and collapse(f[k]) != "neutral")
        to_neutral = sum(1 for k in flips if collapse(f[k]) == "neutral"
                         and collapse(s[k]) != "neutral")
        print(f"  direction: neutral->committed {to_committed}, committed->neutral {to_neutral}, "
              f"other {len(flips) - to_committed - to_neutral}")
        nn_s = sum(1 for k in keys if collapse(s[k]) != "neutral")
        nn_f = sum(1 for k in keys if collapse(f[k]) != "neutral")
        print(f"  non-neutral verdicts: {nn_s}/{len(keys)} (0-1 s) -> {nn_f}/{len(keys)} (full span)")
        # success rate against the target, per arm
        for which in ("base", "v3/LoRA"):
            ks = [k for k in keys if arm(k) == which and target_of(k)]
            if not ks:
                continue
            ok_s = sum(1 for k in ks if collapse(s[k]) == collapse(target_of(k)))
            ok_f = sum(1 for k in ks if collapse(f[k]) == collapse(target_of(k)))
            b_only = sum(1 for k in ks if collapse(s[k]) == collapse(target_of(k))
                         and collapse(f[k]) != collapse(target_of(k)))
            f_only = sum(1 for k in ks if collapse(f[k]) == collapse(target_of(k))
                         and collapse(s[k]) != collapse(target_of(k)))
            print(f"  {which:8s} target hit: {ok_s}/{len(ks)} (0-1 s) -> {ok_f}/{len(ks)} "
                  f"(full span), paired McNemar p={mcnemar(b_only, f_only):.3f}")
        # the decisive control: does base ever express sad on x02?
        for tag, d in (("0-1 s   ", s), ("full-span", f)):
            ks = [k for k in keys if "x02" in k and arm(k) == "base"]
            sad = [k for k in ks if collapse(d[k]) == "sad"]
            print(f"  base x02 judged sad, {tag}: {len(sad)}/{len(ks)} "
                  f"{[k.split('__')[1][:26] for k in sad]}")
        print("  flipped clips:")
        for k in flips:
            print(f"    {k[:54]:54s} {s[k]:9s} -> {f[k]}")
    reliability(load, collapse)


def reliability(load, collapse):
    """Is full-clip scoring merely *different*, or is it a *better* measurement?

    Reliability needs no ground truth: if scoring the whole clip makes two independent judges from
    different capability tiers agree more on the same clips, the longer window is measuring the
    intended quantity more consistently. Significance by paired bootstrap over clips.
    """
    import random
    q2s = load_many(["judge15_8fr.jsonl", "judge_basefull_q2.jsonl"])
    q2f = load("judge_full_q2.jsonl")
    q3s = load_many(["judge_all88_q3vl.jsonl", "judge_basefull_q3vl.jsonl"])
    q3f = load("judge_full_q3vl.jsonl")
    keys = sorted(set(q2s) & set(q2f) & set(q3s) & set(q3f))

    def kappa(a, b, ks):
        n = len(ks)
        labs = sorted({collapse(a[k]) for k in ks} | {collapse(b[k]) for k in ks})
        po = sum(collapse(a[k]) == collapse(b[k]) for k in ks) / n
        pe = sum((sum(collapse(a[k]) == l for k in ks) / n) *
                 (sum(collapse(b[k]) == l for k in ks) / n) for l in labs)
        return po, ((po - pe) / (1 - pe) if pe < 1 else 0.0)

    po_s, k_s = kappa(q2s, q3s, keys)
    po_f, k_f = kappa(q2f, q3f, keys)
    print(f"\n=== Inter-judge reliability, Qwen2-VL vs Qwen3-VL-32B, n={len(keys)} ===")
    print(f"  0-1 s frames : agreement {po_s*100:.1f}%  Cohen kappa={k_s:.3f}")
    print(f"  full span    : agreement {po_f*100:.1f}%  Cohen kappa={k_f:.3f}")
    d0 = k_f - k_s
    random.seed(7)
    B, le = 20000, 0
    for _ in range(B):
        bs = [random.choice(keys) for _ in keys]
        if kappa(q3f, q2f, bs)[1] - kappa(q3s, q2s, bs)[1] <= 0:
            le += 1
    print(f"  delta kappa = {d0:+.3f}; paired bootstrap P(delta<=0) = {le/B:.4f} (B={B})")
    print("  -> scoring the whole clip is not merely different: the two judges agree more,")
    print("     so the longer window is the more reliable measurement of the same quantity.")


if __name__ == "__main__":
    main()
