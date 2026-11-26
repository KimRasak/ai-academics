#!/usr/bin/env python3
"""Inter-judge agreement over the full 88 clip-config matrix (round-22).

Turns the paper's "evaluation is the weak link" claim into statistics:
  * pairwise Cohen's kappa between every judge pair
  * Fleiss' kappa over all judges
  * per-judge label distribution (how much of the 15-class taxonomy each judge uses)
  * exact-agreement rate and unanimity rate

Inputs (raw judge logs, same 88 clip-configs, same 15-class protocol, 8 frames):
  judge15_8fr.jsonl        Qwen2-VL-7B
  judge_all88_q25.jsonl    Qwen2.5-VL-7B
  judge_all88_q3vl.jsonl   Qwen3-VL-32B
  judge_all88_q38.jsonl    Qwen3.8-27B
Verdict = last WORD:SCORE in the raw response (same rule as verify_judge_counts.py).
Usage: python3 analyze_agreement88.py
"""
from __future__ import annotations
import json, os, re, itertools
from collections import Counter

EMO = ["happy","joyful","sad","angry","worried","cheerful","surprised","relieved",
       "anxious","curious","neutral","disgusted","fearful","amused","other"]
EMOSET = set(EMO)
# Paper's synonym rule: joyful/cheerful map to their in-taxonomy synonym happy.
SYN = {"joyful": "happy", "cheerful": "happy"}

LOGDIR = os.environ.get("JUDGE_LOG_DIR", "")
if not LOGDIR:
    for c in ("judge15_logs", "release/judge_logs", os.path.dirname(os.path.abspath(__file__))):
        if os.path.exists(os.path.join(c, "judge15_8fr.jsonl")):
            LOGDIR = c
            break
    else:
        LOGDIR = "judge15_logs"

FILES = [
    ("Qwen2-VL-7B",   "judge15_8fr.jsonl"),
    ("Qwen2.5-VL-7B", "judge_all88_q25.jsonl"),
    ("Qwen3-VL-32B",  "judge_all88_q3vl.jsonl"),
    ("Qwen3.8-27B",   "judge_all88_q38.jsonl"),
]


def last_emo(v: str | None) -> str | None:
    if not v:
        return None
    hits = re.findall(r"\b(" + "|".join(EMO) + r")\b\s*:\s*(\d{1,3})", v.lower())
    if hits:
        return hits[-1][0]
    hits = re.findall(r"\b([a-z]+)\s*:\s*(\d{1,3})", v.lower())
    return hits[-1][0] if hits else None


def load(fn):
    path = os.path.join(LOGDIR, fn)
    if not os.path.exists(path):
        return None
    d = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        w = last_emo(r.get("verdict"))
        if w:
            d[r["case"]] = w
    return d


def collapse(w, syn=True):
    return SYN.get(w, w) if syn else w


def cohen_kappa(a, b):
    """a, b: equal-length label lists."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(rows):
    """rows: list of label lists (one per item, all same rater count)."""
    if not rows:
        return float("nan")
    cats = sorted({x for r in rows for x in r})
    n = len(rows[0])
    N = len(rows)
    Pi = []
    col = Counter()
    for r in rows:
        c = Counter(r)
        Pi.append((sum(v * v for v in c.values()) - n) / (n * (n - 1)))
        col.update(c)
    Pbar = sum(Pi) / N
    Pe = sum((col[c] / (N * n)) ** 2 for c in cats)
    return 1.0 if Pe == 1 else (Pbar - Pe) / (1 - Pe)


def target_of(case):
    m = re.search(r"_to_([a-z]+)_", case)
    return m.group(1) if m else None


def group_of(case):
    """Model family from the frames_* prefix (base vs each LoRA version).

    NOTE: frames_refswap_lora_* are v3 clips (the mechanism test in the paper's Finding D),
    not v1 -- an earlier version of this script mislabelled them.
    """
    p = case.split("__")[0]
    if p.startswith(("frames_base", "frames_x02ctrl", "frames_x02sweep_base", "frames_s16",
                     "frames_refswap_base")):
        return "base"
    if "lora4" in p:
        return "v4"
    if "lora3" in p or "refswap_lora" in p or "x02sweep_lora" in p or "s999" in p:
        return "v3"
    if "lora2" in p:
        return "v2"
    if p == "frames_lora" or p.startswith("frames_lora_8"):
        return "v1"
    return "other"


def pool_kind(case):
    """Which sub-pool a config belongs to -- the 88 configs are NOT homogeneous."""
    p = case.split("__")[0]
    if "refswap" in p:
        return "mechanism (reference-swap probe)"
    if "x02sweep" in p or "x02ctrl" in p or "s999" in p:
        return "x02 seed sweep"
    if p.endswith("8s") or p.endswith("_8"):
        return "non-16-step / 4-frame"
    return "transfer task at 16 steps"


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact p for [[a,b],[c,d]] (pure Python)."""
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def p_of(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p0 = p_of(a)
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= p0 + 1e-12))


def success_by_judge(avail, cases):
    """Success = judged emotion equals target under the paper's synonym rule.

    The 88 configs are a HETEROGENEOUS pool (transfer tasks at 16 steps, reference-swap
    mechanism probes, an x02 seed sweep, and non-16-step/4-frame rows), and base covers only
    4 of the 8 transfer tasks at 16 steps. Aggregate base-vs-v3 percentages over this pool are
    therefore NOT task-matched and must not be read as an effect size; the matched comparison
    below is the defensible one.
    """
    print("\n=== pool composition (the 88 configs are not homogeneous) ===")
    kinds = Counter(pool_kind(c) for c in cases)
    for k, v in kinds.most_common():
        print(f"  {k:34s} {v}")
    groups = sorted({group_of(c) for c in cases})
    print("\n=== unmatched aggregate over the whole pool (context only, NOT an effect size) ===")
    hdr = "  {:14s}".format("judge") + "".join(f"{g:>8s}" for g in groups) + f"{'total':>9s}"
    print(hdr)
    for n, d in avail:
        cells, tot, tn = [], 0, 0
        for g in groups:
            gc = [c for c in cases if group_of(c) == g]
            gc = [c for c in gc if c in d]
            ok = sum(1 for c in gc if collapse(d[c], True) == collapse(target_of(c) or "", True))
            cells.append(f"{ok}/{len(gc)}")
            tot += ok
            tn += len(gc)
        print("  {:14s}".format(n) + "".join(f"{c:>8s}" for c in cells) + f"{f'{tot}/{tn}':>9s}")

    print("\n=== MATCHED comparison (primary): tasks where a base 16-step clip exists ===")
    base16 = {}
    v3 = {"s42": {}, "s1234": {}}
    for c in cases:
        m = re.search(r"(x\d\d)", c)
        if not m:
            continue
        task = m.group(1)
        p = c.split("__")[0]
        if p in ("frames_base16", "frames_base16b"):
            base16.setdefault(task, c)
        elif p == "frames_lora3":
            v3["s42"][task] = c
        elif p == "frames_lora3s":
            v3["s1234"][task] = c
    print("  each judge scored on the tasks IT covers (Qwen3.8-27B loses rows to unparseable output)")
    for n, d in avail:
        tasks = sorted(t for t in base16 if t in v3["s42"] and t in v3["s1234"]
                       and base16[t] in d and v3["s42"][t] in d and v3["s1234"][t] in d)
        if not tasks:
            continue
        b = sum(1 for t in tasks if collapse(d[base16[t]], True) == collapse(target_of(base16[t]), True))
        out = []
        for sd in ("s42", "s1234"):
            v = sum(1 for t in tasks if collapse(d[v3[sd][t]], True) == collapse(target_of(v3[sd][t]), True))
            p = fisher_exact_2x2(v, len(tasks) - v, b, len(tasks) - b)
            out.append(f"v3 {sd}: {v}/{len(tasks)} (p={p:.3f})")
        verdict = "gain" if any(int(o.split(": ")[1].split("/")[0]) > b for o in out) else "tie/none"
        print(f"  {n:14s} [{','.join(tasks)}] base {b}/{len(tasks)}   " + "   ".join(out) + f"  -> {verdict}")
    print("  Honest reading: the original judge sees a gain; the newer judges see a tie because")
    print("  they already credit the base. No judge shows base > v3 on the matched set, so the")
    print("  correct claim is 'the gain is not reproducible under stronger judges', NOT a sign flip.")


def nonneutral_agreement(avail, cases):
    """Raw agreement is inflated by the shared 'neutral' default; two ways to strip it.

    'either' drops only the configs where BOTH judges said neutral -> how much of the raw
    agreement was the shared default. 'both' keeps only configs where both committed to a
    non-neutral emotion -> when they do commit, do they pick the same word?
    """
    print("\n=== agreement with the shared 'neutral' default stripped ===")
    for (n1, d1), (n2, d2) in itertools.combinations(avail, 2):
        e = [(collapse(d1[c], True), collapse(d2[c], True)) for c in cases
             if not (collapse(d1[c], True) == "neutral" and collapse(d2[c], True) == "neutral")]
        b = [(x, y) for x, y in e if x != "neutral" and y != "neutral"]
        pe = sum(1 for x, y in e if x == y) / len(e) if e else float("nan")
        pb = sum(1 for x, y in b if x == y) / len(b) if b else float("nan")
        print(f"  {n1:14s} vs {n2:14s}  either-committed n={len(e):3d} agreement={pe:.1%}"
              f"   |  both-committed n={len(b):3d} agreement={pb:.1%}")
    print("  Reading: the raw 42-72% figures are inflated by the shared neutral default. When only")
    print("  one judge commits they almost always disagree, but when BOTH commit to a non-neutral")
    print("  emotion they nearly always pick the same word -- the disagreement is about WHETHER an")
    print("  emotion is present, not which one.")


VARIANTS = [
    ("Qwen2-VL-7B",  "judge15_8fr.jsonl",       "judge_all88_q2_rev.jsonl"),
    ("Qwen3-VL-32B", "judge_all88_q3vl.jsonl",  "judge_all88_q3vl_rev.jsonl"),
]


def within_judge_stability():
    """Same judge, same frames, same greedy decoding; only the option-list ORDER differs.

    Separates systematic judge bias from instrument instability: if a judge's verdicts are
    stable under option reordering, the between-judge disagreement is bias, not noise.
    """
    rows = []
    for name, base_fn, rev_fn in VARIANTS:
        a, b = load(base_fn), load(rev_fn)
        if not a or not b:
            continue
        common = sorted(set(a) & set(b))
        if not common:
            continue
        la = [collapse(a[c], True) for c in common]
        lb = [collapse(b[c], True) for c in common]
        po = sum(1 for x, y in zip(la, lb) if x == y) / len(common)
        flips = [(c, a[c], b[c]) for c in common if collapse(a[c], True) != collapse(b[c], True)]
        rows.append((name, len(common), po, cohen_kappa(la, lb), flips))
    if not rows:
        return
    print("\n=== within-judge stability under option reordering (control) ===")
    print("  same judge, same frames, greedy; only the 15-class list order is reversed")
    for name, n, po, k, flips in rows:
        print(f"  {name:14s} n={n:3d} self-agreement={po:.1%} kappa={k:+.3f} ({len(flips)} flips)")
        for c, x, y in flips[:5]:
            print(f"      flip: {c[:56]:56s} {x} -> {y}")
    print("  Reading: high self-agreement => between-judge disagreement is systematic bias;")
    print("           low self-agreement  => the instrument is unstable against itself.")


def main():
    data = [(name, load(fn)) for name, fn in FILES]
    missing = [n for n, d in data if d is None]
    avail = [(n, d) for n, d in data if d]
    print(f"log dir: {LOGDIR}")
    for n, d in avail:
        print(f"  {n:14s} {len(d)} parsed clip-configs")
    if missing:
        print(f"  MISSING (sweep not finished): {', '.join(missing)}")
    if len(avail) < 2:
        print("need >=2 judges")
        return

    common = sorted(set.intersection(*[set(d) for _, d in avail]))
    print(f"\ncommon clip-configs judged by all {len(avail)} judges: {len(common)}")

    for syn in (False, True):
        tag = "synonym-collapsed (joyful/cheerful->happy)" if syn else "raw 15-class"
        print(f"\n=== agreement, {tag} ===")
        # pairwise on each PAIR's own common set (uses all available data per pair)
        print("  pairwise Cohen's kappa (each pair on its own common set):")
        for (n1, d1), (n2, d2) in itertools.combinations(avail, 2):
            cs = sorted(set(d1) & set(d2))
            a = [collapse(d1[c], syn) for c in cs]
            b = [collapse(d2[c], syn) for c in cs]
            po = sum(1 for x, y in zip(a, b) if x == y) / len(cs)
            print(f"    {n1:14s} vs {n2:14s} n={len(cs):3d} kappa={cohen_kappa(a,b):+.3f}"
                  f"  raw agreement={po:.1%}")
        # Fleiss on the full intersection only (requires all raters per item)
        cols = {n: [collapse(d[c], syn) for c in common] for n, d in avail}
        rows = [[cols[n][i] for n, _ in avail] for i in range(len(common))]
        print(f"  Fleiss' kappa ({len(avail)} judges, n={len(common)}): {fleiss_kappa(rows):+.3f}")
        unan = sum(1 for r in rows if len(set(r)) == 1)
        print(f"  unanimous clip-configs: {unan}/{len(common)} ({unan/len(common):.1%})")
        maj = sum(1 for r in rows if max(Counter(r).values()) >= 3)
        print(f"  >=3/4 majority:        {maj}/{len(common)} ({maj/len(common):.1%})")

    print("\n=== per-judge label usage (raw 15-class, common set) ===")
    for n, d in avail:
        c = Counter(d[x] for x in common)
        used = len(c)
        top = ", ".join(f"{k}:{v}" for k, v in c.most_common(6))
        print(f"  {n:14s} uses {used}/15 classes | {top}")

    allcases = sorted(set().union(*[set(d) for _, d in avail]))
    success_by_judge(avail, allcases)
    nonneutral_agreement(avail, common)

    within_judge_stability()

    print("\n=== negative-emotion emissions (sad/angry/worried/anxious/disgusted/fearful) ===")
    NEG = {"sad", "angry", "worried", "anxious", "disgusted", "fearful"}
    for n, d in avail:
        c = Counter(d[x] for x in common)
        tot = sum(v for k, v in c.items() if k in NEG)
        detail = ", ".join(f"{k}:{c[k]}" for k in NEG if c.get(k))
        print(f"  {n:14s} {tot}/{len(common)} clip-configs ({detail or 'none'})")


if __name__ == "__main__":
    main()
