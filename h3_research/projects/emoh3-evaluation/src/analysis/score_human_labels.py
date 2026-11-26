#!/usr/bin/env python3
"""Score human labels against the four VLM judges (round-23).

Consumes one or more results/human-label/labels_*.json produced by the blind-label tool
plus the raw judge logs, and answers the questions the reviewers said were blocking:

  1. inter-annotator agreement (if >=2 annotators)
  2. annotator accuracy on the known-emotion portraits (are the humans reliable?)
  3. each judge's agreement with the human consensus (which judge, if any, tracks humans?)
  4. the decisive x02 pair under human labels
  5. the base-vs-v3 effect re-scored with HUMAN labels (does the +7.0pp / -8.1pp sign
     flip resolve one way or the other?)

Run from the project root: python3 src/analysis/score_human_labels.py
"""
from __future__ import annotations
import json, re, glob, os, itertools
from collections import Counter
from pathlib import Path

EMO = ["happy","joyful","sad","angry","worried","cheerful","surprised","relieved",
       "anxious","curious","neutral","disgusted","fearful","amused","other"]
SYN = {"joyful": "happy", "cheerful": "happy"}
NEG = {"sad","angry","worried","anxious","disgusted","fearful"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HL = str(PROJECT_ROOT / "results" / "human-label")
LOGDIR = str(PROJECT_ROOT / "results" / "judge15-logs")
JUDGES = [("Qwen2-VL-7B","judge15_8fr.jsonl"), ("Qwen2.5-VL-7B","judge_all88_q25.jsonl"),
          ("Qwen3-VL-32B","judge_all88_q3vl.jsonl"), ("Qwen3.8-27B","judge_all88_q38.jsonl")]
PORTRAIT_JUDGE = [("Qwen2-VL-7B","judge15_35.jsonl"), ("Qwen2.5-VL-7B","judge25_results.jsonl"),
                  ("Qwen3-VL-32B","judge3vl_results.jsonl"), ("Qwen3.8-27B","judge38_results.jsonl")]

c = lambda w: SYN.get(w, w) if w else w


def last_emo(v):
    if not v:
        return None
    h = re.findall(r"\b(" + "|".join(EMO) + r")\b\s*:\s*(\d{1,3})", v.lower())
    if h:
        return h[-1][0]
    h = re.findall(r"\b([a-z]+)\s*:\s*(\d{1,3})", v.lower())
    return h[-1][0] if h else None


def load_judge(fn, prefix=""):
    p = os.path.join(LOGDIR, fn)
    if not os.path.exists(p):
        return {}
    d = {}
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        w = last_emo(r.get("verdict"))
        if w:
            key = r["case"]
            if prefix and key.startswith(prefix):
                key = key[len(prefix):]
            d[key] = w
    return d


def cohen(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] / n * cb[k] / n for k in set(ca) | set(cb))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def target_of(case):
    m = re.search(r"_to_([a-z]+)_", case)
    return c(m.group(1)) if m else None


def group_of(case):
    p = case.split("__")[0]
    if p.startswith(("frames_base", "frames_x02ctrl", "frames_x02sweep_base", "frames_s16",
                     "frames_refswap_base")):
        return "base"
    if "lora3" in p or "x02sweep_lora" in p or "s999" in p:
        return "v3"
    return "other"


def main():
    MAP = json.load(open(os.path.join(HL, "MAP.json")))
    files = sorted(glob.glob(os.path.join(HL, "labels_*.json")))
    if not files:
        print("no annotator files yet: expected results/human-label/labels_<name>.json")
        print("(open results/human-label/index.html, label, download the result there)")
        return
    ann = {}
    for f in files:
        d = json.load(open(f))
        name = d.get("annotator") or os.path.basename(f)[7:-5]
        ann[name] = {k: v["emotion"] for k, v in d["labels"].items()}
        print(f"annotator {name}: {len(ann[name])} labels  [{os.path.basename(f)}]")

    names = sorted(ann)
    both = sorted(set.intersection(*[set(a) for a in ann.values()])) if len(ann) > 1 else sorted(ann[names[0]])

    if len(ann) > 1:
        print("\n=== inter-annotator agreement ===")
        for n1, n2 in itertools.combinations(names, 2):
            a = [c(ann[n1][k]) for k in both]
            b = [c(ann[n2][k]) for k in both]
            po = sum(1 for x, y in zip(a, b) if x == y) / len(both)
            print(f"  {n1} vs {n2}: kappa={cohen(a,b):+.3f} agreement={po:.1%} (n={len(both)})")

    # human consensus (majority; ties -> first annotator)
    cons = {}
    for k in both:
        cnt = Counter(c(ann[n][k]) for n in names if k in ann[n])
        cons[k] = cnt.most_common(1)[0][0]

    # 2. annotator reliability on known-emotion portraits
    print("\n=== annotators vs intended portrait labels (reliability check) ===")
    port = {i: v for i, v in MAP.items() if v["kind"] == "portrait"}
    def intended(src):
        base = src.split("/")[-1].replace("_raw.png", "").replace(".png", "")
        parts = base.split("_")
        for p in parts:
            if p in EMO:
                return c(p)
        return None
    pit = {i: intended(v["source"]) for i, v in port.items()}
    pit = {i: t for i, t in pit.items() if t}
    for n in names:
        ok = sum(1 for i, t in pit.items() if i in ann[n] and c(ann[n][i]) == t)
        tot = sum(1 for i in pit if i in ann[n])
        neg = [(i, t) for i, t in pit.items() if t in NEG and i in ann[n]]
        okneg = sum(1 for i, t in neg if c(ann[n][i]) == t)
        print(f"  {n}: {ok}/{tot} correct ({ok/max(tot,1):.0%}); negative emotions {okneg}/{len(neg)}")

    # 3. judge vs human on the same portraits
    print("\n=== judge vs human consensus, portraits (same items) ===")
    for jn, jf in PORTRAIT_JUDGE:
        jd = load_judge(jf, prefix="sanity_")
        pairs = []
        for i, v in port.items():
            src = v["source"].split("/")[-1]
            key = src.replace(".png", "")
            if key in jd and i in cons:
                pairs.append((c(jd[key]), cons[i]))
        if pairs:
            a = [x for x, _ in pairs]
            b = [y for _, y in pairs]
            po = sum(1 for x, y in zip(a, b) if x == y) / len(pairs)
            print(f"  {jn:14s} n={len(pairs):3d} agreement with humans={po:.1%} kappa={cohen(a,b):+.3f}")

    # 4/5. clips
    clip = {i: v["source"] for i, v in MAP.items() if v["kind"] == "clip"}
    print("\n=== judge vs human consensus, video clips (same items) ===")
    for jn, jf in JUDGES:
        jd = load_judge(jf)
        pairs = [(c(jd[s]), cons[i]) for i, s in clip.items() if s in jd and i in cons]
        if pairs:
            a = [x for x, _ in pairs]
            b = [y for _, y in pairs]
            po = sum(1 for x, y in zip(a, b) if x == y) / len(pairs)
            print(f"  {jn:14s} n={len(pairs):3d} agreement with humans={po:.1%} kappa={cohen(a,b):+.3f}")

    print("\n=== decisive x02 pair under human labels ===")
    for i, s in sorted(clip.items(), key=lambda kv: kv[1]):
        if "x02" in s and ("lora3s" in s or "base16__" in s or "x02ctrl" in s):
            who = ", ".join(f"{n}:{ann[n].get(i,'-')}" for n in names)
            print(f"  {s:52s} human={cons.get(i,'-'):10s} ({who})")

    print("\n=== base vs v3 effect, scored by HUMAN labels ===")
    rows = {"base": [], "v3": []}
    for i, s in clip.items():
        g = group_of(s)
        if g in rows and i in cons:
            rows[g].append(c(cons[i]) == target_of(s))
    for g in ("base", "v3"):
        n = len(rows[g])
        k = sum(rows[g])
        print(f"  {g:5s} {k}/{n}" + (f" = {k/n:.1%}" if n else ""))
    if rows["base"] and rows["v3"]:
        rb = sum(rows["base"]) / len(rows["base"])
        rv = sum(rows["v3"]) / len(rows["v3"])
        print(f"  delta = {100*(rv-rb):+.1f}pp -> " +
              ("v3 > base (original judge's direction)" if rv > rb else
               "base > v3 (stronger judges' direction)" if rb > rv else "tie"))
        print("  NOTE: compare with judge-scored deltas: Qwen2-VL +7.0pp, Qwen2.5-VL -2.9pp, "
              "Qwen3-VL-32B -8.1pp (full 88-config matrix).")


if __name__ == "__main__":
    main()
