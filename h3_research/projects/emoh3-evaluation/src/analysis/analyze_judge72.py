#!/usr/bin/env python3
"""Analyze judge72_results.jsonl: sanity accuracy + per-case verdicts."""
import json, re, sys
from pathlib import Path

def emo(v):
    m = re.match(r"([A-Za-z]+):", (v or "").strip())
    return m.group(1).lower() if m else None

def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    sanity = [r for r in rows if r.get("case","").startswith("sanity_")]
    strict = lenient = 0
    for r in sanity:
        k = r.get("known",""); v = emo(r.get("verdict"))
        if not k: continue
        if v == k: strict += 1
        if v == k or (k in ("joyful","excited") and v=="happy"): lenient += 1
    print(f"=== 72B SANITY: total={len(sanity)} strict={strict} lenient={lenient} ===")
    for r in sanity:
        k = r.get("known","")
        print(f"  {r.get('case'):60s} known={k:12s} -> {r.get('verdict')}")
    print("\n=== NON-SANITY ===")
    for r in rows:
        if not r.get("case","").startswith("sanity_"):
            print(f"  {r.get('case'):30s} -> {r.get('verdict')}")
    # confusion: known -> judged
    print("\n=== known -> judged (all sanity) ===")
    agg = {}
    for r in sanity:
        k = r.get("known",""); v = emo(r.get("verdict")) or "?"
        agg.setdefault(k, []).append(v)
    for k in sorted(agg): print(f"  {k:12s}: {agg[k]}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "judge72_results.jsonl")
