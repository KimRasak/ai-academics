#!/usr/bin/env python3
"""Comprehensive 4-judge comparison for the paper (strong-accept round).
Judges: Qwen2-VL-7B (paper), Qwen2.5-VL-7B, Qwen3-VL-32B, Qwen3.8-27B.
Produces: (1) taxonomy-relevant sanity accuracy per judge, (2) decisive x02 verdict per judge,
(3) v3-task verdict matrix, (4) per-case table.
"""
import json, re, os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGDIR = PROJECT_ROOT / "results" / "judge15-logs"
EMO = {"happy","joyful","sad","angry","worried","cheerful","surprised","relieved",
       "anxious","curious","neutral","disgusted","fearful","amused","other"}
def last_emo(v):
    if not v: return None
    hits = re.findall(r"\b(" + "|".join(EMO) + r")\b\s*:\s*(\d{1,3})", v.lower())
    if hits: return f"{hits[-1][0]}:{hits[-1][1]}"
    hits = re.findall(r"\b([a-z]+)\s*:\s*(\d{1,3})", v.lower())
    if hits: return f"{hits[-1][0]}:{hits[-1][1]}"
    return None
def word(p): return p.split(":")[0] if p else None

# ---- load Qwen2-VL (sanity15, 21) ----
q2={}
for l in open(LOGDIR / "judge_sanity15.jsonl"):
    r=json.loads(l); b=os.path.basename(r["ref"]); base=b[:b.rfind(".")]
    q2[base]={"known":r["known"],"v":last_emo(r["verdict"])}
# ---- load jsonl-based judges ----
def load35(path, strip):
    d={}
    for l in open(path):
        r=json.loads(l); c=r.get("case","")
        if c.startswith(strip): d[c[len(strip):]]={"known":r.get("known"),"v":last_emo(r.get("verdict"))}
    return d
q25=load35(LOGDIR / "judge25_results.jsonl","sanity_")
q32=load35(LOGDIR / "judge3vl_results.jsonl","sanity_")
q38=load35(LOGDIR / "judge38_results.jsonl","sanity_")
# v2 raw rows for q25/q32 (need to add when run); q38 has v2raw_ prefix
q38_raw={}
for l in open(LOGDIR / "judge38_results.jsonl"):
    r=json.loads(l); c=r.get("case","")
    if c.startswith("v2raw_"): q38_raw[c[7:]]={"known":r.get("known"),"v":last_emo(r.get("verdict"))}

def taxacc(d, keys=None):
    keys = keys or d.keys()
    ok=tot=nop=0
    for c in keys:
        if d[c]["known"] not in EMO: continue
        tot+=1
        w=word(d[c]["v"])
        if w is None: nop+=1
        elif w==d[c]["known"]: ok+=1
    return ok,tot,nop

print("=== SANITY: taxonomy-relevant strict accuracy ===")
print(f"  Qwen2-VL-7B   : {taxacc(q2)}   (21-set)")
print(f"  Qwen2.5-VL-7B : {taxacc(q25)}  (35-set)")
print(f"  Qwen3-VL-32B  : {taxacc(q32)}  (35-set)")
print(f"  Qwen3.8-27B   : {taxacc(q38)}  (35-set)")

print("\n=== SANITY: lenient (joyful/excited/cheerful->happy) ===")
def taxacc_len(d):
    ok=tot=0
    for c,k in d.items():
        if k["known"] not in EMO: continue
        tot+=1; w=word(k["v"])
        if w==k["known"] or (k["known"] in ("joyful","excited","cheerful") and w=="happy"): ok+=1
    return ok,tot
for nm,d in [("Qwen2-VL-7B",q2),("Qwen2.5-VL-7B",q25),("Qwen3-VL-32B",q32),("Qwen3.8-27B",q38)]:
    print(f"  {nm}: {taxacc_len(d)}")

# ---- decisive x02 + tasks (multi-frame clips) ----
def loadcases(path):
    d={}
    for l in open(path):
        r=json.loads(l); d[r["case"]]=last_emo(r.get("verdict"))
    return d
c25=loadcases(LOGDIR / "judge25_results.jsonl")
c32=loadcases(LOGDIR / "judge3vl_results.jsonl")
c38=loadcases(LOGDIR / "judge38_results.jsonl")
c2={}
for l in open(LOGDIR / "judge_sanity15.jsonl"):  # not used for clips; placeholder
    pass

print("\n=== DECISIVE x02 (v3 vs base, same seed s1234) ===")
for case in ["v3_x02_s1234","base_x02_s1234"]:
    print(f"  {case}: Qwen2.5-VL={c25.get(case)} Qwen3-VL-32B={c32.get(case)} Qwen3.8-27B={c38.get(case)}")
# note: Qwen2-VL decisive was run separately (judge15_8fr): sad:80 vs neutral:50 (montage)

print("\n=== V3 TASKS (8) ===")
v3targets={"v3_x01_s1234":"joyful","v3_x02_s1234":"sad","v3_x03_s1234":"happy",
           "v3_x04_s1234":"worried","v3_x05_s1234":"surprised","v3_x06_s1234":"relieved",
           "v3_x07_s1234":"cheerful","v3_x08_s1234":"angry"}
for case in sorted(v3targets):
    t=v3targets[case]
    print(f"  {case} target={t:10s} Qwen2.5-VL={str(c25.get(case)):10s} Qwen3-VL-32B={str(c32.get(case)):10s} Qwen3.8-27B={str(c38.get(case))}")

print("\n=== V2-RAW known portraits (competence for sad/surprised) ===")
print("  (Qwen2-VL: sad->sad, surprised->surprised [correct]; 32B/2.5-VL TBD; Qwen3.8 below)")
for case in sorted(q38_raw):
    print(f"  {case}: Qwen3.8-27B={q38_raw[case]['v']} (known={q38_raw[case]['known']})")
