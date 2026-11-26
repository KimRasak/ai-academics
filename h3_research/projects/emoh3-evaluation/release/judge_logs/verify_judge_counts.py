#!/usr/bin/env python3
"""Reproducibility verification: derive every headline judge number from raw logs.
Parsing method: take the LAST '<word>:<score>' in each raw verdict (the model settles on its
final answer at the end; earlier mentions in its preamble are reasoning, not the answer).
Strict = exact 15-class match of known label; lenient = paper's synonym rule (joyful/cheerful/happy one family).
Usage: python3 verify_judge_counts.py
"""
import json, re, os
EMO={"happy","joyful","sad","angry","worried","cheerful","surprised","relieved",
     "anxious","curious","neutral","disgusted","fearful","amused","other"}
# Auto-detect log directory: workspace (judge15_logs/) OR release (release/judge_logs/) OR script dir.
import os as _os
LOG = os.environ.get("JUDGE_LOG_DIR", "")
if not LOG:
    cands = ["judge15_logs", "release/judge_logs", _os.path.dirname(_os.path.abspath(__file__))]
    LOG = next((c for c in cands if _os.path.isdir(c) and _os.path.exists(_os.path.join(c, "judge38_results.jsonl"))), "judge15_logs")
def last_emo(v):
    if not v: return None
    hits=re.findall(r"\b(" + "|".join(EMO) + r")\b\s*:\s*(\d{1,3})", v.lower())
    if hits: return hits[-1][0]
    hits=re.findall(r"\b([a-z]+)\s*:\s*(\d{1,3})", v.lower())
    return hits[-1][0] if hits else None

def load(path, prefix):
    d={}
    for l in open(path):
        r=json.loads(l); c=r.get("case","")
        if c.startswith(prefix):
            key=c[len(prefix):] if prefix else c
            d[key]={"known":r.get("known"),"v":last_emo(r.get("verdict"))}
    return d

# judges: (name, dict, source)
judges=[
 ("Qwen2-VL-7B",   load(f"{LOG}/judge15_35.jsonl","sanity_"), "judge15_35.jsonl (full 35)"),
 ("Qwen2.5-VL-7B", load(f"{LOG}/judge25_results.jsonl","sanity_"), "judge25_results.jsonl"),
 ("Qwen3-VL-32B",  load(f"{LOG}/judge3vl_results.jsonl","sanity_"), "judge3vl_results.jsonl"),
 ("Qwen3.8-27B",   load(f"{LOG}/judge38_results.jsonl","sanity_"), "judge38_results.jsonl"),
]
# sanity cases from the 35-set (union of all judges' keys)
allkeys=set()
for _,d,_ in judges: allkeys.update(d.keys())
tax17=sorted(k for k in allkeys if next((d[k]["known"] for _,d,_ in judges if k in d), None) in EMO)
print(f"Taxonomy-relevant sanity set ({len(tax17)} portraits):")
for k in tax17: print(f"   {k}")
print()
print("=== SANITY: strict / lenient / parsed-of-17 (per judge) ===")
for name,d,src in judges:
    strict=len=parsed=neg=0
    HAPPY={"joyful","cheerful","happy"}; NEG={"sad","angry","worried","anxious","disgusted","fearful"}
    for k in tax17:
        if k not in d: continue
        w=d[k]["v"]
        if w is None: continue
        parsed+=1
        kn=d[k]["known"]
        if w==kn: strict+=1
        if w==kn or (kn in HAPPY and w in HAPPY): len+=1
        if kn in NEG and w==kn: neg+=1
    print(f"  {name:14s} strict={strict}/17 lenient={len}/17 parsed={parsed}/17 neg-emotion={neg}/5   [{src}]")

print("\n=== DECISIVE x02 (v3 vs base, same seed) ===")
def loadclip(path):
    d={}
    for l in open(path):
        r=json.loads(l); d[r["case"]]=last_emo(r.get("verdict"))
    return d
c25=loadclip(f"{LOG}/judge25_results.jsonl")
c32=loadclip(f"{LOG}/judge3vl_results.jsonl")
c38=loadclip(f"{LOG}/judge38_results.jsonl")
c15=loadclip(f"{LOG}/judge15_8fr.jsonl")  # Qwen2-VL decisive (8-frame)
# Qwen2-VL decisive cell: v3 = frames_lora3s x02 (sad:80); base = frames_base16 x02 (neutral)
v3q2=c15.get('frames_lora3s__x02_neutral_to_sad_lora')
bq2=c15.get('frames_base16__x02_neutral_to_sad_base')
v3k='v3_x02_s1234'; bk='base_x02_s1234'
print(f"  v3 x02:  Qwen2-VL={v3q2}  Qwen2.5-VL={c25.get(v3k)}  Qwen3-VL-32B={c32.get(v3k)}  Qwen3.8-27B={c38.get(v3k)}")
print(f"  base x02: Qwen2-VL={bq2}  Qwen2.5-VL={c25.get(bk)}  Qwen3-VL-32B={c32.get(bk)}  Qwen3.8-27B={c38.get(bk)}")
