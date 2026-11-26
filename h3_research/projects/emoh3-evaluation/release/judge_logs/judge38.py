#!/usr/bin/env python3
"""Qwen3.8-27B judge (latest Qwen generation, native VLM, Qwen3_5ForConditionalGeneration).
Same 15-class protocol. Judges: 35 sanity + 4 v2-raw + decisive x02 pair + v3 8 tasks + base 4 tasks.
"""
from __future__ import annotations
import json, os, torch
from pathlib import Path
from PIL import Image

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
MODEL = "/gluster_hdd_cv/user/jinzili/weights/Qwen3.8-27B"
OUT = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/judge38_results.jsonl")
EMOTIONS = ["happy","joyful","sad","angry","worried","cheerful","surprised",
            "relieved","anxious","curious","neutral","disgusted","fearful","amused","other"]
REFDIR = Path("/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3/inputs/expressions_0825")
REFDIR2 = Path("/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3/inputs/expressions_0825_v2")
ROOT = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot")
DECISIVE = [
    ("v3_x02_s1234", ROOT/"frames_lora3s/x02_neutral_to_sad_lora"),
    ("base_x02_s1234", ROOT/"frames_x02ctrl/x02_neutral_to_sad_base"),
]
V3 = [("v3_x%02d_s1234"%i, ROOT/("frames_lora3s/x%02d_%s_lora"%(i,n))) for i,n in
      [(1,"neutral_to_joyful"),(2,"neutral_to_sad"),(3,"angry_to_happy"),(4,"joyful_to_worried"),
       (5,"amused_to_surprised"),(6,"anxious_to_relieved"),(7,"sad_to_cheerful"),(8,"curious_to_angry")]]
BASE = [
    ("base_x01", ROOT/"frames_base16/x01_neutral_to_joyful_base"),
    ("base_x02", ROOT/"frames_base16/x02_neutral_to_sad_base"),
    ("base_x03", ROOT/"frames_base16/x03_angry_to_happy_base"),
    ("base_x07", ROOT/"frames_base16/x07_sad_to_cheerful_base"),
]
# v2 raw known-emotion portraits (apples-to-apples with Qwen2-VL's 21-set)
V2RAW = [("v2raw_"+p.stem, p) for p in sorted(REFDIR2.glob("*_raw.png"))]
PROMPT = ("You are an anime expression evaluator. Look at the anime character's face "
          "across these video frames (in temporal order). Which emotion is the character "
          "expressing? Respond with EXACTLY ONE word from: "+", ".join(EMOTIONS)+", "
          "a colon, and a confidence score 0-100. No explanation, no reasoning. Format: WORD:SCORE")

from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

def ask(proc, model, images, prompt):
    content = [{"type":"image"}]*len(images) + [{"type":"text","text":prompt}]
    msgs = [{"role":"system","content":"You are an anime expression evaluator in a forced-choice test. You MUST output exactly one line: an emotion word from the provided list, a colon, and a score 0-100 (e.g., sad:80). Output NOTHING else: no preface, no explanation, no reasoning, no analysis. Your entire response is that single line."},
            {"role":"user","content":content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   chat_template_kwargs={"enable_thinking": False})
    inputs = proc(text=[text], images=images, return_tensors="pt")
    inputs = {k: v.to("cuda") if hasattr(v,"to") else v for k,v in inputs.items()}
    gen = model.generate(**inputs, max_new_tokens=900, do_sample=False)
    return proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

def judge(proc, model, label, d, n=8, known=None):
    p = Path(d)
    if p.is_file() and p.suffix.lower() in (".png",".jpg",".jpeg"):
        imgs = [Image.open(p).convert("RGB")]
    else:
        frames = sorted(p.glob("f*.jpg"))[:n]
        if not frames:
            rec = {"case": label, "model": "Qwen3.8-27B", "known": known, "verdict": "NO_FRAMES", "n_frames": 0}
            print(json.dumps(rec, ensure_ascii=False), flush=True); return rec
        imgs = [Image.open(f).convert("RGB") for f in frames]
    v = ask(proc, model, imgs, PROMPT)
    rec = {"case": label, "model": "Qwen3.8-27B", "known": known, "verdict": v, "n_frames": len(imgs)}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    with OUT.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    return rec

def main():
    torch.set_grad_enabled(False)
    print("loading Qwen3.8-27B ...", flush=True)
    m = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print("Qwen3.8-27B loaded", flush=True)
    for p in sorted(REFDIR.glob("*.png")):
        known = p.stem.split("_")[1] if "_" in p.stem else p.stem
        judge(proc, m, "sanity_"+p.stem, p, n=1, known=known)
    for label, d in V2RAW:
        known = d.stem.split("_")[0]
        judge(proc, m, label, d, n=1, known=known)
    for label, d in DECISIVE: judge(proc, m, label, d, n=8)
    for label, d in V3: judge(proc, m, label, d, n=8)
    for label, d in BASE: judge(proc, m, label, d, n=8)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
