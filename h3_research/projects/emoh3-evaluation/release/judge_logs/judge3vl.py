#!/usr/bin/env python3
"""Immediate strong judge: Qwen3-VL-32B-Instruct (latest-gen open Qwen VLM, on disk now).
Same 15-class protocol. Judges: 35 sanity, decisive x02 pair, v3 8 tasks, base 4 tasks.
"""
from __future__ import annotations
import json, os, torch
from pathlib import Path
from PIL import Image

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
MODEL = "/gluster_hdd_cv/user/jinzili/weights/Qwen3-VL-32B-Instruct"
OUT = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/judge3vl_results.jsonl")
EMOTIONS = ["happy","joyful","sad","angry","worried","cheerful","surprised",
            "relieved","anxious","curious","neutral","disgusted","fearful","amused","other"]
REFDIR = Path("/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3/inputs/expressions_0825")
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
PROMPT = ("You are an anime expression evaluator. Look at the anime character's face "
          "across these video frames (in temporal order). Which emotion is the character "
          "expressing? Answer with exactly one word from: "+", ".join(EMOTIONS)+". "
          "Then a colon, then a confidence score 0-100. Format: WORD:SCORE")

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def ask(proc, model, images, prompt):
    content = [{"type":"image"}]*len(images) + [{"type":"text","text":prompt}]
    msgs = [{"role":"user","content":content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=images, return_tensors="pt")
    inputs = {k: v.to("cuda") if hasattr(v,"to") else v for k,v in inputs.items()}
    gen = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    return proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

def judge(proc, model, label, d, n=8, known=None):
    p = Path(d)
    if p.is_file() and p.suffix.lower() in (".png",".jpg",".jpeg"):
        imgs = [Image.open(p).convert("RGB")]
    else:
        frames = sorted(p.glob("f*.jpg"))[:n]
        if not frames:
            rec = {"case": label, "model": "Qwen3-VL-32B", "known": known, "verdict": "NO_FRAMES", "n_frames": 0}
            print(json.dumps(rec, ensure_ascii=False), flush=True); return rec
        imgs = [Image.open(f).convert("RGB") for f in frames]
    v = ask(proc, model, imgs, PROMPT)
    rec = {"case": label, "model": "Qwen3-VL-32B", "known": known, "verdict": v, "n_frames": len(imgs)}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    with OUT.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    return rec

def main():
    torch.set_grad_enabled(False)
    print("loading Qwen3-VL-32B ...", flush=True)
    m = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print("Qwen3-VL-32B loaded", flush=True)
    for p in sorted(REFDIR.glob("*.png")):
        known = p.stem.split("_")[1] if "_" in p.stem else p.stem
        judge(proc, m, "sanity_"+p.stem, p, n=1, known=known)
    for label, d in DECISIVE: judge(proc, m, label, d, n=8)
    for label, d in V3: judge(proc, m, label, d, n=8)
    for label, d in BASE: judge(proc, m, label, d, n=8)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
