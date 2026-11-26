#!/usr/bin/env python3
"""Qwen2-VL-7B on ALL 35 expressions_0825 portraits (same 15-class protocol as judge15.py)."""
from __future__ import annotations
import json, os, torch
from pathlib import Path
from PIL import Image
os.environ.setdefault("CUDA_VISIBLE_DEVICES","7")
MODEL = "/gluster_hdd_cv/user/jinzili/weights/reward_models/Qwen2-VL-7B-Instruct"
OUT = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/judge15_35.jsonl")
REFDIR = Path("/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3/inputs/expressions_0825")
EMOTIONS = ["happy", "joyful", "sad", "angry", "worried", "cheerful", "surprised",
            "relieved", "anxious", "curious", "neutral", "disgusted", "fearful", "amused", "other"]
PROMPT = ("You are an anime expression evaluator. Look at the anime character's face. "
          "Which emotion is the character expressing? Answer with exactly one word from: "
          + ", ".join(EMOTIONS) + ". Then a colon, then a confidence score 0-100. Format: WORD:SCORE")
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
def main():
    torch.set_grad_enabled(False)
    m = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    for p in sorted(REFDIR.glob("*.png")):
        known = p.stem.split("_")[1] if "_" in p.stem else p.stem
        imgs=[Image.open(p).convert("RGB")]
        content=[{"type":"image"}]*len(imgs)+[{"type":"text","text":PROMPT}]
        msgs=[{"role":"user","content":content}]
        text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs=proc(text=[text], images=imgs, return_tensors="pt")
        inputs={k: v.to("cuda") if hasattr(v,"to") else v for k,v in inputs.items()}
        gen=m.generate(**inputs, max_new_tokens=32, do_sample=False)
        v=proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        rec={"case":"sanity_"+p.stem,"model":"Qwen2-VL-7B","known":known,"verdict":v,"n_frames":1}
        print(json.dumps(rec,ensure_ascii=False),flush=True)
        with OUT.open("a") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")
    print("DONE",flush=True)
if __name__=="__main__": main()
