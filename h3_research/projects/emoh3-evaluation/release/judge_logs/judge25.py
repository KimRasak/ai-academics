#!/usr/bin/env python3
"""Second, independent VLM judge: Qwen2.5-VL-7B-Instruct.
Judges (a) the decisive x02 s1234 pair (v3 + base, 8 frames each) and
(b) the 21 known-emotion sanity portraits, under the same 15-class protocol.
"""
from __future__ import annotations
import json, os, torch
from pathlib import Path
from PIL import Image

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
MODEL = "/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct"
OUT = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/judge25_results.jsonl")
EMOTIONS = ["happy","joyful","sad","angry","worried","cheerful","surprised",
            "relieved","anxious","curious","neutral","disgusted","fearful","amused","other"]
REFDIR = Path("/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3/inputs/expressions_0825")
DECISIVE = [
    ("v3_x02_s1234", "/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_lora3s/x02_neutral_to_sad_lora"),
    ("base_x02_s1234", "/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_x02ctrl/x02_neutral_to_sad_base"),
]

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def ask(proc, model, images, prompt):
    content = [{"type":"image"}]*len(images) + [{"type":"text","text":prompt}]
    msgs = [{"role":"user","content":content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=images, return_tensors="pt")
    inputs = {k: v.to("cuda") if hasattr(v,"to") else v for k,v in inputs.items()}
    gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    return proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

def main():
    torch.set_grad_enabled(False)
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                           device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print("model loaded", flush=True)
    prompt = ("You are an anime expression evaluator. Look at the anime character's face "
              "across these video frames (in temporal order). Which emotion is the character "
              "expressing? Answer with exactly one word from: "+", ".join(EMOTIONS)+". "
              "Then a colon, then a confidence score 0-100. Format: WORD:SCORE")
    # decisive pair
    for label, d in DECISIVE:
        frames = sorted(Path(d).glob("f*.jpg"))[:8]
        imgs = [Image.open(f).convert("RGB") for f in frames]
        v = ask(proc, m, imgs, prompt)
        rec = {"case": label, "model": "Qwen2.5-VL-7B", "verdict": v, "n_frames": len(imgs)}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    # sanity portraits
    for p in sorted(REFDIR.glob("*.png")):
        known = p.stem.split("_")[1] if "_" in p.stem else p.stem
        im = Image.open(p).convert("RGB")
        v = ask(proc, m, [im], prompt)
        rec = {"case": "sanity_"+p.stem, "known": known, "verdict": v, "n_frames": 1}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main()
