#!/usr/bin/env python3
"""Identity-consistency judge: is the generated character the SAME anime character as the reference?

For each case, shows Qwen2-VL-7B the reference portrait + one generated frame and asks for a
YES/NO same-character verdict with confidence. Reads frames already extracted by extract8_frames.py.
Env: H3_ID_FRAMES (base dir containing <case>_lora/f00..f07.jpg), H3_ID_REF (json: case -> ref png),
H3_ID_OUT, optional H3_ID_CASES (comma list to restrict).
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
MODEL = "/gluster_hdd_cv/user/jinzili/weights/reward_models/Qwen2-VL-7B-Instruct"
FRAMES_DIR = Path(os.environ.get("H3_ID_FRAMES", "/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_lora3"))
REF_JSON = os.environ.get("H3_ID_REF", "/gluster_hdd_cv/user/jinzili/tmp/pilot/jobs_lora3.json")
OUT = Path(os.environ.get("H3_ID_OUT", "/gluster_hdd_cv/user/jinzili/tmp/pilot/identity_lora3.jsonl"))
ONLY = os.environ.get("H3_ID_CASES", "")

def main():
    torch.set_grad_enabled(False)
    m = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                        device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print("model loaded", flush=True)
    refmap = {}
    for j in json.load(open(REF_JSON)):
        case = j["case_id"]
        refs = j.get("references") or []
        refmap[case] = refs[0] if refs else None
    for d in sorted(FRAMES_DIR.iterdir()):
        if not d.is_dir():
            continue
        case = d.name
        case_key = case.replace("_lora", "")
        if ONLY and case_key not in ONLY.split(","):
            continue
        if case_key not in refmap or not refmap[case_key]:
            print("skip", case, "no ref"); continue
        frames = sorted(d.glob("f*.jpg"))
        if not frames:
            continue
        img = Image.open(frames[len(frames)//2]).convert("RGB")  # middle frame
        ref = Image.open(refmap[case_key]).convert("RGB")
        content = [
            {"type": "image"}, {"type": "image"},
            {"type": "text", "text": (
                "These are two anime images: the FIRST is a character portrait, the SECOND is a frame "
                "from a generated video. Is the character in the second image the SAME character as in "
                "the first (same identity, same design)? Answer with exactly YES or NO, then a colon, "
                "then a confidence score 0-100. Format: YES:SCORE or NO:SCORE"
            )},
        ]
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[ref, img], return_tensors="pt")
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        gen = m.generate(**inputs, max_new_tokens=32, do_sample=False)
        out_text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0].strip()
        rec = {"case": case, "ref": refmap[case_key], "verdict": out_text}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
