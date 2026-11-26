#!/usr/bin/env python3
"""Emotion judge v2: 15-class taxonomy that includes ALL 8 task targets
(added joyful, worried, cheerful to the original 12), so every target is
literal-testable (no synonym map, no vacuous failure).
"""
from __future__ import annotations
import json, os, sys, torch
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
MODEL = "/gluster_hdd_cv/user/jinzili/weights/reward_models/Qwen2-VL-7B-Instruct"
FRAMES_DIR = Path(os.environ.get("H3_JUDGE_FRAMES", "/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_lora3"))
OUT = Path(os.environ.get("H3_JUDGE_OUT", "/gluster_hdd_cv/user/jinzili/tmp/pilot/judge15_results.jsonl"))
MAXFR = int(os.environ.get("H3_JUDGE_MAXFR", "8"))

EMOTIONS = ["happy", "joyful", "sad", "angry", "worried", "cheerful", "surprised",
            "relieved", "anxious", "curious", "neutral", "disgusted", "fearful",
            "amused", "other"]

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    torch.set_grad_enabled(False)
    m = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print("model loaded", flush=True)

    for d in sorted(FRAMES_DIR.iterdir()):
        if not d.is_dir():
            continue
        case = d.name
        if which and case != which:
            continue
        frames = sorted(d.glob("f*.jpg"))[:MAXFR]
        if not frames:
            continue
        images = [Image.open(f).convert("RGB") for f in frames]
        content = [{"type": "image"}] * len(images)
        content.append({"type": "text", "text": (
            "You are an anime expression evaluator. Look at the anime character's face "
            "across these video frames (in temporal order). Which emotion is the character "
            "expressing? Answer with exactly one word from: " + ", ".join(EMOTIONS) + ". "
            "Then a colon, then a confidence score 0-100. Format: WORD:SCORE"
        )})
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=images, return_tensors="pt")
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        gen = m.generate(**inputs, max_new_tokens=32, do_sample=False)
        out_text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0].strip()
        rec = {"case": case, "verdict": out_text, "n_frames": len(frames)}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
