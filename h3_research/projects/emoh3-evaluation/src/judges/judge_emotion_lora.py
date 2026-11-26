#!/usr/bin/env python3
"""Emotion judge for LoRA outputs: Qwen2-VL-7B on frames_lora/<case>.
Same as judge_emotion.py but FRAMES_DIR/OUT come from env and '_lora' suffix is stripped.
"""
from __future__ import annotations
import json, os, sys, torch
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
MODEL = "/gluster_hdd_cv/user/jinzili/weights/reward_models/Qwen2-VL-7B-Instruct"
FRAMES_DIR = Path(os.environ.get("H3_JUDGE_FRAMES", "/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_lora"))
OUT = Path(os.environ.get("H3_JUDGE_OUT", "/gluster_hdd_cv/user/jinzili/tmp/pilot/judge_lora_results.jsonl"))

EMOTIONS = ["happy", "sad", "angry", "surprised", "anxious", "curious",
            "neutral", "disgusted", "fearful", "amused", "relieved", "other"]

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
        frames = sorted(d.glob("f*.jpg"))
        if not frames:
            continue
        images = [Image.open(f).convert("RGB") for f in frames[:4]]
        content = [{"type": "image"}] * len(images)
        content.append({"type": "text", "text": (
            "You are an anime expression evaluator. Look at the anime character's face "
            "in these video frames. Which emotion is the character expressing? "
            "Answer with exactly one word from: " + ", ".join(EMOTIONS) + ". "
            "Then a colon, then a confidence score 0-100. Format: WORD:SCORE"
        )})
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=images, return_tensors="pt")
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        gen = m.generate(**inputs, max_new_tokens=32, do_sample=False)
        out_text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0].strip()
        target = case.split("_to_")[-1].replace("_lora", "") if "_to_" in case else case
        rec = {"case": case, "target_emotion": target, "verdict": out_text}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
