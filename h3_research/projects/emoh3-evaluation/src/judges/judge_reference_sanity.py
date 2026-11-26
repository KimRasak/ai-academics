#!/usr/bin/env python3
"""Judge sanity check: classify KNOWN-LABEL reference portraits with the same Qwen2-VL judge
prompt/taxonomy used for generated clips. If the judge mislabels these, its verdicts on generated
clips are suspect; if it labels them well, that is evidence the judge can see emotion in anime faces.
"""
from __future__ import annotations
import json, os, torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
MODEL = "/gluster_hdd_cv/user/jinzili/weights/reward_models/Qwen2-VL-7B-Instruct"
BASE = "/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3/inputs"
OUT = Path(os.environ.get("H3_SANITY_OUT", "/gluster_hdd_cv/user/jinzili/tmp/pilot/judge_reference_sanity.jsonl"))

EMOTIONS = ["happy", "joyful", "sad", "angry", "worried", "cheerful", "surprised",
            "relieved", "anxious", "curious", "neutral", "disgusted", "fearful",
            "amused", "other"]

# (relative path, known label mapped to the 15-class taxonomy)
REFS = [
    ("expressions_0825/a01_amused_1534125952_v1.png", "amused"),
    ("expressions_0825/a02_angry_1534927518_v1.png", "angry"),
    ("expressions_0825/a03_anxious_1535662828_v1.png", "anxious"),
    ("expressions_0825/a07_curious_1536617199_v1.png", "curious"),
    ("expressions_0825/a08_disgusted_1536815405_v1.png", "disgusted"),
    ("expressions_0825/a11_joyful_1537451704_v1.png", "joyful"),
    ("expressions_0825/a15_neutral_1537882489_v1.png", "neutral"),
    ("expressions_0825/a18_sad_1538269564_v1.png", "sad"),
    ("expressions_0825/a21_scared_1538613676_v1.png", "fearful"),
    ("expressions_0825/a24_surprised_1538797510_v1.png", "surprised"),
    ("expressions_0825_v2/angry_1534125952_v1_raw.png", "angry"),
    ("expressions_0825_v2/joyful_1534125952_v1_raw.png", "joyful"),
    ("expressions_0825_v2/sad_1534125952_v1_raw.png", "sad"),
    ("expressions_0825_v2/surprised_1534125952_v1_raw.png", "surprised"),
    ("expressions_0825/b01_joyful_1534125952_v1.png", "joyful"),
    ("expressions_0825/b02_joyful_1534125952_v2.png", "joyful"),
    ("expressions_0825/b03_joyful_1534125952_v3.png", "joyful"),
    ("expressions_0825/b04_joyful_1534125952_v4.png", "joyful"),
    ("expressions_0825/b05_joyful_1534125952_v5.png", "joyful"),
    ("expressions_0825/c02_sad_1538378146_v2_10s.png", "sad"),
    ("expressions_0825/d01_joyful_1536757874_v3_speech.png", "joyful"),
]

def main():
    torch.set_grad_enabled(False)
    m = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    print("model loaded", flush=True)

    results = []
    for rel, known in REFS:
        p = Path(BASE) / rel
        img = Image.open(p).convert("RGB")
        content = [{"type": "image"},
                   {"type": "text", "text": (
                       "You are an anime expression evaluator. Look at the anime character's face "
                       "in this image. Which emotion is the character expressing? "
                       "Answer with exactly one word from: " + ", ".join(EMOTIONS) + ". "
                       "Then a colon, then a confidence score 0-100. Format: WORD:SCORE"
                   )}]
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt")
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        gen = m.generate(**inputs, max_new_tokens=32, do_sample=False)
        out_text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0].strip()
        judged = out_text.split(":")[0].strip().lower()
        ok = (judged == known)
        rec = {"ref": rel, "known": known, "judged": judged, "verdict": out_text, "ok": ok}
        results.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(results); acc = sum(r["ok"] for r in results)
    print(f"SUMMARY accuracy={acc}/{n} ({acc/n:.2f})")
    # per-emotion breakdown
    from collections import Counter, defaultdict
    by = defaultdict(list)
    for r in results:
        by[r["known"]].append(r["ok"])
    for k, v in by.items():
        print(f"  {k}: {sum(v)}/{len(v)}")

if __name__ == "__main__":
    main()
