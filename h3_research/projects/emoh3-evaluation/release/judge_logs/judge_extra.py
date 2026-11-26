#!/usr/bin/env python3
"""Judge the 4 v2-raw known portraits with Qwen2.5-VL-7B and Qwen3-VL-32B (apples-to-apples).
Runs both models sequentially on the same GPU. Writes judge_extra.jsonl.
"""
from __future__ import annotations
import json, os, torch, sys
from pathlib import Path
from PIL import Image
os.environ.setdefault("CUDA_VISIBLE_DEVICES","6")
REFDIR2 = Path("/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3/inputs/expressions_0825_v2")
OUT = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/judge_extra.jsonl")
EMOTIONS = ["happy","joyful","sad","angry","worried","cheerful","surprised",
            "relieved","anxious","curious","neutral","disgusted","fearful","amused","other"]
PROMPT = ("You are an anime expression evaluator. Look at the anime character's face. "
          "Which emotion is the character expressing? Respond with EXACTLY ONE word from: "
          +", ".join(EMOTIONS)+", a colon, and a confidence score 0-100. No explanation. Format: WORD:SCORE")
TARGETS = [("v2raw_"+p.stem, p) for p in sorted(REFDIR2.glob("*_raw.png"))]

def run(model_path, class_import, tag):
    from transformers import AutoProcessor
    mod = __import__("transformers")
    cls = getattr(mod, class_import)
    print(f"loading {tag} ...", flush=True)
    m = cls.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"{tag} loaded", flush=True)
    for label, p in TARGETS:
        known = p.stem.split("_")[0]
        imgs=[Image.open(p).convert("RGB")]
        content=[{"type":"image"}]*len(imgs)+[{"type":"text","text":PROMPT}]
        msgs=[{"role":"user","content":content}]
        text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs=proc(text=[text], images=imgs, return_tensors="pt")
        inputs={k: v.to("cuda") if hasattr(v,"to") else v for k,v in inputs.items()}
        gen=m.generate(**inputs, max_new_tokens=64, do_sample=False)
        v=proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        rec={"case":label,"model":tag,"known":known,"verdict":v,"n_frames":1}
        print(json.dumps(rec,ensure_ascii=False),flush=True)
        with OUT.open("a") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")
    del m
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"{tag} DONE", flush=True)

if __name__=="__main__":
    torch.set_grad_enabled(False)
    run("/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct" if len(sys.argv)<2 else sys.argv[1],
        "Qwen2_5_VLForConditionalGeneration","Qwen2.5-VL-7B")
    run("/gluster_hdd_cv/user/jinzili/weights/Qwen3-VL-32B-Instruct" if len(sys.argv)<3 else sys.argv[2],
        "Qwen3VLForConditionalGeneration","Qwen3-VL-32B")
    print("ALL DONE", flush=True)
