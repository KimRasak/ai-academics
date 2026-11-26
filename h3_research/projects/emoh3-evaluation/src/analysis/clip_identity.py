#!/usr/bin/env python3
"""Identity-preservation via CLIP cosine similarity (ViT-L/14).

For each case, average CLIP image cosine similarity between the reference portrait and 3 frames
(early/middle/late) of the generated clip. Reports per-case mean similarity for base and LoRA-v3
across seeds. Env: H3_CLIP_FRAMES = base dir with <case>/f*.jpg subdirs; H3_CLIP_OUT; H3_CLIP_REF
(json mapping case->ref png); H3_CLIP_CASES (optional comma list).
"""
from __future__ import annotations
import json, os
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
MODEL = "/gluster_hdd_cv/user/jinzili/weights/reward_models/clip-vit-large-patch14"
FRAMES_DIR = Path(os.environ.get("H3_CLIP_FRAMES", "/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_lora3"))
REF_JSON = os.environ.get("H3_CLIP_REF", "/gluster_hdd_cv/user/jinzili/tmp/pilot/jobs_lora3.json")
OUT = Path(os.environ.get("H3_CLIP_OUT", "/gluster_hdd_cv/user/jinzili/tmp/pilot/clip_identity.jsonl"))
ONLY = os.environ.get("H3_CLIP_CASES", "")

def main():
    torch.set_grad_enabled(False)
    m = CLIPModel.from_pretrained(MODEL).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    print("model loaded", flush=True)
    refmap = {}
    for j in json.load(open(REF_JSON)):
        refs = j.get("references") or []
        refmap[j["case_id"]] = refs[0] if refs else None
    for d in sorted(FRAMES_DIR.iterdir()):
        if not d.is_dir():
            continue
        case = d.name
        case_key = case
        for suf in ("_lora", "_base"):
            if case_key.endswith(suf):
                case_key = case_key[: -len(suf)]
                break
        if ONLY and case_key not in ONLY.split(","):
            continue
        if case_key not in refmap or not refmap[case_key]:
            print("skip", case, "no ref"); continue
        frames = sorted(d.glob("f*.jpg"))
        if not frames:
            continue
        # pick early/middle/late
        picks = [frames[0], frames[len(frames)//2], frames[-1]]
        ref = Image.open(refmap[case_key]).convert("RGB")
        sims = []
        for fr in picks:
            im = Image.open(fr).convert("RGB")
            inp = proc(images=[ref, im], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = m.get_image_features(**inp)
                if not isinstance(out, torch.Tensor):
                    out = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0]
                feats = out
                feats = feats / feats.norm(dim=-1, keepdim=True)
            sim = (feats[0] @ feats[1]).item()
            sims.append(sim)
        mean = sum(sims) / len(sims)
        rec = {"case": case, "ref": refmap[case_key],
               "sim_frames": [round(s, 4) for s in sims], "mean": round(mean, 4)}
        print(json.dumps(rec), flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
