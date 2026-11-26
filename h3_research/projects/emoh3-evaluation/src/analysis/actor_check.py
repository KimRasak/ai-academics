#!/usr/bin/env python3
"""CLIP cross-portrait similarity matrix to determine if eval actors are distinct characters.

Portrait IDs: aNN_<emo>_<ts>_v1.png. Known same-character pair: a01 (amused, 1534125952) and
b01 (joyful, 1534125952) share the ID 1534125952. If a01..a18 are the SAME character in different
expressions, cross-portrait similarities should be uniformly high (~ same as a01-b01). If distinct
characters, different-ID pairs should be clearly lower.
"""
from __future__ import annotations
import os
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
MODEL = "/gluster_hdd_cv/user/jinzili/weights/reward_models/clip-vit-large-patch14"
BASE = Path("/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3/inputs/expressions_0825")
V2 = Path("/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3/inputs/expressions_0825_v2")

PORTS = ["a01_amused_1534125952_v1.png", "a02_angry_1534927518_v1.png",
         "a03_anxious_1535662828_v1.png", "a07_curious_1536617199_v1.png",
         "a11_joyful_1537451704_v1.png", "a15_neutral_1537882489_v1.png",
         "a18_sad_1538269564_v1.png",
         "b01_joyful_1534125952_v1.png",  # same char as a01 (ID 1534125952)
         "joyful_1534125952_v1_raw.png",  # training-char portrait (V2 dir)
         "angry_1534125952_v1_raw.png",   # training-char portrait (V2 dir)
         ]

def main():
    torch.set_grad_enabled(False)
    m = CLIPModel.from_pretrained(MODEL).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    feats = {}
    for p in PORTS:
        cand = (BASE / p) if (BASE / p).exists() else (V2 / p)
        im = Image.open(cand).convert("RGB")
        inp = proc(images=im, return_tensors="pt").to("cuda")
        with torch.no_grad():
            f = m.get_image_features(**inp)
            if not isinstance(f, torch.Tensor):
                f = f.pooler_output if hasattr(f, "pooler_output") else f.last_hidden_state[:, 0]
            f = f / f.norm(dim=-1, keepdim=True)
        feats[p] = f[0]
    print(f"{'':34s}", end="")
    for p in PORTS:
        print(f"{p[:8]:>9s}", end="")
    print()
    names = PORTS
    for i, p in enumerate(names):
        print(f"{p[:34]:34s}", end="")
        for j, q in enumerate(names):
            s = (feats[p] @ feats[q]).item()
            print(f"{s:9.3f}", end="")
        print()
    # key comparisons
    print("\nSame-char diff-emotion (a01 vs b01, both ID 1534125952):",
          f"{feats['a01_amused_1534125952_v1.png'] @ feats['b01_joyful_1534125952_v1.png']:.3f}")
    print("Same char across sessions (a01 vs joyful_1534125952_v1_raw):",
          f"{feats['a01_amused_1534125952_v1.png'] @ feats['joyful_1534125952_v1_raw.png']:.3f}")
    for q in ["a02_angry_1534927518_v1.png", "a15_neutral_1537882489_v1.png", "a18_sad_1538269564_v1.png"]:
        print(f"a01 (char1534125952) vs {q}: {feats['a01_amused_1534125952_v1.png'] @ feats[q]:.3f}")

if __name__ == "__main__":
    main()
