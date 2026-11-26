#!/usr/bin/env python3
"""
Exp 17D (P3 Controlled Comparison — Weight-Updating LoRA vs. Zero-Weight-Update 1-Pass Surgery
         on Scale-48 Unseen-Symbol Generalization AND 96-Question General Non-Affect VQA Forgetting):

Trains a clean rank-8 LLM LoRA on Qwen2.5-VL-7B using Fold-0 train split (38 chars x 4 seen variants:
orig, placebo_hairpin, forehead_shadow_on, sweat_drop_on) and evaluates on:
  1. Held-out Fold-0 test split (10 unseen chars x 9 variants, incl. 4 UNSEEN symbols)
  2. 32 100% Native End-to-End Anime Images (cross-domain Synth -> Native transfer)
  3. 96 General Non-Affect Anime VQA Questions (Gender + Hair Color on all 48 chars) to test forgetting!
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp17c_scale48_5fold_and_unified_1pass as e17c


class LoRALinear(nn.Module):
    """Pure-PyTorch rank-r LoRA wrapper (zero peft dependency)."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scale = alpha / float(r)
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device, dtype=torch.bfloat16) * (1.0 / np.sqrt(r)))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    params: list[nn.Parameter] = []
    targets = ("q_proj", "k_proj", "v_proj", "o_proj")
    # Locate language model layers cleanly
    layers, _, _ = e17c._get_lm(model)
    for layer in layers:
        attn = layer.self_attn
        for name in targets:
            if hasattr(attn, name) and isinstance(getattr(attn, name), nn.Linear):
                wrapped = LoRALinear(getattr(attn, name), r=r, alpha=alpha)
                setattr(attn, name, wrapped)
                params.extend([wrapped.lora_A, wrapped.lora_B])
    return params


@torch.no_grad()
def eval_all(model, processor, recs, recs_nat, te_chars, ids5, ids4, ids2):
    def encode(img: Image.Image, prompt: str):
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        return processor(text=[t], images=[img], return_tensors="pt").to(model.device)

    # 1. Held-out Fold-0 9-variant test
    te_recs = [r for r in recs if r["char_id"] in te_chars]
    by_v = {v: [] for v in e17c.EXPECTED9}
    for r in te_recs:
        img = Image.open(r["image_path"]).convert("RGB")
        lg = model(**encode(img, e17c.DIRECT5_PROMPT)).logits[0, -1, ids5].float().cpu().numpy()
        pred = e17c.LABELS5[int(np.argmax(lg))]
        by_v[r["variant"]].append(int(pred == e17c.EXPECTED9[r["variant"]]))
    v_acc = {v: round(float(np.mean(xs)), 4) for v, xs in by_v.items()}
    test9 = {
        "macro9": round(float(np.mean(list(v_acc.values()))), 4),
        "clean": round(float(np.mean([v_acc["orig"], v_acc["placebo_hairpin"]])), 4),
        "seen2": round(float(np.mean([v_acc["forehead_shadow_on"], v_acc["sweat_drop_on"]])), 4),
        "unseen4": round(float(np.mean([v_acc[v] for v in e17c.UNSEEN4])), 4),
        "by_variant": v_acc,
    }

    # 2. Native 32 transfer
    nat_v = {v: [] for v in e17c.EXPECTED_NAT}
    for r in recs_nat:
        img = Image.open(r["image_path"]).convert("RGB")
        lg = model(**encode(img, e17c.DIRECT4_PROMPT)).logits[0, -1, ids4].float().cpu().numpy()
        pred = e17c.LABELS4[int(np.argmax(lg))]
        nat_v[r["variant"]].append(int(pred == e17c.EXPECTED_NAT[r["variant"]]))
    nv_acc = {v: round(float(np.mean(xs)), 4) for v, xs in nat_v.items()}
    nat_res = {
        "macro4": round(float(np.mean(list(nv_acc.values()))), 4),
        "clean": nv_acc["native_cheerful"],
        "by_variant": nv_acc,
    }

    # 3. 96 General Non-Affect VQA questions (all 48 chars x {Gender, Hair})
    orig48 = [r for r in recs if r["variant"] == "orig"]
    g_ok, h_ok = [], []
    for r in orig48:
        img = Image.open(r["image_path"]).convert("RGB")
        exp_g = "A" if r["gender"] == "girl" else "B"
        exp_h = e17c._hair_label(r["hair_color"])
        lg_g = model(**encode(img, e17c.VQA_GENDER_PROMPT)).logits[0, -1, ids2].float().cpu().numpy()
        lg_h = model(**encode(img, e17c.VQA_HAIR_PROMPT)).logits[0, -1, ids4].float().cpu().numpy()
        g_ok.append(int(["A", "B"][int(np.argmax(lg_g))] == exp_g))
        h_ok.append(int(e17c.LABELS4[int(np.argmax(lg_h))] == exp_h))
    vqa = {
        "gender_acc": round(float(np.mean(g_ok)), 4),
        "hair_acc": round(float(np.mean(h_ok)), 4),
        "overall_96q_acc": round(float(np.mean(g_ok + h_ok)), 4),
    }
    return {"test9_fold0": test9, "native32": nat_res, "general_vqa_96q": vqa}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest48", type=Path, required=True)
    ap.add_argument("--manifest-native", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs_nat = [json.loads(l) for l in args.manifest_native.read_text(encoding="utf-8").splitlines() if l.strip()]

    chars = sorted({r["char_id"] for r in recs})
    te_chars = set(chars[0::5])  # exact Fold-0 test split (10 chars)
    tr_chars = set(chars) - te_chars
    tr_recs = [r for r in recs if r["char_id"] in tr_chars and r["variant"] in e17c.SEEN_TRAIN_VARS]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    ids5 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS5]
    ids4 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS4]
    ids2 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in ["A", "B"]]

    base_model.eval()
    before = eval_all(base_model, processor, recs, recs_nat, te_chars, ids5, ids4, ids2)
    print(f"[BEFORE LoRA] Fold0 Macro9={before['test9_fold0']['macro9']:.3f} UNSEEN4={before['test9_fold0']['unseen4']:.3f} | Native32={before['native32']['macro4']:.3f} | VQA96={before['general_vqa_96q']['overall_96q_acc']:.4f}", flush=True)

    lora_params = inject_lora(base_model, r=8, alpha=16.0)
    model = base_model
    model.train()
    opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    print(f"Injected pure-PyTorch rank-8 LoRA: {sum(p.numel() for p in lora_params):,} trainable params", flush=True)

    # Preload images for fast training
    imgs_cache = {r["image_path"]: Image.open(r["image_path"]).convert("RGB") for r in tr_recs}
    rng = random.Random(42)

    for ep in range(args.epochs):
        rng.shuffle(tr_recs)
        losses = []
        for r in tr_recs:
            img = imgs_cache[r["image_path"]]
            lbl_idx = e17c.LABELS5.index(e17c.EXPECTED9[r["variant"]])
            m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": e17c.DIRECT5_PROMPT}]}]
            t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            inp = processor(text=[t], images=[img], return_tensors="pt").to(model.device)
            out = model(**inp)
            lg5 = out.logits[0, -1, ids5].float().unsqueeze(0)
            target = torch.tensor([lbl_idx], device=model.device)
            loss = torch.nn.functional.cross_entropy(lg5, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        print(f"  LoRA epoch {ep+1}/{args.epochs} loss={float(np.mean(losses)):.4f}", flush=True)

    model.eval()
    after = eval_all(model, processor, recs, recs_nat, te_chars, ids5, ids4, ids2)
    print(f"[AFTER  LoRA] Fold0 Macro9={after['test9_fold0']['macro9']:.3f} SEEN2={after['test9_fold0']['seen2']:.3f} UNSEEN4={after['test9_fold0']['unseen4']:.3f} | Native32={after['native32']['macro4']:.3f} | VQA96={after['general_vqa_96q']['overall_96q_acc']:.4f}", flush=True)

    res = {"model": Path(args.model_dir).name, "n_train_samples": len(tr_recs), "epochs": args.epochs, "before_lora": before, "after_lora": after}
    out_p = args.out_dir / f"exp17d_lora_forgetting_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
