#!/usr/bin/env python3
"""
Exp 23 (P3 Mechanistically-Guided Training Comparison — Late-Only Circuit-Targeted KL-Anchored LoRA
        vs. Full-LLM LoRA (Exp 17D) vs. Zero-Weight-Update 1-Pass Circuit Surgery (Exp 17C)!):

Uses our Exp 14A/16 Causal Double-Dissociation directly to guide parameter-efficient fine-tuning:
  1. FREEZE 100% of vision encoder & early/mid LLM layers [L0..L20] (preserving mid-layer representation!).
  2. Inject rank-8 LoRA ONLY into Late Override Layers [L21..L27] targeting the model's causal override module:
       - InternVL3-8B: Late `self_attn` (q/k/v/o_proj)
       - Qwen2.5-VL-7B & LLaVA-OV-7B: Late `mlp` (gate/up/down_proj)
  3. Train on Fold-0 train split ONLY (38 chars x 4 seen variants: orig, placebo, forehead_shadow, sweat_drop)
     with exact KL-Anchoring to base logits on clean train faces (ZERO test chars, ZERO unseen symbols!).

Evaluates head-to-head on:
  - Fold-0 Held-Out 10 Unseen Chars x 9 Variants (including 4 UNSEEN symbols!)
  - 32 100% Native End-to-End Anime Images (Synth -> Native OOD!)
  - 96 General Non-Affect Anime VQA Questions (Gender + Hair Color — measuring exact VQA forgetting!)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp17c_scale48_5fold_and_unified_1pass as e17c
import exp17d_lora_vs_surgery_vqa_forgetting as e17d


def inject_late_circuit_lora(model: nn.Module, model_name: str, r: int = 8, alpha: float = 16.0) -> tuple[list[nn.Parameter], str, list[int]]:
    for p in model.parameters():
        p.requires_grad = False
    layers, _, _ = e17c._get_lm(model)
    nL = len(layers)
    l_start = int(nL * 0.75)  # Late override window only (e.g. L21..L27 on 28L!)
    use_attn = "internvl" in model_name.lower()
    mode_str = "late_ATTN" if use_attn else "late_MLP"
    params: list[nn.Parameter] = []
    for li in range(l_start, nL):
        layer = layers[li]
        if use_attn:
            sub = layer.self_attn
            names = ("q_proj", "k_proj", "v_proj", "o_proj")
        else:
            sub = layer.mlp
            names = ("gate_proj", "up_proj", "down_proj")
        for nm in names:
            if hasattr(sub, nm) and isinstance(getattr(sub, nm), nn.Linear):
                w = e17d.LoRALinear(getattr(sub, nm), r=r, alpha=alpha)
                setattr(sub, nm, w)
                params.extend([w.lora_A, w.lora_B])
    return params, mode_str, [l_start, nL - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest48", type=Path, required=True)
    ap.add_argument("--manifest-native", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--kl-weight", type=float, default=2.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs_nat = [json.loads(l) for l in args.manifest_native.read_text(encoding="utf-8").splitlines() if l.strip()]

    chars = sorted({r["char_id"] for r in recs})
    te_chars = set(chars[0::5])  # exact Fold-0 test split (10 chars)
    tr_chars = set(chars) - te_chars
    tr_recs = [r for r in recs if r["char_id"] in tr_chars and r["variant"] in e17c.SEEN_TRAIN_VARS]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    ids5 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS5]
    ids4 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS4]
    ids2 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in ["A", "B"]]

    model.eval()
    before = e17d.eval_all(model, processor, recs, recs_nat, te_chars, ids5, ids4, ids2)
    print(f"[{Path(args.model_dir).name} BEFORE] Macro9={before['test9_fold0']['macro9']:.3f} U4={before['test9_fold0']['unseen4']:.3f} | Nat32={before['native32']['macro4']:.3f} | VQA96={before['general_vqa_96q']['overall_96q_acc']:.4f}", flush=True)

    # Precompute base logits on clean train faces for exact KL-anchoring (zero test leakage!)
    def encode(img: Image.Image, prompt: str):
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        return processor(text=[t], images=[img], return_tensors="pt").to(model.device)

    imgs_cache = {r["image_path"]: Image.open(r["image_path"]).convert("RGB") for r in tr_recs}
    base_clean_probs: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for r in tr_recs:
            if r["variant"] in {"orig", "placebo_hairpin"}:
                lg = model(**encode(imgs_cache[r["image_path"]], e17c.DIRECT5_PROMPT)).logits[0, -1, ids5].float()
                base_clean_probs[r["image_path"]] = F.softmax(lg, dim=-1).detach()

    # Inject Late-Only Circuit-Targeted LoRA
    lora_params, mode_str, layer_rng = inject_late_circuit_lora(model, args.model_dir, r=8, alpha=16.0)
    print(f"[{Path(args.model_dir).name}] Injected {mode_str} LoRA on layers {layer_rng}: {sum(p.numel() for p in lora_params):,} params", flush=True)

    model.train()
    opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    rng = random.Random(42)

    for ep in range(args.epochs):
        rng.shuffle(tr_recs)
        losses = []
        for r in tr_recs:
            img = imgs_cache[r["image_path"]]
            lbl_idx = e17c.LABELS5.index(e17c.EXPECTED9[r["variant"]])
            inp = encode(img, e17c.DIRECT5_PROMPT)
            lg5 = model(**inp).logits[0, -1, ids5].float().unsqueeze(0)
            target = torch.tensor([lbl_idx], device=model.device)
            loss = F.cross_entropy(lg5, target)
            if r["image_path"] in base_clean_probs:
                p_ref = base_clean_probs[r["image_path"]].unsqueeze(0)
                kl = F.kl_div(F.log_softmax(lg5, dim=-1), p_ref, reduction="batchmean")
                loss = loss + args.kl_weight * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            losses.append(float(loss.item()))
        print(f"  [{Path(args.model_dir).name}] ep {ep+1}/{args.epochs} loss={float(np.mean(losses)):.4f}", flush=True)

    model.eval()
    after = e17d.eval_all(model, processor, recs, recs_nat, te_chars, ids5, ids4, ids2)
    print(f"[{Path(args.model_dir).name} AFTER {mode_str}] Macro9={after['test9_fold0']['macro9']:.3f} Seen2={after['test9_fold0']['seen2']:.3f} U4={after['test9_fold0']['unseen4']:.3f} | Nat32={after['native32']['macro4']:.3f} | VQA96={after['general_vqa_96q']['overall_96q_acc']:.4f} (G={after['general_vqa_96q']['gender_acc']:.3f} H={after['general_vqa_96q']['hair_acc']:.3f})", flush=True)

    res = {
        "model": Path(args.model_dir).name,
        "circuit_target": mode_str,
        "layers": layer_rng,
        "kl_weight": args.kl_weight,
        "before": before,
        "after_late_kl_lora": after,
    }
    out_p = args.out_dir / f"exp23_late_kl_lora_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
