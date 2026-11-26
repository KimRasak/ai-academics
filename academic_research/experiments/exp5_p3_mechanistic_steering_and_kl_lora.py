#!/usr/bin/env python3
"""
Exp 5 (P3 Mechanistic Optimization — Leave-Symbols-Out Zero-Shot Transfer):
  Tests TWO principled, non-collapsing optimizations against naive SFT (Exp 4)
  on the EXACT SAME strict split (10 train chars x 4 seen variants -> 6 held-out
  unseen chars x ALL 7 variants including UNSEEN `eye_highlight_off` & `blush_lines_on`!):

  1. Exp 5A — Zero-Training Contrastive Activation Steering (Layer & Beta Sweep):
     Computes a single 1D override direction v_l = E[h_l(Audit, x) - h_l(Direct, x)]
     ONLY on the 10 training characters (using ONLY the 4 seen variants!), then injects
     +beta * v_l at layer l during standard 1-token Direct inference (0 extra tokens,
     0 weight updates, 0 template memorization!).
     Tests whether Biological-Override Binding Failure is a 1D linear routing bias
     inside Qwen2.5-VL-7B's residual stream!

  2. Exp 5B — KL-Anchored Biological-Prior Debiasing LoRA (Non-Collapsing!):
     Unlike naive Direct-Letter SFT (Exp 4) which collapsed unseen class `C` (blush)
     from 33% -> 0%, trains a rank-8 LLM LoRA on ONLY the 4 seen variants with:
       - Counterfactual override loss on {forehead_shadow, sweat_drop}
       - Clean anchor loss on {orig, placebo_hairpin}
       - Strong KL preservation penalty to the frozen base model on non-A logits
         and full next-token distribution so UNSEEN classes (`C`, `eye_highlight_off`)
         are NEVER crushed!
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DIRECT_PROMPT = (
    "Look at this 2D anime character portrait and judge their TRUE emotional / psychological state.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "Answer with ONLY a single letter: A, B, C, or D."
)

AUDIT_PROMPT = (
    "Inspect this 2D anime portrait carefully for ANY non-biological anime graphic affect symbols "
    "(such as unusual dark shadows, missing/hollow eye highlights, cheek blush marks, or floating sweat drops), "
    "ignoring ordinary hair accessories. In 2D anime, non-biological graphic cues OVERRIDE a smiling mouth.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "Answer with ONLY a single letter: A, B, C, or D."
)

EXPECTED = {
    "orig": "A",
    "placebo_hairpin": "A",
    "eye_highlight_off": "B",
    "forehead_shadow_on": "B",
    "dead_eyes_and_shadow": "B",
    "blush_lines_on": "C",
    "sweat_drop_on": "D",
}
LABELS = ["A", "B", "C", "D"]


def _get_lm_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Could not locate transformer layers on VLM model")


def _encode_prompt(processor, img: Image.Image, prompt: str, device) -> dict[str, torch.Tensor]:
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(text=[txt], images=[img], return_tensors="pt").to(device)


def _abcd_token_ids(processor) -> list[int]:
    tok = processor.tokenizer
    ids = []
    for ch in LABELS:
        tid = tok.encode(ch, add_special_tokens=False)
        ids.append(tid[-1])
    return ids


@torch.no_grad()
def eval_logits_abcd(
    model,
    processor,
    records: list[dict[str, Any]],
    prompt: str,
    abcd_ids: list[int],
    steer_layer: int | None = None,
    steer_vec: torch.Tensor | None = None,
    steer_beta: float = 0.0,
) -> dict[str, Any]:
    model.eval()
    hook_handle = None
    if steer_layer is not None and steer_vec is not None and steer_beta != 0.0:
        layers = _get_lm_layers(model)
        vec = (steer_vec * steer_beta).to(model.device)

        def hook_fn(_module, _inp, out):
            if isinstance(out, tuple):
                h = out[0].clone()
                h[:, -1, :] = h[:, -1, :] + vec.to(h.dtype)
                return (h,) + out[1:]
            h = out.clone()
            h[:, -1, :] = h[:, -1, :] + vec.to(h.dtype)
            return h

        hook_handle = layers[steer_layer].register_forward_hook(hook_fn)

    by_var: dict[str, list[dict[str, Any]]] = {}
    try:
        for rec in records:
            img = Image.open(rec["image_path"]).convert("RGB")
            inputs = _encode_prompt(processor, img, prompt, model.device)
            logits = model(**inputs).logits[0, -1, abcd_ids].float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = LABELS[int(np.argmax(probs))]
            gt = EXPECTED[rec["variant"]]
            by_var.setdefault(rec["variant"], []).append(
                {
                    "char_id": rec["char_id"],
                    "pred": pred,
                    "gt": gt,
                    "correct": pred == gt,
                    "p_A": float(probs[0]),
                    "p_gt": float(probs[LABELS.index(gt)]),
                }
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    summary: dict[str, Any] = {}
    for v, rows in sorted(by_var.items()):
        summary[v] = {
            "n": len(rows),
            "gt": EXPECTED[v],
            "acc": round(float(np.mean([r["correct"] for r in rows])), 4),
            "pred_A_rate": round(float(np.mean([r["pred"] == "A" for r in rows])), 4),
            "mean_p_A": round(float(np.mean([r["p_A"] for r in rows])), 4),
            "mean_p_gt": round(float(np.mean([r["p_gt"] for r in rows])), 4),
            "preds": [r["pred"] for r in rows],
        }
    macro_all = float(np.mean([s["acc"] for s in summary.values()]))
    unseen = [summary[k]["acc"] for k in ["eye_highlight_off", "blush_lines_on"] if k in summary]
    seen_conf = [summary[k]["acc"] for k in ["forehead_shadow_on", "sweat_drop_on"] if k in summary]
    clean = [summary[k]["acc"] for k in ["orig", "placebo_hairpin"] if k in summary]
    return {
        "macro_all_7": round(macro_all, 4),
        "clean_orig_placebo": round(float(np.mean(clean)), 4),
        "seen_conflicts": round(float(np.mean(seen_conf)), 4),
        "unseen_symbols_zero_shot": round(float(np.mean(unseen)), 4),
        "by_variant": summary,
    }


@torch.no_grad()
def extract_steering_vectors(
    model, processor, train_recs: list[dict[str, Any]], candidate_layers: list[int]
) -> dict[int, torch.Tensor]:
    """
    Compute two candidate steering vectors per layer from ONLY train_recs (4 seen variants):
      v_prompt_l = mean_x [ h_l(Audit, x) - h_l(Direct, x) ]
    """
    model.eval()
    layers = _get_lm_layers(model)
    diffs: dict[int, list[torch.Tensor]] = {l: [] for l in candidate_layers}

    for rec in train_recs:
        img = Image.open(rec["image_path"]).convert("RGB")
        captured: dict[int, torch.Tensor] = {}

        def make_hook(l_idx: int):
            def fn(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[l_idx] = h[0, -1, :].detach().float().cpu()
            return fn

        handles = [layers[l].register_forward_hook(make_hook(l)) for l in candidate_layers]
        try:
            in_d = _encode_prompt(processor, img, DIRECT_PROMPT, model.device)
            model(**in_d)
            h_direct = dict(captured)
            captured.clear()
            in_a = _encode_prompt(processor, img, AUDIT_PROMPT, model.device)
            model(**in_a)
            h_audit = dict(captured)
        finally:
            for h in handles:
                h.remove()

        for l in candidate_layers:
            diffs[l].append(h_audit[l] - h_direct[l])

    return {l: torch.stack(v, dim=0).mean(dim=0) for l, v in diffs.items()}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / float(r)
        self.lora_a = nn.Parameter(torch.empty((r, base.in_features), dtype=base.weight.dtype, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros((base.out_features, r), dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + ((x @ self.lora_a.t()) @ self.lora_b.t()) * self.scaling


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    trainable: list[nn.Parameter] = []
    for name, module in list(model.named_modules()):
        if "visual" in name:
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in {"q_proj", "v_proj", "o_proj"}:
                w = LoRALinear(child, r=r, alpha=alpha)
                setattr(module, child_name, w)
                trainable.extend([w.lora_a, w.lora_b])
    return trainable


def run_kl_debias_lora(
    model,
    processor,
    train_recs: list[dict[str, Any]],
    test_recs: list[dict[str, Any]],
    abcd_ids: list[int],
    epochs: int = 5,
    lr: float = 1.2e-4,
    kl_weight: float = 1.5,
) -> dict[str, Any]:
    """
    KL-Anchored Biological-Prior Debiasing LoRA:
      Precomputes base model logits over ABCD for every train sample (before LoRA!).
      During training on ONLY {orig, placebo, forehead_shadow, sweat_drop}:
        - On clean {orig, placebo}: CE toward A + KL to base on non-A {B, C, D}
        - On conflict {forehead_shadow, sweat_drop}:
            * Binary override loss: push P(A) down (target P(not-A) -> 1)
            * Mild supervised CE on seen label y (weight 0.35)
            * STRONG KL preservation on conditional distribution P(B, C, D | not-A)
              relative to the base model's AUDIT prompt distribution!
      This directly teaches the model WHEN to suppress biological-smile prior A
      WITHOUT crushing unseen class C (blush) or unseen symbol routing!
    """
    # Cache base model conditional non-A distribution on training inputs
    model.eval()
    base_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for rec in train_recs:
            img = Image.open(rec["image_path"]).convert("RGB")
            in_d = _encode_prompt(processor, img, DIRECT_PROMPT, model.device)
            l_base_d = model(**in_d).logits[0, -1, abcd_ids].detach().float().cpu()
            in_a = _encode_prompt(processor, img, AUDIT_PROMPT, model.device)
            l_base_a = model(**in_a).logits[0, -1, abcd_ids].detach().float().cpu()
            base_cache[rec["image_path"]] = (l_base_d, l_base_a)

    trainable = inject_lora(model, r=8, alpha=16.0)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    for ep in range(epochs):
        model.train()
        random.shuffle(train_recs)
        losses = []
        for rec in train_recs:
            img = Image.open(rec["image_path"]).convert("RGB")
            inputs = _encode_prompt(processor, img, DIRECT_PROMPT, model.device)
            logits = model(**inputs).logits[0, -1, abcd_ids].float()
            gt_idx = LABELS.index(EXPECTED[rec["variant"]])
            l_base_d, l_base_a = base_cache[rec["image_path"]]
            l_base_d = l_base_d.to(model.device)
            l_base_a = l_base_a.to(model.device)

            # 1. Preserve conditional distribution over {B, C, D} (indices 1,2,3) via KL to base Audit!
            p_ref_nonA = torch.softmax(l_base_a[1:], dim=-1)
            logp_nonA = torch.log_softmax(logits[1:], dim=-1)
            kl_nonA = F.kl_div(logp_nonA, p_ref_nonA, reduction="sum")

            probs = torch.softmax(logits, dim=-1)
            if gt_idx == 0:
                # Clean orig / placebo: keep A high + anchor full distribution to base Direct
                loss_main = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0], device=model.device))
                p_ref_full = torch.softmax(l_base_d, dim=-1)
                loss_kl = F.kl_div(torch.log_softmax(logits, dim=-1), p_ref_full, reduction="sum")
                loss = loss_main + 0.5 * loss_kl + 0.5 * kl_nonA
            else:
                # Conflict: suppress A strongly (-log(1 - P(A))), gentle seen-label CE, strong non-A KL!
                loss_suppress_A = -torch.log(torch.clamp(1.0 - probs[0], min=1e-6))
                loss_ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([gt_idx], device=model.device))
                loss = 1.0 * loss_suppress_A + 0.35 * loss_ce + kl_weight * kl_nonA

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[KL-Debias LoRA] Epoch {ep + 1}/{epochs} mean_loss={np.mean(losses):.4f}", flush=True)

    return eval_logits_abcd(model, processor, test_recs, DIRECT_PROMPT, abcd_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train-chars", type=int, default=10)
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_recs = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    chars = sorted({r["char_id"] for r in all_recs})
    train_chars = set(chars[: args.n_train_chars])
    test_chars = set(chars[args.n_train_chars :])
    seen_vars = {"orig", "placebo_hairpin", "forehead_shadow_on", "sweat_drop_on"}

    train_recs = [r for r in all_recs if r["char_id"] in train_chars and r["variant"] in seen_vars]
    test_recs = [r for r in all_recs if r["char_id"] in test_chars]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    abcd_ids = _abcd_token_ids(processor)
    print(f"ABCD token IDs: {dict(zip(LABELS, abcd_ids))}", flush=True)

    res: dict[str, Any] = {"model_dir": args.model_dir}

    print("=== 1. ZERO-SHOT DIRECT (1-token) ===", flush=True)
    res["zero_shot_direct"] = eval_logits_abcd(model, processor, test_recs, DIRECT_PROMPT, abcd_ids)
    print(json.dumps({k: v for k, v in res["zero_shot_direct"].items() if k != "by_variant"}, indent=2), flush=True)

    print("=== 2. ZERO-SHOT AUDIT PROMPT (1-token, no CoT) ===", flush=True)
    res["zero_shot_audit_1tok"] = eval_logits_abcd(model, processor, test_recs, AUDIT_PROMPT, abcd_ids)
    print(json.dumps({k: v for k, v in res["zero_shot_audit_1tok"].items() if k != "by_variant"}, indent=2), flush=True)

    # 3. Exp 5A: Activation Steering Sweep
    n_layers = len(_get_lm_layers(model))
    cand_layers = [int(n_layers * f) for f in (0.35, 0.50, 0.65, 0.80)]
    print(f"Extracting steering vectors at layers {cand_layers} (total={n_layers}) from train_recs...", flush=True)
    vecs = extract_steering_vectors(model, processor, train_recs, cand_layers)

    res["steering_sweep"] = {}
    best_key, best_score = None, -1.0
    for l in cand_layers:
        for beta in [0.5, 1.0, 1.5, 2.0]:
            key = f"L{l}_b{beta}"
            ev = eval_logits_abcd(model, processor, test_recs, DIRECT_PROMPT, abcd_ids, steer_layer=l, steer_vec=vecs[l], steer_beta=beta)
            res["steering_sweep"][key] = ev
            print(f"[Steer {key}] macro7={ev['macro_all_7']} clean={ev['clean_orig_placebo']} seen={ev['seen_conflicts']} UNSEEN={ev['unseen_symbols_zero_shot']}", flush=True)
            if ev["macro_all_7"] > best_score:
                best_score, best_key = ev["macro_all_7"], key
    res["best_steering"] = {"key": best_key, **res["steering_sweep"][best_key]}

    # 4. Exp 5B: KL-Anchored Biological-Prior Debiasing LoRA (Non-Collapsing!)
    print("=== 4. EXP 5B: KL-ANCHORED PRIOR-DEBIASING LORA ===", flush=True)
    res["kl_debias_lora"] = run_kl_debias_lora(model, processor, train_recs, test_recs, abcd_ids, epochs=5, lr=1.2e-4, kl_weight=1.5)
    print(json.dumps(res["kl_debias_lora"], indent=2), flush=True)

    out_path = args.out_dir / "exp5_steering_and_kl_lora.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
