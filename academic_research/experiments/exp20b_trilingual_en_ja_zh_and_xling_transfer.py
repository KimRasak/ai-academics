#!/usr/bin/env python3
"""
Exp 20B (P3 Trilingual Generalization & Zero-Shot Cross-Lingual Mechanistic Gate Transfer —
         English / Japanese (日本語) / Chinese (中文) on Scale-48 = 432 Images!):

Tests the critical cultural/linguistic hypothesis:
  "Anime affect tropes (ハイライトオフ, ガーン/影, 汗マーク, 怒りマーク 💢) originate in
   Japanese/Chinese ACG culture. Is Biological-Override merely an English-prompt artifact,
   or does it persist identically in native Japanese and Chinese prompts?"

AND tests ZERO-SHOT CROSS-LINGUAL TRANSFER of our frozen mid-layer anomaly gate:
  - Train the frozen mid-layer linear gate ONLY on English prompts (fc_00..fc_23 x {orig, placebo}
    vs {forehead_shadow, sweat_drop}).
  - Evaluate ZERO-SHOT on held-out unseen characters (fc_24..fc_47, 24 chars x 9 variants = 216 imgs)
    prompted in **Japanese (`JA`)** and **Chinese (`ZH`)** — testing BOTH:
      1. Direct JA / Direct ZH Biological-Override rates & Bio-vs-Vein gap (`bio_angry_brows` vs `💢`)
      2. Zero-Shot `EN -> JA` and `EN -> ZH` 1-Pass Unified Bidirectional Circuit Surgery!
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp17c_scale48_5fold_and_unified_1pass as e17c

PROMPTS = {
    "EN": e17c.DIRECT5_PROMPT,
    "JA": (
        "この2Dアニメキャラクターのイラストを見て、本当の感情・心理状態を判断してください。\n"
        "最も適切な選択肢を1つだけ選んでください：\n"
        "A) 心から明るく、無邪気で楽しそうに喜んでいる\n"
        "B) 心が虚ろ・不気味・闇を抱えている、または暗い本心を隠している\n"
        "C) 恥ずかしい、照れている、または顔を赤らめている\n"
        "D) 気まずい、戸惑っている、または冷や汗をかいて焦っている\n"
        "E) イライラしている、怒っている、または笑顔の裏で怒りを抑えている\n"
        "アルファベット1文字のみ（A、B、C、D、E）で答えてください。"
    ),
    "ZH": (
        "请仔细观察这张二次元动漫角色立绘，判断角色内心真实的感情与心理状态。\n"
        "请只选择最符合的一项：\n"
        "A) 发自内心感到开心、明朗、无忧无虑\n"
        "B) 内心空洞、阴森、黑化，或隐藏着阴暗/危险的情绪\n"
        "C) 害羞、脸红、不好意思或腼腆\n"
        "D) 尴尬、无语、紧张不安或冒冷汗\n"
        "E) 烦躁、生气，或在微笑背后强忍着怒火\n"
        "请仅回复一个大写字母：A、B、C、D 或 E。"
    ),
}


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest48", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    chars = sorted({r["char_id"] for r in recs})
    tr_chars = set(chars[:24])   # Train gate ONLY in English on first 24 chars
    te_chars = set(chars[24:])   # Test ZERO-SHOT on 24 strictly unseen chars in EN, JA, ZH!

    tr_recs = [r for r in recs if r["char_id"] in tr_chars and r["variant"] in e17c.SEEN_TRAIN_VARS]
    te_recs = [r for r in recs if r["char_id"] in te_chars]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ids5 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS5]

    layers, final_norm, lm_head = e17c._get_lm(model)
    nL = len(layers)
    l_mid = int(nL * 0.55)
    l_late_gate = int(nL * 0.72)
    l_surg = l_late_gate + 1

    W5 = lm_head.weight[ids5].detach().float()
    wb5 = W5[0] - W5[1:].mean(dim=0)
    if final_norm is not None and hasattr(final_norm, "weight"):
        wb5 = wb5 * final_norm.weight.detach().float()
    w_bio5 = (wb5 / (wb5.norm() + 1e-8)).to(model.device)

    cap: dict[int, np.ndarray] = {}
    surg_on = {"on": False}
    hooks = []
    for li in (l_mid, l_late_gate):
        def _mk_c(idx):
            def _fn(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                cap[idx] = h[0, -1, :].detach().float().cpu().numpy()
            return _fn
        hooks.append(layers[li].register_forward_hook(_mk_c(li)))

    for li in range(l_surg, nL):
        def _mk_s():
            def _fn(_m, _i, out):
                if not surg_on["on"]:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                v = h[0, -1, :].float()
                proj = torch.dot(v, w_bio5)
                if proj.item() > 0.0:
                    h = h.clone()
                    h[0, -1, :] = (v - 1.5 * proj * w_bio5).to(h.dtype)
                    return (h,) + out[1:] if isinstance(out, tuple) else h
                return out
            return _fn
        hooks.append(layers[li].self_attn.register_forward_hook(_mk_s()))
        hooks.append(layers[li].mlp.register_forward_hook(_mk_s()))

    def encode(img: Image.Image, prompt: str):
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        return processor(text=[t], images=[img], return_tensors="pt").to(model.device)

    # 1. Fit EN-ONLY frozen mid-layer anomaly gate + per-language clean baseline means (clean faces only!)
    X_en_tr, y_en_tr = [], []
    clean_surg_by_lang = {lang: [] for lang in PROMPTS}
    clean_orig_paths = [r["image_path"] for r in tr_recs if r["variant"] in {"orig", "placebo_hairpin"}]

    for r in tr_recs:
        img = Image.open(r["image_path"]).convert("RGB")
        cap.clear(); surg_on["on"] = False
        model(**encode(img, PROMPTS["EN"]))
        X_en_tr.append(np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0))
        y_en_tr.append(0.0 if r["variant"] in {"orig", "placebo_hairpin"} else 1.0)

    X_en_tr = np.stack(X_en_tr, axis=0)
    y_en_tr = np.array(y_en_tr, dtype=np.float32)
    w_en, b_en = e17c._fit_logreg(X_en_tr, y_en_tr)
    tau_en = e17c._cal_thresh(e17c._score(X_en_tr, w_en, b_en), y_en_tr)

    # Clean-prior mean logit vector on clean train faces (16 clean faces per language, zero conflict labels!)
    for op in clean_orig_paths[:16]:
        img = Image.open(op).convert("RGB")
        for lang, prm in PROMPTS.items():
            surg_on["on"] = True
            clean_surg_by_lang[lang].append(model(**encode(img, prm)).logits[0, -1, ids5].float().cpu().numpy())
    mu_surg = {lang: np.mean(vs, axis=0) for lang, vs in clean_surg_by_lang.items()}

    # 2. Evaluate on 24 strictly unseen test chars x 9 variants = 216 imgs in EN, JA, and ZH!
    results = {}
    for lang, prm in PROMPTS.items():
        dir_ok = {v: [] for v in e17c.EXPECTED9}
        p1_ok = {v: [] for v in e17c.EXPECTED9}
        fire_v = {v: [] for v in e17c.EXPECTED9}
        corr_dir_all, corr_p1_all = [], []

        for r in te_recs:
            img = Image.open(r["image_path"]).convert("RGB")
            inp = encode(img, prm)
            cap.clear(); surg_on["on"] = False
            lg_d = model(**inp).logits[0, -1, ids5].float().cpu().numpy()
            h = np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0)[None, :]
            # ZERO-SHOT cross-lingual gate application: EN-trained (w_en, b_en, tau_en) on JA/ZH residual stream!
            fired = bool(e17c._score(h, w_en, b_en)[0] >= tau_en)

            surg_on["on"] = True
            lg_s = model(**inp).logits[0, -1, ids5].float().cpu().numpy()

            exp_l = e17c.EXPECTED9[r["variant"]]
            pd = e17c.LABELS5[int(np.argmax(lg_d))]
            p1 = "A" if not fired else e17c.LABELS5[1 + int(np.argmax((lg_s - mu_surg[lang])[1:]))]

            c_d, c_1 = int(pd == exp_l), int(p1 == exp_l)
            dir_ok[r["variant"]].append(c_d)
            p1_ok[r["variant"]].append(c_1)
            fire_v[r["variant"]].append(int(fired))
            corr_dir_all.append(c_d)
            corr_p1_all.append(c_1)

        def summ(dmap):
            bv = {v: round(float(np.mean(xs)), 4) for v, xs in dmap.items()}
            return {
                "macro9": round(float(np.mean(list(bv.values()))), 4),
                "clean": round(float(np.mean([bv["orig"], bv["placebo_hairpin"]])), 4),
                "unseen4": round(float(np.mean([bv[v] for v in e17c.UNSEEN4])), 4),
                "bio_vs_vein_gap": round(bv["bio_angry_brows"] - bv["anger_vein_on"], 4),
                "by_variant": bv,
            }

        sd, s1 = summ(dir_ok), summ(p1_ok)
        pval = e17c._mcnemar_exact_p(corr_dir_all, corr_p1_all)
        results[lang] = {
            "direct": sd,
            "ours_1pass_xling_from_EN": s1,
            "mcnemar_p": round(pval, 6),
            "xling_gate_fire_rate": {v: round(float(np.mean(xs)), 4) for v, xs in fire_v.items()},
        }
        print(
            f"[{Path(args.model_dir).name}] {lang} | Direct: Macro9={sd['macro9']:.3f} Clean={sd['clean']:.2f} UNSEEN4={sd['unseen4']:.3f} (bio={sd['by_variant']['bio_angry_brows']:.2f} vs vein={sd['by_variant']['anger_vein_on']:.2f}) | "
            f"OURS-1Pass(EN-gate->{lang}): Macro9={s1['macro9']:.3f} Clean={s1['clean']:.2f} UNSEEN4={s1['unseen4']:.3f} (p={pval:.5f})",
            flush=True,
        )

    for hk in hooks:
        hk.remove()
    out_p = args.out_dir / f"exp20b_trilingual_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "n_test_chars": len(te_chars), "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
