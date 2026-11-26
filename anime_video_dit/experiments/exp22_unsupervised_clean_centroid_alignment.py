#!/usr/bin/env python3
"""
Exp 22 (P3 Unsupervised Clean-Centroid Domain Alignment — Fixing Zero-Shot Cross-Lingual,
        Cross-Style, and Cross-Generator Threshold Drift with ZERO Target Conflict Labels!):

Mechanistic insight from Exp 20B & Exp 14B:
  The frozen mid-layer non-biological anomaly direction `w_src` trained ONLY on English
  Animagine-XL synthetic counterfactuals (`forehead_shadow`, `sweat_drop`) is ALREADY
  aligned across languages (JA, ZH) and styles (B&W manga, watercolor, Illustrious-XL),
  but changing prompt language or art style adds a constant residual shift `Delta_mu_domain`
  that shifts the scalar gate threshold `tau` (causing e.g. Qwen3-32B EN->JA clean acc to drop!).

2-Line Unsupervised Clean-Centroid Alignment (ZERO target conflict labels!):
  - During source training (EN Animagine-XL): center features by source clean centroid `mu_clean_src`
    before L2 normalization and logistic regression: `z(x) = normalize(h(x) - mu_clean_src)`.
  - At target domain deployment (JA, ZH, Multi-Style, Illustrious-XL): estimate ONLY `mu_clean_tgt`
    from a handful of unlabeled clean portraits in the target domain (leave-one-out on small sets!)
    and center: `z_tgt(x) = normalize(h(x) - mu_clean_tgt)`, then apply the FROZEN `(w_src, b_src, tau_src)`!

Directly compares:
  1. `Direct` (baseline)
  2. `1-Pass Unaligned (Raw EN Gate -> Target)`
  3. `1-Pass Clean-Centroid Aligned (OURS — ZERO target conflict labels!)`
across:
  - Trilingual EN / JA / ZH (24 unseen chars x 9 variants = 216 imgs/lang)
  - 4-Style Multi-Domain (`bw_manga`, `retro90s`, `chibi_sd`, `watercolor`, 40 imgs)
  - Cross-Generator (`Illustrious-XL-v2.0`, 16 imgs)
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
import exp20b_trilingual_en_ja_zh_and_xling_transfer as e20b


def _center_norm(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    Xc = X - mu[None, :]
    return Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8)


def _fit_centered_logreg(X: np.ndarray, y: np.ndarray, mu_clean: np.ndarray, reg: float = 5.0):
    Xn = _center_norm(X, mu_clean)
    w = np.zeros(Xn.shape[1], dtype=np.float32)
    b = 0.0
    for _ in range(400):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
        err = p - y
        w -= 0.05 * ((Xn.T @ err) / len(y) + reg * w / len(y))
        b -= 0.05 * float(np.mean(err))
    sc = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
    tau = e17c._cal_thresh(sc, y)
    return w, b, tau


def _score_centered(X: np.ndarray, mu_clean: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    Xn = _center_norm(X, mu_clean)
    return 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -30, 30)))


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest48", type=Path, required=True)
    ap.add_argument("--manifest-illustrious", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs_ill = [json.loads(l) for l in args.manifest_illustrious.read_text(encoding="utf-8").splitlines() if l.strip()]

    chars = sorted({r["char_id"] for r in recs})
    tr_chars = set(chars[:24])
    te_chars = set(chars[24:])
    tr_recs = [r for r in recs if r["char_id"] in tr_chars and r["variant"] in e17c.SEEN_TRAIN_VARS]
    te_recs = [r for r in recs if r["char_id"] in te_chars]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ids5 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS5]
    ids4 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS4]

    layers, final_norm, lm_head = e17c._get_lm(model)
    nL = len(layers)
    l_mid = int(nL * 0.55)
    l_late_gate = int(nL * 0.72)
    l_surg = l_late_gate + 1

    def mk_wbio(ids):
        W = lm_head.weight[ids].detach().float()
        wb = W[0] - W[1:].mean(dim=0)
        if final_norm is not None and hasattr(final_norm, "weight"):
            wb = wb * final_norm.weight.detach().float()
        return (wb / (wb.norm() + 1e-8)).to(model.device)

    w_bio5 = mk_wbio(ids5)
    w_bio4 = mk_wbio(ids4)

    cap: dict[int, np.ndarray] = {}
    surg_state = {"on": False, "wbio": w_bio5}
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
                if not surg_state["on"]:
                    return out
                wb = surg_state["wbio"]
                h = out[0] if isinstance(out, tuple) else out
                v = h[0, -1, :].float()
                proj = torch.dot(v, wb)
                if proj.item() > 0.0:
                    h = h.clone()
                    h[0, -1, :] = (v - 1.5 * proj * wb).to(h.dtype)
                    return (h,) + out[1:] if isinstance(out, tuple) else h
                return out
            return _fn
        hooks.append(layers[li].self_attn.register_forward_hook(_mk_s()))
        hooks.append(layers[li].mlp.register_forward_hook(_mk_s()))

    def encode(img: Image.Image, prompt: str):
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        return processor(text=[t], images=[img], return_tensors="pt").to(model.device)

    # 1. Fit BOTH Raw EN gate AND Clean-Centroid-Aligned EN gate strictly on EN train chars (fc_00..fc_23)!
    X_en_tr, y_en_tr = [], []
    for r in tr_recs:
        img = Image.open(r["image_path"]).convert("RGB")
        cap.clear(); surg_state["on"] = False
        model(**encode(img, e20b.PROMPTS["EN"]))
        X_en_tr.append(np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0))
        y_en_tr.append(0.0 if r["variant"] in {"orig", "placebo_hairpin"} else 1.0)
    X_en_tr = np.stack(X_en_tr, axis=0)
    y_en_tr = np.array(y_en_tr, dtype=np.float32)

    mu_clean_en = np.mean(X_en_tr[y_en_tr == 0], axis=0)
    w_raw, b_raw = e17c._fit_logreg(X_en_tr, y_en_tr)
    tau_raw = e17c._cal_thresh(e17c._score(X_en_tr, w_raw, b_raw), y_en_tr)
    w_ctr, b_ctr, tau_ctr = _fit_centered_logreg(X_en_tr, y_en_tr, mu_clean_en)

    # 2. Compute target-language clean centroids + clean logit baselines using ONLY 12 clean TRAIN chars (fc_00..fc_11, ZERO conflict labels, ZERO test chars!)
    clean_tr_paths = [r["image_path"] for r in tr_recs if r["variant"] == "orig"][:12]
    mu_clean_lang = {}
    mu_surg_lang = {}
    for lang, prm in e20b.PROMPTS.items():
        hs, lgs = [], []
        for cp in clean_tr_paths:
            img = Image.open(cp).convert("RGB")
            inp = encode(img, prm)
            cap.clear(); surg_state["on"] = False
            model(**inp)
            hs.append(np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0))
            surg_state["on"] = True; surg_state["wbio"] = w_bio5
            lgs.append(model(**inp).logits[0, -1, ids5].float().cpu().numpy())
        mu_clean_lang[lang] = np.mean(hs, axis=0)
        mu_surg_lang[lang] = np.mean(lgs, axis=0)

    # 3. Evaluate Trilingual EN / JA / ZH on 24 strictly unseen test chars (216 imgs/lang)
    trilingual_res = {}
    for lang, prm in e20b.PROMPTS.items():
        ok_d, ok_raw, ok_ctr = {v: [] for v in e17c.EXPECTED9}, {v: [] for v in e17c.EXPECTED9}, {v: [] for v in e17c.EXPECTED9}
        all_d, all_raw, all_ctr = [], [], []
        for r in te_recs:
            img = Image.open(r["image_path"]).convert("RGB")
            inp = encode(img, prm)
            cap.clear(); surg_state["on"] = False
            lg_d = model(**inp).logits[0, -1, ids5].float().cpu().numpy()
            h = np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0)[None, :]
            surg_state["on"] = True; surg_state["wbio"] = w_bio5
            lg_s = model(**inp).logits[0, -1, ids5].float().cpu().numpy()

            f_raw = bool(e17c._score(h, w_raw, b_raw)[0] >= tau_raw)
            f_ctr = bool(_score_centered(h, mu_clean_lang[lang], w_ctr, b_ctr)[0] >= tau_ctr)

            exp_l = e17c.EXPECTED9[r["variant"]]
            pd = e17c.LABELS5[int(np.argmax(lg_d))]
            p_raw = "A" if not f_raw else e17c.LABELS5[1 + int(np.argmax((lg_s - mu_surg_lang[lang])[1:]))]
            p_ctr = "A" if not f_ctr else e17c.LABELS5[1 + int(np.argmax((lg_s - mu_surg_lang[lang])[1:]))]

            cd, cr, cc = int(pd == exp_l), int(p_raw == exp_l), int(p_ctr == exp_l)
            ok_d[r["variant"]].append(cd); ok_raw[r["variant"]].append(cr); ok_ctr[r["variant"]].append(cc)
            all_d.append(cd); all_raw.append(cr); all_ctr.append(cc)

        def summ(dm):
            bv = {v: round(float(np.mean(xs)), 4) for v, xs in dm.items()}
            return {
                "macro9": round(float(np.mean(list(bv.values()))), 4),
                "clean": round(float(np.mean([bv["orig"], bv["placebo_hairpin"]])), 4),
                "unseen4": round(float(np.mean([bv[v] for v in e17c.UNSEEN4])), 4),
            }

        sd, sr, sc = summ(ok_d), summ(ok_raw), summ(ok_ctr)
        trilingual_res[lang] = {
            "direct": sd,
            "raw_1pass": sr,
            "centered_1pass": sc,
            "p_ctr_vs_direct": round(e17c._mcnemar_exact_p(all_d, all_ctr), 6),
            "p_ctr_vs_raw": round(e17c._mcnemar_exact_p(all_raw, all_ctr), 6),
        }
        print(
            f"[{Path(args.model_dir).name}] {lang} | Direct={sd['macro9']:.3f}(cl={sd['clean']:.2f},U4={sd['unseen4']:.3f}) | "
            f"Raw-1P={sr['macro9']:.3f}(cl={sr['clean']:.2f},U4={sr['unseen4']:.3f}) | "
            f"ALIGNED-1P={sc['macro9']:.3f}(cl={sc['clean']:.2f},U4={sc['unseen4']:.3f}, p_vs_dir={trilingual_res[lang]['p_ctr_vs_direct']:.5f})",
            flush=True,
        )

    # 4. Evaluate Cross-Generator Illustrious-XL-v2.0 (leave-one-out clean centroid over the 4 clean Illustrious imgs — zero conflict labels!)
    ill_feats, ill_dir_lg, ill_surg_lg = [], [], []
    for r in recs_ill:
        img = Image.open(r["image_path"]).convert("RGB")
        inp4 = encode(img, e17c.DIRECT4_PROMPT)
        cap.clear(); surg_state["on"] = False
        ill_dir_lg.append(model(**inp4).logits[0, -1, ids4].float().cpu().numpy())
        ill_feats.append(np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0))
        surg_state["on"] = True; surg_state["wbio"] = w_bio4
        ill_surg_lg.append(model(**inp4).logits[0, -1, ids4].float().cpu().numpy())

    ill_feats = np.stack(ill_feats, axis=0)
    clean_ill_idx = [i for i, r in enumerate(recs_ill) if r["variant"] == "native_cheerful"]
    # Also compute EN Animagine 4-opt clean surg mean
    mu4_surg_en = np.mean([model(**encode(Image.open(cp).convert("RGB"), e17c.DIRECT4_PROMPT)).logits[0, -1, ids4].float().cpu().numpy() for cp in clean_tr_paths], axis=0)

    ill_d, ill_r, ill_c = [], [], []
    for i, r in enumerate(recs_ill):
        # Leave-one-out clean centroid if this image itself is clean (strict zero leakage!)
        c_pool = [j for j in clean_ill_idx if j != i]
        mu_ill_clean = np.mean(ill_feats[c_pool], axis=0)
        mu_ill_surg = np.mean([ill_surg_lg[j] for j in c_pool], axis=0)

        h = ill_feats[i:i+1]
        f_raw = bool(e17c._score(h, w_raw, b_raw)[0] >= tau_raw)
        f_ctr = bool(_score_centered(h, mu_ill_clean, w_ctr, b_ctr)[0] >= tau_ctr)
        exp_l = r["expected"]
        pd = e17c.LABELS4[int(np.argmax(ill_dir_lg[i]))]
        pr = "A" if not f_raw else e17c.LABELS4[1 + int(np.argmax((ill_surg_lg[i] - mu4_surg_en)[1:]))]
        pc = "A" if not f_ctr else e17c.LABELS4[1 + int(np.argmax((ill_surg_lg[i] - mu_ill_surg)[1:]))]
        ill_d.append((r["variant"], pd == exp_l))
        ill_r.append((r["variant"], pr == exp_l))
        ill_c.append((r["variant"], pc == exp_l))

    def summ_ill(pairs):
        bv = {v: round(float(np.mean([c for vv, c in pairs if vv == v])), 4) for v in e17c.EXPECTED_NAT}
        return {"macro4": round(float(np.mean(list(bv.values()))), 4), "clean": bv["native_cheerful"], "by_variant": bv}

    ill_res = {"direct": summ_ill(ill_d), "raw_1pass": summ_ill(ill_r), "centered_1pass": summ_ill(ill_c)}
    print(f"[{Path(args.model_dir).name}] Illustrious-v2 | Direct={ill_res['direct']['macro4']:.3f}(cl={ill_res['direct']['clean']:.2f}) | Raw-1P={ill_res['raw_1pass']['macro4']:.3f}(cl={ill_res['raw_1pass']['clean']:.2f}) | ALIGNED-1P={ill_res['centered_1pass']['macro4']:.3f}(cl={ill_res['centered_1pass']['clean']:.2f})", flush=True)

    for hk in hooks:
        hk.remove()

    out_p = args.out_dir / f"exp22_aligned_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "trilingual": trilingual_res, "illustrious_v2": ill_res}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
