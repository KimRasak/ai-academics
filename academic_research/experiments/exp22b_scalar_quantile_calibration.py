#!/usr/bin/env python3
"""
Exp 22B (P3 1D Scalar Clean-Shift Calibration — Fixing Cross-Lingual & Cross-Generator
         Threshold Drift WITHOUT Distorting the L2-Normalized Clean Cone!):

Follow-up ablation directly motivated by Exp 22's honest negative result:
  - Exp 22 showed that subtracting a 4096D clean centroid BEFORE L2 normalization blows up
    character-identity noise around the origin and degrades gate accuracy.
  - Instead, keep the exact L2-normalized residual `h / ||h||` and FROZEN EN gate `(w_en, b_en)`,
    and shift ONLY the 1D scalar decision threshold using 12 unlabeled clean target faces:
      `tau_tgt = tau_en + (mean(logit_score(clean_tgt)) - mean(logit_score(clean_en)))`
    (ZERO target conflict labels, ZERO feature distortion, 1 single scalar parameter!)
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


def _raw_logit_score(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return Xn @ w + b


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
    tr_chars, te_chars = set(chars[:24]), set(chars[24:])
    tr_recs = [r for r in recs if r["char_id"] in tr_chars and r["variant"] in e17c.SEEN_TRAIN_VARS]
    te_recs = [r for r in recs if r["char_id"] in te_chars]

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    ids5 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS5]
    ids4 = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in e17c.LABELS4]

    layers, final_norm, lm_head = e17c._get_lm(model)
    nL = len(layers)
    l_mid, l_late_gate = int(nL * 0.55), int(nL * 0.72)
    l_surg = l_late_gate + 1

    def mk_wbio(ids):
        W = lm_head.weight[ids].detach().float()
        wb = W[0] - W[1:].mean(dim=0)
        if final_norm is not None and hasattr(final_norm, "weight"):
            wb = wb * final_norm.weight.detach().float()
        return (wb / (wb.norm() + 1e-8)).to(model.device)

    w_bio5, w_bio4 = mk_wbio(ids5), mk_wbio(ids4)
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

    # 1. Fit FROZEN EN gate in raw L2-norm space
    X_tr, y_tr = [], []
    for r in tr_recs:
        cap.clear(); surg_state["on"] = False
        model(**encode(Image.open(r["image_path"]).convert("RGB"), e20b.PROMPTS["EN"]))
        X_tr.append(np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0))
        y_tr.append(0.0 if r["variant"] in {"orig", "placebo_hairpin"} else 1.0)
    X_tr = np.stack(X_tr, axis=0)
    y_tr = np.array(y_tr, dtype=np.float32)
    w_en, b_en = e17c._fit_logreg(X_tr, y_tr)
    z_tr = _raw_logit_score(X_tr, w_en, b_en)
    tau_en = e17c._cal_thresh(z_tr, y_tr)
    mu_z_clean_en = float(np.mean(z_tr[y_tr == 0]))

    # 2. Estimate ONLY 1D scalar clean shift + clean logit baseline from 12 clean train faces per language
    clean_tr_paths = [r["image_path"] for r in tr_recs if r["variant"] == "orig"][:12]
    tau_lang, mu_surg_lang = {}, {}
    for lang, prm in e20b.PROMPTS.items():
        zs, lgs = [], []
        for cp in clean_tr_paths:
            img = Image.open(cp).convert("RGB")
            inp = encode(img, prm)
            cap.clear(); surg_state["on"] = False
            model(**inp)
            h = np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0)[None, :]
            zs.append(float(_raw_logit_score(h, w_en, b_en)[0]))
            surg_state["on"] = True; surg_state["wbio"] = w_bio5
            lgs.append(model(**inp).logits[0, -1, ids5].float().cpu().numpy())
        tau_lang[lang] = tau_en + (float(np.mean(zs)) - mu_z_clean_en)
        mu_surg_lang[lang] = np.mean(lgs, axis=0)

    # 3. Evaluate Trilingual EN / JA / ZH on 24 strictly unseen test chars (216 imgs/lang)
    tri_res = {}
    for lang, prm in e20b.PROMPTS.items():
        ok_d, ok_r, ok_s = {v: [] for v in e17c.EXPECTED9}, {v: [] for v in e17c.EXPECTED9}, {v: [] for v in e17c.EXPECTED9}
        all_d, all_r, all_s = [], [], []
        for r in te_recs:
            img = Image.open(r["image_path"]).convert("RGB")
            inp = encode(img, prm)
            cap.clear(); surg_state["on"] = False
            lg_d = model(**inp).logits[0, -1, ids5].float().cpu().numpy()
            h = np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0)[None, :]
            z = float(_raw_logit_score(h, w_en, b_en)[0])
            surg_state["on"] = True; surg_state["wbio"] = w_bio5
            lg_s = model(**inp).logits[0, -1, ids5].float().cpu().numpy()

            exp_l = e17c.EXPECTED9[r["variant"]]
            pd = e17c.LABELS5[int(np.argmax(lg_d))]
            pr = "A" if z < tau_en else e17c.LABELS5[1 + int(np.argmax((lg_s - mu_surg_lang[lang])[1:]))]
            ps = "A" if z < tau_lang[lang] else e17c.LABELS5[1 + int(np.argmax((lg_s - mu_surg_lang[lang])[1:]))]
            cd, cr, cs = int(pd == exp_l), int(pr == exp_l), int(ps == exp_l)
            ok_d[r["variant"]].append(cd); ok_r[r["variant"]].append(cr); ok_s[r["variant"]].append(cs)
            all_d.append(cd); all_r.append(cr); all_s.append(cs)

        def summ(dm):
            bv = {v: round(float(np.mean(xs)), 4) for v, xs in dm.items()}
            return {"macro9": round(float(np.mean(list(bv.values()))), 4), "clean": round(float(np.mean([bv["orig"], bv["placebo_hairpin"]])), 4), "unseen4": round(float(np.mean([bv[v] for v in e17c.UNSEEN4])), 4)}

        sd, sr, ss = summ(ok_d), summ(ok_r), summ(ok_s)
        tri_res[lang] = {
            "direct": sd,
            "raw_tau_en": sr,
            "scalar_shifted_1pass": ss,
            "p_shift_vs_dir": round(e17c._mcnemar_exact_p(all_d, all_s), 6),
            "p_shift_vs_raw": round(e17c._mcnemar_exact_p(all_r, all_s), 6),
        }
        print(
            f"[{Path(args.model_dir).name}] {lang} | Dir={sd['macro9']:.3f}(cl={sd['clean']:.2f}) | Raw={sr['macro9']:.3f}(cl={sr['clean']:.2f}) | "
            f"SCALAR-SHIFT-1P={ss['macro9']:.3f}(cl={ss['clean']:.2f},U4={ss['unseen4']:.3f}, p_dir={tri_res[lang]['p_shift_vs_dir']:.5f}, p_raw={tri_res[lang]['p_shift_vs_raw']:.5f})",
            flush=True,
        )

    # 4. Illustrious-XL-v2.0 with leave-one-out 1D scalar clean shift
    ill_z, ill_d_lg, ill_s_lg = [], [], []
    for r in recs_ill:
        img = Image.open(r["image_path"]).convert("RGB")
        inp4 = encode(img, e17c.DIRECT4_PROMPT)
        cap.clear(); surg_state["on"] = False
        ill_d_lg.append(model(**inp4).logits[0, -1, ids4].float().cpu().numpy())
        h = np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0)[None, :]
        ill_z.append(float(_raw_logit_score(h, w_en, b_en)[0]))
        surg_state["on"] = True; surg_state["wbio"] = w_bio4
        ill_s_lg.append(model(**inp4).logits[0, -1, ids4].float().cpu().numpy())

    c_ill = [i for i, r in enumerate(recs_ill) if r["variant"] == "native_cheerful"]
    # Need clean EN 4-opt score mean
    z_clean4_en = []
    lg_clean4_en = []
    for cp in clean_tr_paths:
        img = Image.open(cp).convert("RGB")
        inp4 = encode(img, e17c.DIRECT4_PROMPT)
        cap.clear(); surg_state["on"] = False
        model(**inp4)
        z_clean4_en.append(float(_raw_logit_score(np.concatenate([cap[l_mid], cap[l_late_gate]], axis=0)[None, :], w_en, b_en)[0]))
        surg_state["on"] = True; surg_state["wbio"] = w_bio4
        lg_clean4_en.append(model(**inp4).logits[0, -1, ids4].float().cpu().numpy())
    mu_z4_en = float(np.mean(z_clean4_en))
    mu_lg4_en = np.mean(lg_clean4_en, axis=0)

    ill_d, ill_r, ill_s = [], [], []
    for i, r in enumerate(recs_ill):
        pool = [j for j in c_ill if j != i]
        tau_ill = tau_en + (float(np.mean([ill_z[j] for j in pool])) - mu_z4_en)
        mu_lg_ill = np.mean([ill_s_lg[j] for j in pool], axis=0)
        exp_l = r["expected"]
        pd = e17c.LABELS4[int(np.argmax(ill_d_lg[i]))]
        pr = "A" if ill_z[i] < tau_en else e17c.LABELS4[1 + int(np.argmax((ill_s_lg[i] - mu_lg4_en)[1:]))]
        ps = "A" if ill_z[i] < tau_ill else e17c.LABELS4[1 + int(np.argmax((ill_s_lg[i] - mu_lg_ill)[1:]))]
        ill_d.append((r["variant"], pd == exp_l))
        ill_r.append((r["variant"], pr == exp_l))
        ill_s.append((r["variant"], ps == exp_l))

    def s_ill(pairs):
        bv = {v: round(float(np.mean([c for vv, c in pairs if vv == v])), 4) for v in e17c.EXPECTED_NAT}
        return {"macro4": round(float(np.mean(list(bv.values()))), 4), "clean": bv["native_cheerful"], "by_variant": bv}

    ill_res = {"direct": s_ill(ill_d), "raw_tau_en": s_ill(ill_r), "scalar_shifted_1pass": s_ill(ill_s)}
    print(f"[{Path(args.model_dir).name}] Illustrious-v2 | Dir={ill_res['direct']['macro4']:.3f}(cl={ill_res['direct']['clean']:.2f}) | Raw={ill_res['raw_tau_en']['macro4']:.3f}(cl={ill_res['raw_tau_en']['clean']:.2f}) | SCALAR-SHIFT-1P={ill_res['scalar_shifted_1pass']['macro4']:.3f}(cl={ill_res['scalar_shifted_1pass']['clean']:.2f})", flush=True)

    for hk in hooks:
        hk.remove()
    out_p = args.out_dir / f"exp22b_scalar_shift_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "trilingual": tri_res, "illustrious_v2": ill_res}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
