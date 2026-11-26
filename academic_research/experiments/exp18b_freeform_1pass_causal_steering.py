#!/usr/bin/env python3
"""
Exp 18B (P3 Frontier Test — Pure 1-Pass Mid-Layer-Gated Activation Steering During
         UNCONSTRAINED Free-Form Open-Ended Captioning, ZERO Prompt Changes, ZERO 2nd Pass!):

Tests the ultimate mechanistic interpretability challenge:
  Keep the open-ended caption prompt 100% UNCHANGED:
    "Describe this anime character's emotional state and mood in 1-2 sentences."
  (ZERO multiple-choice options, ZERO mention of anime tropes, ZERO Audit prompt, ZERO 2nd pass!)

  Strictly from train characters (fc_00..fc_35) x {orig, placebo} vs {forehead_shadow, sweat_drop}
  (ZERO unseen symbols!):
    1. Fit mid-layer anomaly gate g(h_mid) at [0.55L, 0.72L].
    2. Compute 1D normalized conflict-vs-clean steering vector v_l = normalize(mu_conf_l - mu_clean_l)
       at late layers [l_steer_start..l_steer_end] on the prefill prompt token.
    3. During a SINGLE model.generate(...) call (1.0x pass!):
       - On the prefill step, evaluate g(h_mid).
       - ONLY if g(h_mid) >= tau, inject (+alpha * ||h|| * v_l) at late layers during generation!
       - If g(h_mid) < tau (clean/placebo), inject 0.0 (100% untouched!).

  Evaluates on 12 held-out unseen chars (fc_36..fc_47) x 6 variants (including UNSEEN
  `eye_highlight_off` & `anger_vein_on` `💢`!) to test whether pure hidden-state steering
  spontaneously flips free-form prose from cheerful override -> hollow/uneasy/ominous!
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp17c_scale48_5fold_and_unified_1pass as e17c

OPEN_PROMPT = "Describe this anime character's emotional state and mood in 1-2 sentences."

EVAL_VARS = [
    ("orig", "cheerful"),
    ("placebo_hairpin", "cheerful"),
    ("eye_highlight_off", "ominous"),
    ("forehead_shadow_on", "ominous"),
    ("dead_eyes_and_shadow", "ominous"),
    ("sweat_drop_on", "uneasy"),
    ("anger_vein_on", "irritated"),
]

CHEER_KW = re.compile(r"\b(cheerful|happy|joyful|bright|positive|warm|carefree|content|delight|upbeat|friendly|pleasant|optimistic|sparkling)\b", re.I)
OMINOUS_KW = re.compile(r"\b(hollow|eerie|ominous|sinister|dark|empty|dead|lifeless|unsettling|creepy|gloomy|shadow|menacing|cold|vacant|blank|disturbing|uneasy|hiding|masking|forced|fake|nervous|anxious|tense|uncomfortable|troubled|upset|stressed|suspicious|sad|melancholy)\b", re.I)
UNEASY_KW = re.compile(r"\b(awkward|uneasy|nervous|anxious|sweat|uncomfortable|tense|flustered|embarrassed|worried|apprehensive|sheepish|hesitant|stressed)\b", re.I)
ANGRY_KW = re.compile(r"\b(angry|irritated|annoyed|frustrated|vein|mad|furious|tense|agitated|exasperated|suppressing|forced|uneasy)\b", re.I)


def score_caption(text: str, expected: str) -> tuple[bool, bool]:
    has_cheer = bool(CHEER_KW.search(text))
    has_om = bool(OMINOUS_KW.search(text))
    has_un = bool(UNEASY_KW.search(text))
    has_an = bool(ANGRY_KW.search(text))
    if expected == "cheerful":
        ok = has_cheer and not (has_om or has_un or has_an)
        return ok, has_cheer
    if expected == "ominous":
        return has_om, (has_cheer and not has_om)
    if expected == "uneasy":
        return (has_un or has_om), (has_cheer and not (has_un or has_om))
    return (has_an or has_om), (has_cheer and not (has_an or has_om))


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest48", type=Path, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--alphas", type=str, default="0.0,0.06,0.10,0.14")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]
    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    chars = sorted({r["char_id"] for r in recs})
    tr_chars = set(chars[:24])      # 24 train chars x 4 seen variants = 96 calibration imgs
    te_chars = sorted(chars[36:])   # 12 strictly unseen test chars

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()

    layers, _, _ = e17c._get_lm(model)
    nL = len(layers)
    l_mid = int(nL * 0.55)
    l_late_gate = int(nL * 0.72)
    steer_layers = list(range(int(nL * 0.65), int(nL * 0.92)))

    def encode(img: Image.Image):
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": OPEN_PROMPT}]}]
        t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        return processor(text=[t], images=[img], return_tensors="pt").to(model.device)

    # 1. Extract train hidden states at gate layers + steering layers (ONLY on OPEN_PROMPT, ZERO labels beyond seen2!)
    cap_all: dict[int, np.ndarray] = {}
    watch = sorted(set([l_mid, l_late_gate] + steer_layers))
    hooks = []
    for li in watch:
        def _mk(idx):
            def _fn(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                cap_all[idx] = h[0, -1, :].detach().float().cpu().numpy()
            return _fn
        hooks.append(layers[li].register_forward_hook(_mk(li)))

    X_gate_tr, y_tr = [], []
    H_clean_by_l: dict[int, list[np.ndarray]] = {li: [] for li in steer_layers}
    H_conf_by_l: dict[int, list[np.ndarray]] = {li: [] for li in steer_layers}

    for r in recs:
        if r["char_id"] in tr_chars and r["variant"] in e17c.SEEN_TRAIN_VARS:
            img = Image.open(r["image_path"]).convert("RGB")
            cap_all.clear()
            model(**encode(img))
            X_gate_tr.append(np.concatenate([cap_all[l_mid], cap_all[l_late_gate]], axis=0))
            is_conf = r["variant"] not in {"orig", "placebo_hairpin"}
            y_tr.append(1.0 if is_conf else 0.0)
            for li in steer_layers:
                (H_conf_by_l if is_conf else H_clean_by_l)[li].append(cap_all[li])

    for hk in hooks:
        hk.remove()

    X_gate_tr = np.stack(X_gate_tr, axis=0)
    y_tr = np.array(y_tr, dtype=np.float32)
    w_g, b_g = e17c._fit_logreg(X_gate_tr, y_tr)
    tau = e17c._cal_thresh(e17c._score(X_gate_tr, w_g, b_g), y_tr)

    # Normalized 1D conflict steering vector per steering layer
    steer_vecs: dict[int, torch.Tensor] = {}
    for li in steer_layers:
        diff = np.mean(H_conf_by_l[li], axis=0) - np.mean(H_clean_by_l[li], axis=0)
        diff = diff / (np.linalg.norm(diff) + 1e-8)
        steer_vecs[li] = torch.tensor(diff, device=model.device, dtype=torch.float32)

    # 2. Register TRUE 1-Pass Gate-and-Steer Hooks during model.generate()!
    state = {"prefill_done": False, "fired": False, "alpha": 0.0, "buf": {}}
    gen_hooks = []

    for li in (l_mid, l_late_gate):
        def _mk_g(idx):
            def _fn(_m, _i, out):
                if not state["prefill_done"]:
                    h = out[0] if isinstance(out, tuple) else out
                    if h.shape[1] > 1:  # prefill step!
                        state["buf"][idx] = h[0, -1, :].detach().float().cpu().numpy()
                        if idx == l_late_gate and l_mid in state["buf"]:
                            feat = np.concatenate([state["buf"][l_mid], state["buf"][l_late_gate]], axis=0)[None, :]
                            state["fired"] = bool(e17c._score(feat, w_g, b_g)[0] >= tau)
                            state["prefill_done"] = True
            return _fn
        gen_hooks.append(layers[li].register_forward_hook(_mk_g(li)))

    for li in steer_layers:
        def _mk_s(idx):
            def _fn(_m, _i, out):
                if state["alpha"] <= 0.0 or not state["fired"]:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                # Steer the active decoding token residual proportionally to its norm!
                h = h.clone()
                v = h[0, -1, :].float()
                nrm = v.norm()
                h[0, -1, :] = (v + state["alpha"] * nrm * steer_vecs[idx]).to(h.dtype)
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return _fn
        gen_hooks.append(layers[li].register_forward_hook(_mk_s(li)))

    by_cv = {(r["char_id"], r["variant"]): r["image_path"] for r in recs}
    results_by_alpha = {}

    for alpha in alphas:
        state["alpha"] = alpha
        summary_v = {}
        samples = {}
        for vname, exp_cat in EVAL_VARS:
            oks, cheers, fires = [], [], []
            sample_txt = ""
            for cid in te_chars:
                img = Image.open(by_cv[(cid, vname)]).convert("RGB")
                inp = encode(img)
                state["prefill_done"] = False
                state["fired"] = False
                state["buf"] = {}
                gen_out = model.generate(**inp, max_new_tokens=55, do_sample=False)
                txt = processor.decode(gen_out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                ok, ch = score_caption(txt, exp_cat)
                oks.append(int(ok))
                cheers.append(int(ch))
                fires.append(int(state["fired"]))
                if not sample_txt:
                    sample_txt = txt
            summary_v[vname] = {
                "acc": round(float(np.mean(oks)), 4),
                "cheerful_override": round(float(np.mean(cheers)), 4),
                "fire_rate": round(float(np.mean(fires)), 4),
            }
            samples[vname] = sample_txt
        clean_acc = float(np.mean([summary_v["orig"]["acc"], summary_v["placebo_hairpin"]["acc"]]))
        conf_acc = float(np.mean([summary_v[v]["acc"] for v, _ in EVAL_VARS if v not in {"orig", "placebo_hairpin"}]))
        results_by_alpha[f"alpha_{alpha:.2f}"] = {
            "clean_acc": round(clean_acc, 4),
            "conflict5_acc": round(conf_acc, 4),
            "macro7": round(float(np.mean([x["acc"] for x in summary_v.values()])), 4),
            "by_variant": summary_v,
            "sample_captions": samples,
        }
        print(
            f"[{Path(args.model_dir).name}] alpha={alpha:.2f} | Macro7={results_by_alpha[f'alpha_{alpha:.2f}']['macro7']:.3f} "
            f"Clean={clean_acc:.3f} Conf5={conf_acc:.3f} | "
            f"eye={summary_v['eye_highlight_off']['acc']:.2f} shd={summary_v['forehead_shadow_on']['acc']:.2f} "
            f"d+s={summary_v['dead_eyes_and_shadow']['acc']:.2f} swt={summary_v['sweat_drop_on']['acc']:.2f} vein={summary_v['anger_vein_on']['acc']:.2f}",
            flush=True,
        )

    for hk in gen_hooks:
        hk.remove()

    out_p = args.out_dir / f"exp18b_1pass_freeform_steer_{Path(args.model_dir).name}.json"
    out_p.write_text(json.dumps({"model": Path(args.model_dir).name, "n_test_chars": len(te_chars), "results": results_by_alpha}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
