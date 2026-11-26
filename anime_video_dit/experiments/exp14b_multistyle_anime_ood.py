#!/usr/bin/env python3
"""
Exp 14B (P3 Multi-Style & Unseen-Semiology Generalization — 4 Art Styles x 5 Tropes):
  Tests whether Biological-Override Binding Failure AND our Frozen Hidden-State
  Anomaly-Gated Binder (trained ONLY on modern TV-anime `forehead_shadow` + `sweat_drop`!)
  generalize ZERO-SHOT across 4 completely distinct 2D anime/manga art styles AND
  2 BRAND-NEW unseen non-biological anime symbols (`dark_aura`, `spiral_dizzy_eyes`):

  Art Styles (2 chars per style x 5 tropes = 40 images, 100% natively generated!):
    1. `retro90s`   : 1990s retro hand-painted cel anime style
    2. `bw_manga`   : Black-and-white Japanese manga ink & screentone (ZERO color!)
    3. `watercolor` : Soft pastel watercolor anime illustration
    4. `chibi_sd`   : Super-deformed 2-head-tall chibi anime style

  Tropes (5 options A..E):
    - `cheerful`    (`A`: genuinely bright happy smile)
    - `empty_eyes`  (`B`: dead hollow eyes + smile)
    - `dark_aura`   (`B`: BRAND-NEW menacing dark aura behind smiling character!)
    - `sweatdrop`   (`D`: awkward floating sweatdrop + smile)
    - `spiral_eyes` (`E`: BRAND-NEW cartoon `@_@` dizzy/confused spiral eyes + smile!)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

LABELS5 = ["A", "B", "C", "D", "E"]
EXPECTED_MS = {
    "cheerful": "A",
    "empty_eyes": "B",
    "dark_aura": "B",
    "sweatdrop": "D",
    "spiral_eyes": "E",
}
EXPECTED_SYNTH = {
    "orig": "A",
    "placebo_hairpin": "A",
    "forehead_shadow_on": "B",
    "sweat_drop_on": "D",
}

DIRECT5_MS = (
    "Look at this 2D anime or manga character portrait and judge their TRUE emotional / psychological state.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or radiating sinister/menacing intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "E) Dizzy, dazed, confused, or overwhelmed\n"
    "Answer with ONLY a single letter: A, B, C, D, or E."
)

AUDIT5_MS = (
    "Inspect this 2D anime or manga portrait carefully for ANY non-biological anime graphic affect symbols "
    "(such as empty/hollow eyes, menacing dark aura/shadow, cheek blush lines, floating sweat drop, or cartoon spiral dizzy eyes).\n"
    "In 2D anime/manga, non-biological graphic cues OVERRIDE a smiling mouth.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or radiating sinister/menacing intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "E) Dizzy, dazed, confused, or overwhelmed\n"
    "Answer with ONLY a single letter: A, B, C, D, or E."
)

STYLES = [
    ("retro90s", "1990s retro anime style, classic cel animation, vintage anime aesthetic, warm film grain"),
    ("bw_manga", "monochrome, greyscale, japanese manga page, black and white ink lineart, screentone shading, no color"),
    ("watercolor", "soft pastel watercolor anime illustration, delicate painterly washes, light dreamy palette"),
    ("chibi_sd", "chibi, super deformed cute anime character, oversized head, tiny body, clean anime style"),
]

TROPES = [
    ("cheerful", "cheerful bright smile, bright sparkling eyes with white highlights, looking at viewer, portrait, masterpiece, high score", "empty eyes, dark aura, sweatdrop, spiral eyes, angry, sad, 3d"),
    ("empty_eyes", "empty eyes, no eye highlights, dull lifeless hollow eyes, yandere gentle smile, looking at viewer, portrait, masterpiece, high score", "sparkling eyes, white highlights, angry frown, crying, 3d"),
    ("dark_aura", "menacing dark aura, dark shadowy flames and ominous gloom radiating behind character, unsettling calm smile, looking at viewer, portrait, masterpiece, high score", "frown, crying, angry open mouth, 3d"),
    ("sweatdrop", "anime sweatdrop on temple, awkward nervous smile, looking at viewer, portrait, masterpiece, high score", "crying, angry, dark aura, empty eyes, 3d"),
    ("spiral_eyes", "spiral eyes, dizzy swirling cartoon eyes, dazed confused smile, looking at viewer, portrait, masterpiece, high score", "normal pupils, crying, angry, 3d"),
]


def generate_multistyle(animagine_dir: str, out_dir: Path) -> list[dict[str, Any]]:
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    img_dir = out_dir / "images"
    sheet_dir = out_dir / "sheets"
    img_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(animagine_dir, torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    records = []
    for sidx, (sname, stags) in enumerate(STYLES):
        for cidx, cdesc in enumerate(["1girl, solo, bob hair, school collar", "1boy, solo, messy hair, jacket"]):
            panels = []
            for tidx, (tname, pos, neg) in enumerate(TROPES):
                gen = torch.Generator(device="cuda").manual_seed(5500 + sidx * 300 + cidx * 70 + tidx * 11)
                img = pipe(
                    prompt=f"{cdesc}, {stags}, {pos}",
                    negative_prompt=neg,
                    width=832,
                    height=832,
                    guidance_scale=5.5,
                    num_inference_steps=26,
                    generator=gen,
                ).images[0]
                cid = f"{sname}_c{cidx}"
                p = img_dir / f"{cid}__{tname}.png"
                img.save(p)
                records.append({"char_id": cid, "style": sname, "variant": tname, "gt": EXPECTED_MS[tname], "image_path": str(p)})
                t = img.resize((224, 224), Image.LANCZOS)
                ImageDraw.Draw(t).rectangle([0, 0, 224, 22], fill=(20, 20, 28))
                ImageDraw.Draw(t).text((5, 4), f"{sname}:{tname}", fill=(255, 255, 255))
                panels.append(t)
            sheet = Image.new("RGB", (224 * 5, 224))
            for i, t in enumerate(panels):
                sheet.paste(t, (i * 224, 0))
            sheet.save(sheet_dir / f"{sname}_c{cidx}__sheet.jpg", quality=90)
        print(f"Generated multi-style suite for style={sname}", flush=True)

    man = out_dir / "manifest_multistyle.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return records


def _get_lm_layers(model):
    for path in [("model", "language_model", "layers"), ("language_model", "layers"), ("model", "layers")]:
        cur = model
        ok = True
        for a in path:
            if hasattr(cur, a):
                cur = getattr(cur, a)
            else:
                ok = False
                break
        if ok:
            return cur
    raise AttributeError("Cannot locate LM layers")


def _fit_logreg(X: np.ndarray, y: np.ndarray, reg: float = 5.0, steps: int = 400, lr: float = 0.05):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    w = np.zeros(Xn.shape[1], dtype=np.float32)
    b = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
        err = p - y
        w -= lr * ((Xn.T @ err) / len(y) + reg * w / len(y))
        b -= lr * float(np.mean(err))
    return w, b


def _score(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return 1.0 / (1.0 + np.exp(-np.clip(Xn @ w + b, -30, 30)))


def _cal_thresh(sc: np.ndarray, y: np.ndarray) -> float:
    best_t, best_b = 0.5, -1.0
    for t in np.linspace(float(sc.min()) - 0.01, float(sc.max()) + 0.01, 101):
        p = (sc >= t).astype(np.float32)
        bal = 0.5 * float(np.mean(p[y == 0] == 0)) + 0.5 * float(np.mean(p[y == 1] == 1))
        if bal > best_b:
            best_b, best_t = bal, float(t)
    return best_t


@torch.no_grad()
def eval_multistyle(synth_manifest: Path, ms_recs: list[dict[str, Any]], model_dir: str, out_dir: Path) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    abcde_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in LABELS5]

    layers = _get_lm_layers(model)
    l_mid, l_late = int(len(layers) * 0.55), int(len(layers) * 0.78)
    cap: dict[int, np.ndarray] = {}

    def mk(idx):
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            cap[idx] = h[0, -1, :].detach().float().cpu().numpy()
        return fn

    hs = [layers[l_mid].register_forward_hook(mk(l_mid)), layers[l_late].register_forward_hook(mk(l_late))]

    def run_set(recs: list[dict[str, Any]]):
        dlg, alg, fd = [], [], []
        for r in recs:
            img = Image.open(r["image_path"]).convert("RGB")
            m_d = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DIRECT5_MS}]}]
            t_d = processor.apply_chat_template(m_d, tokenize=False, add_generation_prompt=True)
            in_d = processor(text=[t_d], images=[img], return_tensors="pt").to(model.device)
            dlg.append(model(**in_d).logits[0, -1, abcde_ids].float().cpu().numpy())
            fd.append(np.concatenate([cap[l_mid], cap[l_late]], axis=0))

            m_a = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": AUDIT5_MS}]}]
            t_a = processor.apply_chat_template(m_a, tokenize=False, add_generation_prompt=True)
            in_a = processor(text=[t_a], images=[img], return_tensors="pt").to(model.device)
            alg.append(model(**in_a).logits[0, -1, abcde_ids].float().cpu().numpy())
        return np.stack(dlg), np.stack(alg), np.stack(fd)

    try:
        synth_all = [json.loads(l) for l in synth_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        tr_chars = sorted({r["char_id"] for r in synth_all})[:10]
        synth_tr = [r for r in synth_all if r["char_id"] in tr_chars and r["variant"] in EXPECTED_SYNTH]
        y_tr = np.array([0.0 if EXPECTED_SYNTH[r["variant"]] == "A" else 1.0 for r in synth_tr], dtype=np.float32)

        tr_dlg, tr_alg, tr_fd = run_set(synth_tr)
        ms_dlg, ms_alg, ms_fd = run_set(ms_recs)
    finally:
        for h in hs:
            h.remove()

    # Fit probe & clean prior ONLY on modern TV-anime synthetic train data!
    clean_mask = y_tr == 0
    mu_clean_d = tr_dlg[clean_mask].mean(axis=0)
    mu_clean_a = tr_alg[clean_mask].mean(axis=0)
    w, b = _fit_logreg(tr_fd, y_tr, reg=5.0)
    tau = _cal_thresh(_score(tr_fd, w, b), y_tr)

    flags = _score(ms_fd, w, b) >= tau
    preds_probe = []
    for i, flg in enumerate(flags):
        if not flg:
            preds_probe.append("A")
        else:
            sc = (ms_dlg[i] - mu_clean_d)[1:] + (ms_alg[i] - mu_clean_a)[1:]
            preds_probe.append(LABELS5[1 + int(np.argmax(sc))])

    preds_dir = [LABELS5[int(np.argmax(lg))] for lg in ms_dlg]
    preds_aud = [LABELS5[int(np.argmax(lg))] for lg in ms_alg]

    def summarize(preds: list[str]) -> dict[str, Any]:
        by_s: dict[str, list[bool]] = {}
        by_t: dict[str, list[bool]] = {}
        for r, p in zip(ms_recs, preds):
            ok = p == EXPECTED_MS[r["variant"]]
            by_s.setdefault(r["style"], []).append(ok)
            by_t.setdefault(r["variant"], []).append(ok)
        return {
            "overall": round(float(np.mean([p == EXPECTED_MS[r["variant"]] for r, p in zip(ms_recs, preds)])), 4),
            "by_style": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_s.items())},
            "by_trope": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_t.items())},
        }

    res = {
        "model": Path(model_dir).name,
        "direct": summarize(preds_dir),
        "audit": summarize(preds_aud),
        "synth_tv_to_multistyle_probe": summarize(preds_probe),
    }
    out_path = out_dir / f"exp14b_multistyle_{Path(model_dir).name}.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"=== EXP 14B MULTI-STYLE (4 STYLES x 5 TROPES): {res['model']} ===")
    for k in ["direct", "audit", "synth_tv_to_multistyle_probe"]:
        s = res[k]
        print(f"  {k:30s} overall={s['overall']:.3f} | styles={s['by_style']} | tropes={s['by_trope']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animagine-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/animagine-xl-4.0")
    ap.add_argument("--synth-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--eval-model", type=str, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    man = args.out_dir / "manifest_multistyle.jsonl"
    if args.generate or not man.exists():
        ms_recs = generate_multistyle(args.animagine_dir, args.out_dir)
    else:
        ms_recs = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.eval_model:
        eval_multistyle(args.synth_manifest, ms_recs, args.eval_model, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
