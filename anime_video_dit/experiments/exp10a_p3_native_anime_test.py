#!/usr/bin/env python3
"""
Exp 10A (P3 Real-Distribution / Native End-to-End Anime Generalization Test):
  Answers the #1 external-validity question:
    "Does Biological-Override Binding Failure happen on 100% NATIVELY GENERATED
     end-to-end 2D anime illustrations (ZERO programmatic overlays!), AND does our
     Frozen Hidden-State Anomaly-Gated Binder trained ONLY on synthetic counterfactuals
     transfer ZERO-SHOT to native end-to-end anime illustrations?"

  Step 1 (--generate): Uses Animagine-XL-4.0 with native Danbooru anime tags
    (ZERO pixel overlays!) to generate 8 characters x 4 native anime tropes = 32 images:
      - `native_cheerful` (`A`: bright detailed eyes, cheerful smile)
      - `native_empty_eyes_smile` (`B`: empty eyes, expressionless dead eyes, gentle smile, yandere)
      - `native_shaded_face_smile` (`B`: shaded face, shadow over eyes, ominous smile)
      - `native_sweatdrop_smile` (`D`: sweatdrop, awkward smile)

  Step 2 (--eval-model <model_dir>):
    Evaluates Direct, Audit, AND Zero-Shot Cross-Domain Transfer of our Frozen
    Hidden-State Anomaly Probe trained ONLY on `exp2_p3_v2` synthetic counterfactuals
    (10 synthetic chars x {orig, placebo, forehead_shadow, sweat_drop}) -> tested
    ZERO-SHOT on the 32 native end-to-end Animagine-XL-4.0 images!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

LABELS = ["A", "B", "C", "D"]
EXPECTED_NATIVE = {
    "native_cheerful": "A",
    "native_empty_eyes_smile": "B",
    "native_shaded_face_smile": "B",
    "native_sweatdrop_smile": "D",
}
EXPECTED_SYNTH = {
    "orig": "A",
    "placebo_hairpin": "A",
    "forehead_shadow_on": "B",
    "sweat_drop_on": "D",
}

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
    "(such as unusual dark shadows over the forehead/eyes, missing/hollow/empty eye highlights, "
    "cheek blush marks, or floating sweat drops), ignoring ordinary hair accessories.\n"
    "In 2D anime, non-biological graphic cues OVERRIDE a smiling mouth.\n"
    "Choose the SINGLE best label:\n"
    "A) Genuinely cheerful, bright, and happily carefree\n"
    "B) Emotionally hollow, eerie, ominous, or hiding sinister/dark intent\n"
    "C) Shy, flustered, or bashfully embarrassed\n"
    "D) Awkward, uneasy, or nervously sweating\n"
    "Answer with ONLY a single letter: A, B, C, or D."
)

NATIVE_CHARS = [
    ("nat_00_blue", "1girl, solo, short blue hair, school uniform, upper body portrait, front view, looking at viewer"),
    ("nat_01_pink", "1girl, solo, pink twintails hair, white blouse, upper body portrait, front view, looking at viewer"),
    ("nat_02_blonde", "1girl, solo, long golden blonde hair, cardigan, upper body portrait, front view, looking at viewer"),
    ("nat_03_silver", "1boy, solo, messy silver hair, jacket, upper body portrait, front view, looking at viewer"),
    ("nat_04_black", "1girl, solo, straight black hair, sailor uniform, upper body portrait, front view, looking at viewer"),
    ("nat_05_red", "1girl, solo, crimson ponytail hair, sweater, upper body portrait, front view, looking at viewer"),
    ("nat_06_green", "1girl, solo, mint green hair, blazer, upper body portrait, front view, looking at viewer"),
    ("nat_07_brown", "1boy, solo, brown hair, collared shirt, upper body portrait, front view, looking at viewer"),
]

TROPE_TAGS = [
    ("native_cheerful", "cheerful bright smile, bright sparkling eyes, white eye highlights, vibrant full color, anime coloring, tv anime screenshot, simple white background, masterpiece, high score", "empty eyes, shaded face, sweatdrop, blush, angry, sad, dark, sketch, lineart, 3d"),
    ("native_empty_eyes_smile", "empty eyes, no eye highlights, dull lifeless dark eyes, yandere, gentle closed-mouth smile, vibrant full color, anime coloring, tv anime screenshot, simple white background, masterpiece, high score", "sparkling eyes, white eye highlights, crying, angry, frown, sketch, lineart, 3d"),
    ("native_shaded_face_smile", "shaded face, heavy dark shadow over eyes and forehead, vertical gloom lines, ominous gentle smile, vibrant full color, anime coloring, tv anime screenshot, simple white background, masterpiece, high score", "frown, crying, angry open mouth, sketch, lineart, 3d"),
    ("native_sweatdrop_smile", "anime sweatdrop on temple, nervous awkward smile, open bright eyes, vibrant full color, anime coloring, tv anime screenshot, simple white background, masterpiece, high score", "crying, angry, frown, shaded face, empty eyes, sketch, lineart, 3d"),
]


def generate_native_suite(animagine_dir: str, out_dir: Path) -> list[dict[str, Any]]:
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    img_dir = out_dir / "images"
    sheet_dir = out_dir / "sheets"
    img_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(animagine_dir, torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    records = []
    for cidx, (cid, cdesc) in enumerate(NATIVE_CHARS):
        panels = []
        for tidx, (tname, pos_tags, neg_tags) in enumerate(TROPE_TAGS):
            # Same base seed per character for similar identity feel, slight offset per trope
            gen = torch.Generator(device="cuda").manual_seed(7700 + cidx * 101 + tidx * 7)
            img = pipe(
                prompt=f"{cdesc}, {pos_tags}",
                negative_prompt=neg_tags,
                width=832,
                height=832,
                guidance_scale=5.5,
                num_inference_steps=26,
                generator=gen,
            ).images[0]
            p = img_dir / f"{cid}__{tname}.png"
            img.save(p)
            records.append({"char_id": cid, "variant": tname, "gt": EXPECTED_NATIVE[tname], "image_path": str(p)})
            t = img.resize((256, 256), Image.LANCZOS)
            ImageDraw.Draw(t).rectangle([0, 0, 256, 24], fill=(20, 20, 28))
            ImageDraw.Draw(t).text((6, 5), tname, fill=(255, 255, 255))
            panels.append(t)

        sheet = Image.new("RGB", (256 * 4, 256))
        for i, t in enumerate(panels):
            sheet.paste(t, (i * 256, 0))
        sheet.save(sheet_dir / f"{cid}__native_sheet.jpg", quality=92)
        print(f"Generated native 4-trope sheet for {cid}", flush=True)

    man = out_dir / "manifest_native.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return records


def _get_lm_layers(model):
    for path in [("model", "language_model", "layers"), ("language_model", "layers"), ("model", "layers")]:
        cur = model
        ok = True
        for attr in path:
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
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
def eval_native_cross_domain(
    synth_manifest: Path, native_records: list[dict[str, Any]], model_dir: str, out_dir: Path
) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    abcd_ids = [processor.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in LABELS]

    layers = _get_lm_layers(model)
    l_mid, l_late = int(len(layers) * 0.55), int(len(layers) * 0.78)
    cap: dict[int, np.ndarray] = {}

    def mk(idx: int):
        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            cap[idx] = h[0, -1, :].detach().float().cpu().numpy()
        return fn

    handles = [layers[l_mid].register_forward_hook(mk(l_mid)), layers[l_late].register_forward_hook(mk(l_late))]

    def run_set(recs: list[dict[str, Any]]):
        d_lg, a_lg, f_d, f_a = [], [], [], []
        for r in recs:
            img = Image.open(r["image_path"]).convert("RGB")
            for prompt, lg_list, f_list in [(DIRECT_PROMPT, d_lg, f_d), (AUDIT_PROMPT, a_lg, f_a)]:
                m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
                t = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                inp = processor(text=[t], images=[img], return_tensors="pt").to(model.device)
                lg_list.append(model(**inp).logits[0, -1, abcd_ids].float().cpu().numpy())
                f_list.append(np.concatenate([cap[l_mid], cap[l_late]], axis=0))
        return np.stack(d_lg), np.stack(a_lg), np.stack(f_d), np.stack(f_a)

    try:
        # Train probe ONLY on synthetic counterfactuals (10 chars x 4 seen variants)
        synth_all = [json.loads(l) for l in synth_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        synth_chars = sorted({r["char_id"] for r in synth_all})[:10]
        synth_tr = [r for r in synth_all if r["char_id"] in synth_chars and r["variant"] in EXPECTED_SYNTH]
        y_tr = np.array([0.0 if EXPECTED_SYNTH[r["variant"]] == "A" else 1.0 for r in synth_tr], dtype=np.float32)

        _, _, tr_fd, tr_fa = run_set(synth_tr)
        nat_dlg, nat_alg, nat_fd, nat_fa = run_set(native_records)
    finally:
        for h in handles:
            h.remove()

    # Fit cross-domain probes strictly on synthetic train data
    wd, bd = _fit_logreg(tr_fd, y_tr, reg=5.0)
    tau_d = _cal_thresh(_score(tr_fd, wd, bd), y_tr)
    wa, ba = _fit_logreg(tr_fa, y_tr, reg=5.0)
    tau_a = _cal_thresh(_score(tr_fa, wa, ba), y_tr)

    def bind(flags: np.ndarray) -> list[str]:
        preds = []
        for i, flg in enumerate(flags):
            if not flg:
                preds.append("A")
            else:
                p1 = np.exp(nat_dlg[i, 1:] - np.max(nat_dlg[i, 1:])); p1 /= p1.sum()
                p2 = np.exp(nat_alg[i, 1:] - np.max(nat_alg[i, 1:])); p2 /= p2.sum()
                preds.append(LABELS[1 + int(np.argmax(p1 + p2))])
        return preds

    def summarize(preds: list[str]) -> dict[str, Any]:
        by_v: dict[str, list[bool]] = {}
        by_a: dict[str, list[bool]] = {}
        for r, p in zip(native_records, preds):
            by_v.setdefault(r["variant"], []).append(p == EXPECTED_NATIVE[r["variant"]])
            by_a.setdefault(r["variant"], []).append(p == "A")
        sv = {v: round(float(np.mean(xs)), 4) for v, xs in sorted(by_v.items())}
        sa = {v: round(float(np.mean(xs)), 4) for v, xs in sorted(by_a.items())}
        conf = float(np.mean([sv["native_empty_eyes_smile"], sv["native_shaded_face_smile"], sv["native_sweatdrop_smile"]]))
        return {"macro_4": round(float(np.mean(list(sv.values()))), 4), "clean": sv["native_cheerful"], "conflict_3": round(conf, 4), "by_variant": sv, "pred_A_rate": sa}

    res = {
        "model": Path(model_dir).name,
        "direct": summarize([LABELS[int(np.argmax(lg))] for lg in nat_dlg]),
        "audit": summarize([LABELS[int(np.argmax(lg))] for lg in nat_alg]),
        "synth2native_probe_direct": summarize(bind(_score(nat_fd, wd, bd) >= tau_d)),
        "synth2native_probe_audit": summarize(bind(_score(nat_fa, wa, ba) >= tau_a)),
    }
    out_path = out_dir / f"exp10a_native_{Path(model_dir).name}.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"=== EXP 10A NATIVE END-TO-END ANIME: {res['model']} ===")
    for k in ["direct", "audit", "synth2native_probe_direct", "synth2native_probe_audit"]:
        s = res[k]
        bv = s["by_variant"]
        print(
            f"  {k:26s} macro4={s['macro_4']:.3f} clean={s['clean']:.2f} conf3={s['conflict_3']:.3f} "
            f"| empty_eye={bv['native_empty_eyes_smile']:.2f} shaded={bv['native_shaded_face_smile']:.2f} sweat={bv['native_sweatdrop_smile']:.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animagine-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/animagine-xl-4.0")
    ap.add_argument("--synth-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--eval-model", type=str, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    man = args.out_dir / "manifest_native.jsonl"
    if args.generate or not man.exists():
        native_recs = generate_native_suite(args.animagine_dir, args.out_dir)
    else:
        native_recs = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.eval_model:
        eval_native_cross_domain(args.synth_manifest, native_recs, args.eval_model, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
