"""SER(audio) <-> FER(face) emotion-faithfulness pipeline (Avatars That Keep Feeling).

Judges (local snapshots, see refs/judges.json):
- SER: speechbrain/emotion-recognition-wav2vec2-IEMOCAP  -> 4 classes [ang, hap, neu, sad]
- FER: trpakov/vit-face-expression (+ dima806 fallback)  -> 7 classes

Metrics per session (audio split into consecutive chunks; per chunk we have
driving audio + generated portrait video):
- ser_dist[c]  : audio-side emotion distribution (4-class)
- fer_dist[c]  : face-side pooled frame-level distribution (7-class -> mapped)
- agree[c]     : agreement between mapped ser_dist and fer_dist
                  (Jensen-Shannon similarity + argmax match + L1 intensity gap)
- expressiveness[c]: 1 - P(neutral) on face side; plus dynamics = std of
                  per-frame argmax index and mean max-prob
- fidelity_curve: agree / expressiveness as functions of chunk position t
Seed-resolved: run per seed; between-seed variance of curve stats.

Usage:
  python3 eval_fidelity.py --session_dir <dir with chunk_XXX/{audio.wav,video.mp4}> \
      --out results/session.json
"""
import argparse, json, os
import numpy as np
import torch
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification, AutoProcessor, AutoModelForAudioClassification
import librosa
import cv2

SER_LABELS = ["ang", "hap", "neu", "sad"]  # speechbrain IEMOCAP order
FER_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # trpakov order (verify at runtime)
MAP = {"ang": "angry", "hap": "happy", "neu": "neutral", "sad": "sad"}  # shared 4-class space
SHARED = ["angry", "happy", "neutral", "sad"]

_device = "cuda" if torch.cuda.is_available() else "cpu"

class Judges:
    def __init__(self, judges_json):
        cfg = json.load(open(judges_json))
        self.fer = {}; self.ser = None
        for c in [x for x in cfg if x["tag"] == "fer"]:
            name = Path(c["path"]).name
            try:
                fe = AutoImageProcessor.from_pretrained(c["path"])
            except Exception:
                fe = AutoFeatureExtractor.from_pretrained(c["path"])
            fm = AutoModelForImageClassification.from_pretrained(c["path"]).to(_device).eval()
            labels = fm.config.id2label
            # normalize label strings
            labels = {k: v.lower() for k, v in labels.items()}
            if isinstance(labels, dict):
                labels = [labels[k] for k in sorted(labels.keys())]
            self.fer[name] = (fe, fm, list(labels))
        ser_cfg = [x for x in cfg if x["tag"] == "ser"][0]
        try:
            self.ser_proc = AutoFeatureExtractor.from_pretrained(ser_cfg["path"])
        except Exception:
            self.ser_proc = AutoProcessor.from_pretrained(ser_cfg["path"])
        self.ser_model = AutoModelForAudioClassification.from_pretrained(ser_cfg["path"]).to(_device).eval()
        self.ser_labels = [l.lower() for l in self.ser_model.config.id2label.values()]
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    @torch.no_grad()
    def ser_chunk(self, wav_path, sr=16000):
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
        if len(wav) < sr:  # pad to >=1s
            wav = np.pad(wav, (0, sr - len(wav) + 1))
        inputs = self.ser_proc(wav, sampling_rate=sr, return_tensors="pt").to(_device)
        logits = self.ser_model(**inputs).logits[0]
        p = torch.softmax(logits, -1).cpu().numpy()
        return {self.ser_labels[i]: float(p[i]) for i in range(len(p))}

    def _face_crops(self, frame_bgr, max_faces=1):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(48, 48))
        if len(faces) == 0:  # portrait videos: fallback center crop
            h, w = frame_bgr.shape[:2]
            s = min(h, w)
            return [frame_bgr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]]
        faces = sorted(faces, key=lambda f: -f[2] * f[3])[:max_faces]
        return [frame_bgr[y:y + h2, x:x + w2] for (x, y, w2, h2) in faces]

    @torch.no_grad()
    def fer_video(self, vid_path, max_frames=32):
        cap = cv2.VideoCapture(str(vid_path))
        per_model = {k: [] for k in self.fer}
        n = 0
        idx = 0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        step = max(1, int(total // max_frames)) if total else 1
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % step == 0 and n < max_frames:
                for crop in self._face_crops(frame):
                    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
                    for name, (fe, fm, labels) in self.fer.items():
                        inputs = fe(images=rgb, return_tensors="pt").to(_device)
                        p = torch.softmax(fm(**inputs).logits[0], -1).cpu().numpy()
                        per_model[name].append({labels[i]: float(p[i]) for i in range(len(p))})
                n += 1
            idx += 1
        cap.release()
        return per_model, n

def pool(dists, labelset):
    """average list of per-frame label->prob dicts into pooled distribution"""
    if not dists: return {l: 0.0 for l in labelset}
    out = {l: 0.0 for l in labelset}
    for d in dists:
        for l in labelset: out[l] += d.get(l, 0.0)
    s = sum(out.values()) or 1.0
    return {l: out[l] / s for l in labelset}

def js_similarity(p, q, eps=1e-8):
    ks = sorted(set(p) | set(q))
    pv = np.array([p.get(k, eps) for k in ks]); qv = np.array([q.get(k, eps) for k in ks])
    pv = pv / pv.sum(); qv = qv / qv.sum()
    m = (pv + qv) / 2
    kl = lambda a, b: np.sum(a * np.log((a + eps) / (b + eps)))
    js = 0.5 * kl(pv, m) + 0.5 * kl(qv, m)
    return float(np.exp(-js))  # 1 = identical

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_dir", required=True)
    ap.add_argument("--judges", default=str(Path(__file__).parent / "refs/judges.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed_tag", default="seed0")
    ap.add_argument("--audio_only", action="store_true", help="SER probe when videos not generated yet")
    args = ap.parse_args()

    J = Judges(args.judges)
    chunks = sorted(Path(args.session_dir).glob("chunk_*"))
    assert chunks, f"no chunk_* in {args.session_dir}"
    rows = []
    for c in chunks:
        audio = c / "audio.wav"; video = c / "video.mp4"
        if not audio.exists() or (not video.exists() and not args.audio_only):
            rows.append({"chunk": c.name, "missing": True}); continue
        ser = J.ser_chunk(audio)                      # 4-class IEMOCAP space
        ser_shared = {MAP[l]: ser.get(l, 0.0) for l in SER_LABELS}   # -> shared names
        if args.audio_only:
            expressiveness = 1.0 - ser_shared["neutral"]
            rows.append({"chunk": c.name, "ser": ser, "audio_expr": expressiveness,
                         "gt_emotion": None}); continue
        per_model, n_frames = J.fer_video(video)
        # pool each FER model then average models (judge ensemble)
        pooled = [pool(d, FER_LABELS) for d in per_model.values() if d]
        fer_full = {l: float(np.mean([p.get(l, 0.0) for p in pooled])) for l in FER_LABELS}
        fer_shared = {l: fer_full.get(l, 0.0) for l in SHARED}
        agree_js = js_similarity(ser_shared, fer_shared)
        argmax_match = float(max(ser_shared, key=ser_shared.get) == max(fer_shared, key=fer_shared.get))
        expressiveness = 1.0 - fer_shared["neutral"]
        # dynamics: per-frame argmax variability (first FER model)
        first = list(per_model.values())[0] if per_model else []
        if len(first) > 1:
            argmaxes = [max(d, key=d.get) for d in first]
            changes = sum(1 for a, b in zip(argmaxes, argmaxes[1:]) if a != b)
            dyn = changes / (len(argmaxes) - 1)  # label-change rate per frame
        else:
            dyn = 0.0
        rows.append({"chunk": c.name, "ser": ser, "fer": fer_full,
                     "agree_js": agree_js, "argmax_match": argmax_match,
                     "expressiveness": expressiveness, "dynamics": dyn,
                     "n_frames": n_frames})
        print(f"{c.name}: agree={agree_js:.3f} match={argmax_match} expr={expressiveness:.3f} dyn={dyn:.3f}")
    ok = [r for r in rows if not r.get("missing")]
    def _mean(rs, key):
        vals = [r[key] for r in rs if key in r]
        return float(np.mean(vals)) if vals else None
    summary = {
        "seed_tag": args.seed_tag,
        "n_chunks": len(rows), "n_ok": len(ok),
        "mean_agree": _mean(ok, "agree_js"),
        "mean_expr": _mean(ok, "expressiveness"),
        "mean_dyn": _mean(ok, "dynamics"),
        "curve": [{"t": i, "agree": r.get("agree_js"), "expr": r.get("expressiveness", r.get("audio_expr")),
                   "dyn": r.get("dynamics"), "match": r.get("argmax_match")}
                  for i, r in enumerate(ok)],
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    print("WROTE", args.out)

if __name__ == "__main__":
    main()
