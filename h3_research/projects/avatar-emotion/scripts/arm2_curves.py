#!/usr/bin/env python3
"""Arm 2: per-frame FER over long session videos -> F(t) drift curves."""
import json, sys
from pathlib import Path
import numpy as np

AVATALK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AVATALK))
from eval_fidelity import Judges, pool  # noqa: E402

SHARED = ['angry', 'happy', 'neutral', 'sad']
MAP7TO4 = {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'}

def curve_for_video(J, v, max_frames=200):
    per_model, n = J.fer_video(v, max_frames=max_frames)
    if n == 0:
        return None, None
    curves = []
    for name, dists in per_model.items():
        labels = J.fer[name][2]
        p7 = pool(dists, labels)
        p4 = {k: 0.0 for k in SHARED}
        for l7, prob in p7.items():
            if l7 in MAP7TO4:
                p4[MAP7TO4[l7]] += prob
        s = sum(p4.values()) or 1.0
        curves.append({k: vv / s for k, vv in p4.items()})
    p = {k: float(np.mean([c[k] for c in curves])) for k in SHARED}
    return p, n

def per_window_curve(J, v, window_frames=96, max_frames=200):
    """Compute FER in sliding windows along the video -> expr(t)."""
    import cv2, torch
    cap = cv2.VideoCapture(str(v))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    step = max(1, total // max_frames) if total else 1
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0 and len(frames) < max_frames:
            crops = J._face_crops(frame)
            if crops:
                rgb = cv2.cvtColor(cv2.resize(crops[0], (224, 224)), cv2.COLOR_BGR2RGB)
                frames.append((idx, rgb))
        idx += 1
    cap.release()
    if not frames:
        return []
    # per-window pooled FER
    out = []
    W = max(1, window_frames // step)
    for ws in range(0, len(frames), W):
        chunk = frames[ws:ws + W]
        ens = []
        for name, (fe, fm, labels) in J.fer.items():
            dists = []
            for _, rgb in chunk:
                inputs = fe(images=rgb, return_tensors="pt").to(next(fm.parameters()).device)
                with torch.no_grad():
                    p = torch.softmax(fm(**inputs).logits[0], -1).cpu().numpy()
                dists.append({labels[i]: float(p[i]) for i in range(len(p))})
            p7 = pool(dists, labels)
            p4 = {k: 0.0 for k in SHARED}
            for l7, prob in p7.items():
                if l7 in MAP7TO4:
                    p4[MAP7TO4[l7]] += prob
            s = sum(p4.values()) or 1.0
            ens.append({k: v / s for k, v in p4.items()})
        p = {k: float(np.mean([e[k] for e in ens])) for k in SHARED}
        t_sec = chunk[0][0] / fps
        out.append({'t': round(t_sec, 1), 'expr': 1 - p['neutral'],
                    'argmax': max(p, key=p.get)})
    return out

def main(gen_dir, out_json):
    J = Judges(str(AVATALK / 'refs/judges.json'))
    vids = sorted(gen_dir.glob('session40_seed*.mp4'))
    res = {}
    for v in vids:
        seed = int(v.stem.split('seed')[1])
        wc = per_window_curve(J, v)
        res[seed] = wc
        print(f"seed{seed}: {len(wc)} windows | expr head/mid/tail:",
              [round(w['expr'], 3) for w in wc[:3]] if wc else '-',
              [round(w['expr'], 3) for w in wc[len(wc)//2:len(wc)//2+3]] if len(wc) > 4 else '-',
              [round(w['expr'], 3) for w in wc[-3:]] if wc else '-')
    json.dump(res, open(out_json, 'w'), indent=1)
    print('WROTE', out_json)

if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2])
