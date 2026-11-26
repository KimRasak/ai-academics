#!/usr/bin/env python3
"""RAFT reward scorer (remote): score generated videos for reward-weighted selection.
reward = 0.5*agree + 0.5*gt_match, gated by arcface>=0.6; audio GT from CREMA-D filename / RAVDESS manifest.
Usage: python raft_score.py --dir /ssd2/jinzili/avdata/gen/raft_K --ref a.png --out rewards.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch, cv2

JUDGES = {
    'fer1': '/ssd2/jinzili/av_weights/judges/vit-face-expression',
    'fer2': '/ssd2/jinzili/av_weights/judges/facial_emotions_image_detection',
    'ser': '/ssd2/jinzili/av_weights/judges/wav2vec2-base-superb-er',
}
SHARED = ['angry', 'happy', 'neutral', 'sad']
MAP4 = {'angry':'angry','hap':'happy','ang':'angry','sad':'sad','neu':'neutral',
        'happy':'happy','sad':'sad','neutral':'neutral','calm':'neutral'}
GT_CODE = {'ANG':'angry','HAP':'happy','SAD':'sad','FEA':'fear','DIS':'disgust','NEU':'neutral'}

def load_models(device):
    from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor, AutoModelForAudioClassification
    fer = {}
    for tag, path in JUDGES.items():
        if tag == 'ser': continue
        try:
            fe = AutoImageProcessor.from_pretrained(path)
        except Exception:
            fe = AutoFeatureExtractor.from_pretrained(path)
        fm = AutoModelForImageClassification.from_pretrained(path).to(device).eval()
        id2l = fm.config.id2label
        if isinstance(id2l, dict): id2l = [id2l[k] for k in sorted(id2l)]
        fer[tag] = (fe, fm, list(id2l))
    se = AutoFeatureExtractor.from_pretrained(JUDGES['ser'])
    sm = AutoModelForAudioClassification.from_pretrained(JUDGES['ser']).to(device).eval()
    sid2l = sm.config.id2label
    if isinstance(sid2l, dict): sid2l = [sid2l[k] for k in sorted(sid2l)]
    return fer, (se, sm, sid2l)

def pool(dists, labels):
    out = {l: 0.0 for l in labels}
    for d in dists:
        for l in labels: out[l] += d.get(l, 0.0)
    s = sum(out.values()) or 1.0
    return {l: v/s for l, v in out.items()}

def fer_video(fer, path, max_frames=16, device='cuda'):
    cap = cv2.VideoCapture(str(path))
    per = {k: [] for k in fer}
    n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if n % 2 == 0:
            h, w = f.shape[:2]
            crop = f[int(0.12*h):int(0.88*h), int(0.22*w):int(0.78*w)]
            rgb = cv2.cvtColor(cv2.resize(crop,(224,224)), cv2.COLOR_BGR2RGB)
            for tag,(fe,fm,labels) in fer.items():
                inp = fe(images=rgb, return_tensors='pt').to(device)
                with torch.no_grad():
                    p = torch.softmax(fm(**inp).logits[0],-1).cpu().numpy()
                per[tag].append({labels[i]: float(p[i]) for i in range(len(p))})
        n += 1
        if n > 400: break
    cap.release()
    pooled = [pool(d, fer[t][2]) for t, d in per.items() if d]
    if not pooled: return None
    p4 = {k: float(np.mean([p.get(k,0) for p in pooled])) for k in SHARED}
    s = sum(p4.values()) or 1.0
    return {k: v/s for k,v in p4.items()}

def ser_audio(se, sm, sid2l, path, device='cuda'):
    import soundfile as sf
    import librosa
    d, sr = sf.read(str(path))
    if d.ndim > 1: d = d.mean(1)
    if sr != 16000:
        d = librosa.resample(np.asarray(d, dtype=np.float32), orig_sr=sr, target_sr=16000)
    inp = se(d, sampling_rate=16000, return_tensors='pt').to(device)
    with torch.no_grad():
        p = torch.softmax(sm(**inp).logits[0],-1).cpu().numpy()
    return {sid2l[i]: float(p[i]) for i in range(len(p))}

def js_sim(a, b):
    import scipy.spatial.distance as ssd
    v1 = np.array([a.get(k,0) for k in SHARED]); v2 = np.array([b.get(k,0) for k in SHARED])
    v1 = v1/v1.sum(); v2 = v2/v2.sum()
    return float(np.exp(-ssd.jensenshannon(v1, v2)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--ref', default='/ssd2/jinzili/echomimic/assets/test_imgs/a.png')
    ap.add_argument('--out', required=True)
    ap.add_argument('--audio_root', default='/ssd2/jinzili/avdata')
    args = ap.parse_args()
    device = 'cuda'
    fer, (se, sm, sid2l) = load_models(device)
    from facenet_pytorch import InceptionResnetV1
    arc = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    ref_rgb = cv2.cvtColor(cv2.resize(cv2.imread(args.ref),(160,160)), cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        e_ref = arc(torch.from_numpy(ref_rgb).permute(2,0,1).float().unsqueeze(0).cuda()/127.5-1)

    rows = []
    for v in sorted(Path(args.dir).glob('*.mp4')):
        stem = v.name.replace('_seed420.mp4','')
        # locate audio + GT
        if stem.startswith('chunk_'):
            wav = Path(args.audio_root)/stem/'audio.wav'
            man = json.load(open('/ssd2/jinzili/avdata/session_manifest.json'))['segments'] if Path('/ssd2/jinzili/avdata/session_manifest.json').exists() else None
            gt = None
            if man:
                idx = int(stem.split('_')[1]); gt = MAP4.get(man[idx]['emotion'])
        else:
            wav = Path(args.audio_root)/'cremad/AudioWAV'/f'{stem}.wav'
            gt = GT_CODE.get(stem.split('_')[2])
        try:
            p4 = fer_video(fer, v)
            assert p4 is not None
            ser = ser_audio(se, sm, sid2l, wav) if wav.exists() else None
            agree = js_sim(p4, ser) if ser else None
            gt_match = max(p4, key=p4.get) == gt if gt else None
            # arcface vs ref (3 frames)
            cap = cv2.VideoCapture(str(v)); frames=[]
            i=0
            while len(frames)<3:
                ok,f = cap.read()
                if not ok: break
                if i%40==0: frames.append(f)
                i+=1
            cap.release()
            arcs=[]
            for f in frames:
                rgb = cv2.cvtColor(cv2.resize(f,(160,160)), cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    e = arc(torch.from_numpy(rgb).permute(2,0,1).float().unsqueeze(0).cuda()/127.5-1)
                arcs.append(float(torch.cosine_similarity(e_ref,e).item()))
            arc_c = float(np.mean(arcs))
            expr = 1 - p4['neutral']
            parts = [w for w in [agree, (1.0 if gt_match else 0.0) if gt_match is not None else None] if w is not None]
            base = float(np.mean(parts)) if parts else expr
            reward = base * (1.0 if arc_c >= 0.6 else 0.0)
            rows.append({'video': v.name, 'expr': expr, 'agree': agree, 'gt_match': gt_match,
                         'arc': arc_c, 'reward': reward})
            print(rows[-1], flush=True)
        except Exception as e:
            rows.append({'video': v.name, 'error': str(e)[:100]})
            print('ERR', v.name, str(e)[:80], flush=True)
    json.dump(rows, open(args.out,'w'), indent=1)
    print('WROTE', args.out)

if __name__ == '__main__':
    main()
