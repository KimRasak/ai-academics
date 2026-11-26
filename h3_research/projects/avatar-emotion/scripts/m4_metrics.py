#!/usr/bin/env python3
"""M4 metrics (remote): per video -> ArcFace identity cosine + mouth-sync proxy.
Usage: python m4_metrics.py --ref a.png --dir /ssd2/jinzili/avdata/gen/eval_vA_step8000 --out m4_vA.json"""
import argparse, json, subprocess, tempfile
from pathlib import Path
import numpy as np
import cv2, torch

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default='/ssd2/jinzili/echomimic/assets/test_imgs/a.png')
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_frames', type=int, default=12)
    ap.add_argument('--audio_map', default=None, help='json {video_name: wav_path}')
    return ap.parse_args()

def load_identity_model(device):
    from facenet_pytorch import InceptionResnetV1
    m = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return m

def face_box(img, detector):
    boxes, probs = detector.detect(img, landmarks=False)
    if boxes is None or len(boxes) == 0 or probs[0] is None or probs[0] < 0.6:
        h, w = img.shape[:2]
        return int(0.22*w), int(0.12*h), int(0.78*w), int(0.88*h)
    x0, y0, x1, y1 = boxes[0]
    return int(max(0,x0)), int(max(0,y0)), int(min(img.shape[1],x1)), int(min(img.shape[0],y1))

def identity_score(net, mtcnn, ref_bgr, frames):
    ref_rgb = cv2.cvtColor(cv2.resize(ref_bgr, (160,160)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(ref_rgb).permute(2,0,1).float().unsqueeze(0).cuda()/127.5-1
    with torch.no_grad():
        e_ref = net(t)
    scores = []
    for f in frames:
        rgb = cv2.cvtColor(cv2.resize(f,(160,160)), cv2.COLOR_BGR2RGB)
        tt = torch.from_numpy(rgb).permute(2,0,1).float().unsqueeze(0).cuda()/127.5-1
        with torch.no_grad():
            e = net(tt)
        scores.append(float(torch.cosine_similarity(e_ref, e).item()))
    return float(np.mean(scores))

def sync_proxy(video_path, detector, audio_wav=None):
    """corr(mouth-region temporal diff energy, audio RMS envelope)"""
    video_path = Path(video_path)
    audio_src = audio_wav
    if audio_src is None:
        wa = video_path.with_name(video_path.stem + '_withaudio.mp4')
        if wa.exists():
            audio_src = str(wa)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    prev = None
    mouth_diff, mouth_ok = [], []
    idx = 0
    bx0=by0=bx1=by1=None
    while True:
        ok, f = cap.read()
        if not ok: break
        if idx % 3 == 0:
            if bx0 is None:
                bx0,by0,bx1,by1 = face_box(f, detector)
                mw, mh = bx1-bx0, by1-by0
                mx0,my0,mx1,my1 = int(bx0+0.25*mw), int(by0+0.55*mh), int(bx1-0.25*mw), int(by0+0.85*mh)
            g = cv2.cvtColor(f[my0:my1, mx0:mx1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None and g.shape == prev.shape:
                mouth_diff.append(float(np.abs(g-prev).mean()))
            prev = g
        idx += 1
    cap.release()
    if len(mouth_diff) < 8:
        return None
    # audio RMS envelope at same fps/3 rate
    import soundfile as sf
    if audio_src is None:
        return None
    if audio_src.endswith('.wav'):
        audio, sr = sf.read(audio_src)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(audio_src),'-ac','1','-ar','16000',tmp.name], check=True)
        audio, sr = sf.read(tmp.name)
    hop = int(sr / (fps/3))
    env = []
    for i in range(0, len(audio)-hop, hop):
        seg = audio[i:i+hop]
        env.append(float(np.sqrt(np.mean(seg**2))+1e-9))
    n = min(len(env), len(mouth_diff))
    if n < 8: return None
    a = np.array(env[:n]); b = np.array(mouth_diff[:n])
    if a.std() < 1e-6 or b.std() < 1e-6: return 0.0
    return float(np.corrcoef(a, b)[0,1])

def main():
    args = parse()
    device = 'cuda'
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    net = load_identity_model(device)
    ref_bgr = cv2.imread(args.ref)
    amap = json.load(open(args.audio_map)) if args.audio_map else None
    vids = sorted(Path(args.dir).glob('*.mp4'))
    rows = []
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total//args.max_frames)
        frames, i = [], 0
        while len(frames) < args.max_frames:
            ok, f = cap.read()
            if not ok: break
            if i % step == 0: frames.append(f)
            i += 1
        cap.release()
        row = {'video': v.name}
        try:
            row['arcface_cos'] = identity_score(net, mtcnn, ref_bgr, frames)
        except Exception as e:
            row['arcface_cos'] = None
        try:
            wav = amap.get(v.name) if amap else None
            row['sync_proxy'] = sync_proxy(v, mtcnn, wav)
        except Exception as e:
            row['sync_proxy'] = None
        rows.append(row)
        print(row, flush=True)
    json.dump(rows, open(args.out,'w'), indent=1)
    print('WROTE', args.out)

if __name__ == '__main__':
    main()
