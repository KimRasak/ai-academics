#!/usr/bin/env python3
"""Census analysis: FER over generated videos x seeds; join with GT emotion.
Outputs: results/census_arm1.json (per-chunk-per-seed rows + summary)."""
import json, sys
from pathlib import Path
import numpy as np

AVATALK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AVATALK))
from eval_fidelity import Judges, pool  # noqa: E402

SHARED = ['angry', 'happy', 'neutral', 'sad']
MAP7TO4 = {'angry': 'angry', 'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad'}

def main(gen_dir, out_json, manifest_path=None, limit=None):
    J = Judges(str(AVATALK / 'refs/judges.json'))
    vids = sorted(gen_dir.rglob('chunk_*_seed*.mp4'))
    if limit:
        vids = vids[:limit]
    man = json.load(open(manifest_path))['segments'] if manifest_path else None
    rows = []
    for v in vids:
        chunk_id = v.stem.split('_seed')[0]
        seed = int(v.stem.split('seed')[1])
        seg_idx = int(chunk_id.split('_')[1])
        gt = man[seg_idx]['emotion'] if man and seg_idx < len(man) else None
        try:
            per_model, n_frames = J.fer_video(v)
        except Exception as e:
            rows.append({'chunk': chunk_id, 'seed': seed, 'error': str(e)[:120]})
            continue
        if n_frames == 0:
            rows.append({'chunk': chunk_id, 'seed': seed, 'error': 'zero frames'})
            continue
        pooled = []
        for name, dists in per_model.items():
            if not dists:
                continue
            labels = J.fer[name][2]
            p7 = pool(dists, labels)
            p4 = {k: 0.0 for k in SHARED}
            for l7, prob in p7.items():
                if l7 in MAP7TO4:
                    p4[MAP7TO4[l7]] += prob
            s = sum(p4.values()) or 1.0
            pooled.append({k: v / s for k, v in p4.items()})
        if not pooled:
            rows.append({'chunk': chunk_id, 'seed': seed, 'error': 'no faces'})
            continue
        p = {k: float(np.mean([e[k] for e in pooled])) for k in SHARED}
        expr = 1 - p['neutral']
        rows.append({'chunk': chunk_id, 'seed': seed, 'gt_emotion': gt,
                     'fer': p, 'expressiveness': expr, 'argmax': max(p, key=p.get)})
        print(f"{chunk_id} seed{seed}: expr={expr:.3f} argmax={max(p, key=p.get)} gt={gt}")
    ok = [r for r in rows if 'fer' in r]
    summary = {'n': len(ok),
               'mean_expr': float(np.mean([r['expressiveness'] for r in ok])) if ok else None,
               'per_seed': {}, 'expr_by_gt': {}}
    for s in sorted({r['seed'] for r in ok}):
        rs = [r for r in ok if r['seed'] == s]
        summary['per_seed'][s] = float(np.mean([r['expressiveness'] for r in rs]))
    for gt in sorted({r['gt_emotion'] for r in ok if r['gt_emotion']}):
        rs = [r for r in ok if r['gt_emotion'] == gt]
        summary['expr_by_gt'][gt] = float(np.mean([r['expressiveness'] for r in rs]))
    exprs = [r['expressiveness'] for r in ok]
    summary['expr_std_across_seeds'] = float(np.std(exprs)) if ok else None
    # SER side on the same chunks (audio): 4-class
    import soundfile as sf
    ser_rows = []
    for r in ok[:0] if False else ok:
        pass
    json.dump({'rows': rows, 'summary': summary}, open(out_json, 'w'), indent=1)
    print('WROTE', out_json, '| n =', len(ok))
    return summary

if __name__ == '__main__':
    gen_dir = Path(sys.argv[1]); out_json = sys.argv[2]
    manifest = sys.argv[3] if len(sys.argv) > 3 else None
    main(gen_dir, out_json, manifest)
