#!/usr/bin/env python3
"""M4: 2h simulated livestream — audio-level SER drift/collapse analysis.
Builds the canonical session_v2 sequence (1440 RAVDESS wavs once + 360 repeats, seed 7),
then SER per 4s window across the full 2h. Writes plan + curves."""
import json, glob, random
import numpy as np
import soundfile as sf

WAV_DIR = 'data/ravdess_wav'
OUT_PLAN = 'results/session_v2_plan.json'
OUT_CURVE = 'results/session_v2_ser_curves.json'

def build_plan():
    wavs = sorted(glob.glob(f'{WAV_DIR}/*.wav'))
    assert len(wavs) >= 1440, len(wavs)
    random.seed(7)
    seq = wavs[:] + random.sample(wavs, 360)   # 1440 + 360 = 1800 segs
    random.shuffle(seq)
    # emotion label from RAVDESS filename: 03-01-EMO-XX-...
    plan = [{'wav': w, 'emotion': w.split('/')[-1].split('_')[2]} for w in seq]
    json.dump(plan, open(OUT_PLAN, 'w'))
    print('plan:', len(plan), 'segments')
    return plan

def main():
    import sys
    sys.path.insert(0, '.')
    from eval_fidelity import Judges
    J = Judges('refs/judges.json')
    plan = build_plan()
    win_s, sr = 4.0, None
    curve = []
    for i, seg in enumerate(plan):
        try:
            d, sr = sf.read(seg['wav'])
            if d.ndim > 1: d = d.mean(1)
            need = int(win_s * sr)
            if len(d) > need: d = d[:need]
            probs = J.ser_chunk_array(d, sr) if hasattr(J, 'ser_chunk_array') else None
            if probs is None:
                import tempfile, soundfile as sf2
                t = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf2.write(t.name, d, sr)
                probs = J.ser_chunk(t.name)
            p = {k: float(v) for k, v in probs.items()}
            expr = 1 - p.get('neu', 0.0)
            curve.append({'t': round(i * 4.0, 1), 'file': seg['wav'].split('/')[-1],
                          'gt': seg['emotion'], 'probs': p, 'expr': expr})
        except Exception as e:
            curve.append({'t': round(i * 4.0, 1), 'error': str(e)[:80]})
        if i % 200 == 0:
            print(i, flush=True)
    # summary: drift over thirds + per-GT
    valid = [c for c in curve if 'expr' in c]
    thirds = [np.mean([c['expr'] for c in valid if c['t'] < 2400]),
              np.mean([c['expr'] for c in valid if 2400 <= c['t'] < 4800]),
              np.mean([c['expr'] for c in valid if c['t'] >= 4800])]
    from collections import defaultdict
    bygt = defaultdict(list)
    for c in valid: bygt[c['gt']].append(c['expr'])
    summary = {'n_windows': len(valid), 'duration_s': len(valid) * 4.0,
               'expr_thirds': [round(float(x), 4) for x in thirds],
               'expr_by_gt': {k: round(float(np.mean(v)), 3) for k, v in sorted(bygt.items())}}
    json.dump({'summary': summary, 'curve': curve}, open(OUT_CURVE, 'w'))
    print(json.dumps(summary, indent=1))

if __name__ == '__main__':
    main()
