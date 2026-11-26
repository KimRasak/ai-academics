#!/usr/bin/env python3
"""Aggregate all eval results into the M3/M4 master matrix.
Reads: results/census_eval_*.json (variants), results/m4e_probes_base.json (2h baseline),
       results/m4c_probes_*.json, remote m4 JSONs pulled locally.
Writes: results/MASTER_MATRIX.json + prints a markdown table."""
import json, glob, os
import numpy as np
from pathlib import Path

GT_MAP = {'angry':'angry','happy':'happy','sad':'sad','neutral':'neutral','calm':'neutral',
          'fear':'fear','disgust':'disgust','surprised':'surprised'}

def census_stats(path):
    d = json.load(open(path))
    rows = [r for r in d['rows'] if 'fer' in r]
    if not rows:
        return None
    exprs = [r['expressiveness'] for r in rows]
    m = [r for r in rows if r['gt_emotion'] in GT_MAP]
    acc = float(np.mean([r['argmax'] == GT_MAP[r['gt_emotion']] for r in m])) if m else None
    from collections import Counter
    return {'n': len(rows), 'expr': float(np.mean(exprs)), 'expr_std': float(np.std(exprs)),
            'gt_match': acc, 'preds': dict(Counter(r['argmax'] for r in rows))}

def m4_stats(path):
    if not Path(path).exists():
        return None
    rows = json.load(open(path))
    arc = [r['arcface_cos'] for r in rows if r.get('arcface_cos') is not None]
    syn = [r['sync_proxy'] for r in rows if r.get('sync_proxy') is not None]
    return {'n': len(rows), 'arc': float(np.mean(arc)) if arc else None,
            'sync': float(np.mean(syn)) if syn else None}

def main():
    matrix = {}
    # 30-audio eval set variants (local census)
    for p in sorted(glob.glob('results/census_eval_*.json')):
        tag = Path(p).stem.replace('census_eval_', '')
        matrix[tag] = census_stats(p)
    # 2h probes baseline
    b = json.load(open('results/m4e_probes_base.json'))
    exprs = [r['expr'] for r in b]
    matrix['probe2h_base'] = {'n': len(b), 'expr': float(np.mean(exprs)), 'expr_std': float(np.std(exprs))}
    # variants on 2h probes
    for p in sorted(glob.glob('results/m4*c_probes_*.json') + glob.glob('results/m4b_probes_*.json')):
        tag = Path(p).stem.replace('m4c_probes_','').replace('m4b_probes_','')
        d = json.load(open(p))
        exprs = [r['expr'] for r in d]
        matrix[f'probe2h_{tag}'] = {'n': len(d), 'expr': float(np.mean(exprs)), 'expr_std': float(np.std(exprs))}
    json.dump(matrix, open('results/MASTER_MATRIX.json','w'), indent=1)
    # markdown table
    print('| variant | set | n | expr↑ | GT-match↑ |', flush=True)
    for tag, st in matrix.items():
        if st is None: continue
        print(f"| {tag} | {'probes' if 'probe' in tag else 'eval30'} | {st.get('n')} | {st.get('expr',0):.3f} | {st.get('gt_match') if st.get('gt_match') is not None else '—'} |")
    # remote m4 (arc/sync) if pulled
    for tag in ['vA2']:
        for kind in ['eval30','probes']:
            pass

if __name__ == '__main__':
    main()
