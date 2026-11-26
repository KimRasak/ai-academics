"""Seed-pair prompt-ranking stability (within model, no cross-model score comparison).

For every pair of seeds (s,t), rank the 16 prompts by their single-seed VT mean
under s and under t, then compute Spearman rho and mean absolute rank
displacement. Quantifies how much a prompt-level ranking changes from seed
choice alone, using the same E3/E4 canonical data as the paper.
"""
import json, glob, statistics
from collections import defaultdict

def load(pattern):
    byp = defaultdict(dict)
    for p in glob.glob(pattern):
        d = json.load(open(p)); vid = d.get('video_id', '')
        if '_seed' not in vid: continue
        pid, seed = vid.rsplit('_seed', 1)
        byp[pid][int(seed)] = d['VT']
    return byp

def load_e3():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'data/scivbench'
    byp = defaultdict(dict)
    for p in (root / 'e3_new_vt/h3_base').glob('*.json'):
        if p.name == 'aggregate.json': continue
        pid, seed = p.stem.rsplit('_seed', 1)
        byp[pid][int(seed)] = json.load(open(p))['VT']
    for d in list((root / 'pilot_full').glob('*_seed0')) + list((root / 'pilot_full').glob('*_seed1')):
        v = json.load(open(root / 'vt_full/h3_base' / (d.name + '.json')))
        byp[d.name.rsplit('_seed', 1)[0]][int(d.name.rsplit('_seed', 1)[1])] = v['VT']
    return byp

def spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}
    rb = {v: i for i, v in enumerate(sorted(range(len(b)), key=lambda i: b[i]))}
    n = len(a); d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))

def analyze(byp, label):
    pids = sorted(byp)
    rhos, disp = [], []
    seeds = sorted(set.intersection(*[set(byp[p]) for p in pids]))
    for i, s in enumerate(seeds):
        for t in seeds[i + 1:]:
            vs = [byp[p][s] for p in pids]; vt = [byp[p][t] for p in pids]
            rhos.append(spearman(vs, vt))
            rs = {v: r for r, v in enumerate(sorted(range(len(vs)), key=lambda i: vs[i]))}
            rt = {v: r for r, v in enumerate(sorted(range(len(vt)), key=lambda i: vt[i]))}
            disp.append(statistics.mean(abs(rs[i] - rt[i]) for i in range(len(vs))))
    out = {'model': label, 'n_seed_pairs': len(rhos),
           'spearman_mean': statistics.mean(rhos), 'spearman_min': min(rhos),
           'mean_abs_rank_shift': statistics.mean(disp), 'max_abs_rank_shift': max(disp)}
    print(json.dumps(out))
    return out

results = [analyze(load_e3(), 'H3'), analyze(load('data/wan_cross/vt2/*.json'), 'Wan2.1-Fun')]
json.dump(results, open('data/wan_cross/seed_rank_stability.json', 'w'), indent=2)
