"""Generate E3/E4 paper figures. No training; pure plotting of measured results."""
import json, glob, statistics, itertools
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

def load(pattern):
    byp = defaultdict(lambda: {'VT': [], 'DD': []})
    for p in glob.glob(pattern):
        d = json.load(open(p)); vid = d.get('video_id', '')
        if '_seed' not in vid: continue
        byp[vid.rsplit('_seed', 1)[0]]['VT'].append(d['VT'])
        byp[vid.rsplit('_seed', 1)[0]]['DD'].append(d['dynamic_degree'])
    return {k: v for k, v in byp.items() if len(v['VT']) >= 2}

def load_e3():
    """Canonical E3 12-seed join: seeds 0..5 (e3_new_vt + pilot_full->vt_full) + seeds 6..11 (e3_s6_11_vt)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'data/scivbench'
    byp = defaultdict(lambda: {'VT': [], 'DD': []})
    for p in (root / 'e3_new_vt/h3_base').glob('*.json'):
        if p.name == 'aggregate.json': continue
        d = json.load(open(p)); ident, seed = p.stem.rsplit('_seed', 1)
        byp[ident]['VT'].append(d['VT']); byp[ident]['DD'].append(d['dynamic_degree'])
    for d in list((root / 'pilot_full').glob('*_seed0')) + list((root / 'pilot_full').glob('*_seed1')):
        v = json.load(open(root / 'vt_full/h3_base' / (d.name + '.json')))
        ident = d.name.rsplit('_seed', 1)[0]
        byp[ident]['VT'].append(v['VT']); byp[ident]['DD'].append(v['dynamic_degree'])
    for p in (root / 'e3_s6_11_vt/h3_base').glob('*.json'):
        if p.name == 'aggregate.json': continue
        d = json.load(open(p)); ident, seed = p.stem.rsplit('_seed', 1)
        byp[ident]['VT'].append(d['VT']); byp[ident]['DD'].append(d['dynamic_degree'])
    return {k: v for k, v in byp.items() if len(v['VT']) >= 2}

h3 = load_e3()
wan = load('data/wan_cross/vt2/*.json')
ltx_raw = {}
for p in glob.glob('data/ltx_cross/vt/*.json'):
    d = json.load(open(p)); vid = d.get('video_id', '')
    if '_seed' not in vid: continue
    pid = vid.rsplit('_seed', 1)[0]
    ltx_raw.setdefault(pid, {'VT': [], 'DD': []})
    ltx_raw[pid]['VT'].append(d['VT']); ltx_raw[pid]['DD'].append(d['dynamic_degree'])
ltx = {k: v for k, v in ltx_raw.items() if len(v['VT']) == 12}
ltx_flip = sum(sum(a != b for a, b in itertools.combinations(v['DD'], 2)) for v in ltx.values()) / 1056
ltx_sds = [statistics.pstdev(v['VT']) for v in ltx.values()]
# merge Wan seeds 6..11 into the same prompt groups
for p in glob.glob('data/wan_cross/vt_s6_11/*.json'):
    d = json.load(open(p)); vid = d.get('video_id', '')
    if '_seed' not in vid: continue
    pid = vid.rsplit('_seed', 1)[0]
    wan.setdefault(pid, {'VT': [], 'DD': []})
    wan[pid]['VT'].append(d['VT']); wan[pid]['DD'].append(d['dynamic_degree'])

def stats(byp):
    vt_sds = [statistics.pstdev(v['VT']) for v in byp.values()]
    dd_means = [statistics.mean(v['DD']) for v in byp.values()]
    flips = 0
    for v in byp.values():
        ones = sum(1 for d in v['DD'] if d == 1)
        flips += ones * (len(v['DD']) - ones)  # all pairs 0<->1
    n_pairs = sum(len(v['DD']) * (len(v['DD']) - 1) // 2 for v in byp.values())
    return vt_sds, flips / n_pairs

h3_sds, h3_flip = stats(h3)
wan_sds, wan_flip = stats(wan)

# Fig 1: within-prompt VT SD vs released adjacent-model gap
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
for axi, sds, name in [(ax[0], h3_sds, 'MiniMax-H3 (E3)'), (ax[1], wan_sds, 'Wan2.1-Fun 1.3B (E4)')]:
    axi.scatter(range(1, len(sds) + 1), sorted(sds), s=22, color='#2b6cb0')
    axi.axhline(0.0066, color='#c53030', ls='--', lw=1.2,
                label='median released adjacent-model gap (0.0066)')
    axi.set_title(f'{name}\nmean SD={statistics.mean(sds):.4f}')
    axi.set_xlabel('prompt (sorted by SD)')
    axi.grid(alpha=0.3)
ax[0].set_ylabel('within-prompt VT SD (12 seeds)')
ax[0].legend(loc='upper left', fontsize=8)
fig.tight_layout(); fig.savefig('paper/assets/fig1_within_prompt_sd.png'); fig.savefig('paper/assets/fig1_within_prompt_sd.pdf'); plt.close(fig)

# Fig 2: DD flip rate + per-prompt DD dots
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
ax[0].bar(['MiniMax-H3', 'Wan2.1-Fun', 'LTX-Video-2B'],
          [h3_flip * 100, wan_flip * 100, ltx_flip * 100],
          color=['#2b6cb0', '#dd6b20', '#38a169'], width=0.55)
ax[0].set_ylabel('DD pairwise flip rate (%)')
ax[0].set_title(f'H3 {h3_flip*100:.2f}%  Wan {wan_flip*100:.2f}%  LTX {ltx_flip*100:.2f}%')
ax[0].grid(alpha=0.3, axis='y')
for axi, byp, name in [(ax[1], h3, 'H3'), (ax[1], wan, 'Wan')]:
    pass
for i, (byp, name, col) in enumerate([(h3, 'MiniMax-H3', '#2b6cb0'), (wan, 'Wan2.1-Fun', '#dd6b20'), (ltx, 'LTX-Video-2B', '#38a169')]):
    xs, ys = [], []
    for j, v in enumerate(sorted(byp.values(), key=lambda v: statistics.mean(v['DD']))):
        xs.extend([i + (j - 8) * 0.028] * len(v['DD'])); ys.extend(v['DD'])
    ax[1].scatter(xs, ys, s=14, alpha=0.75, color=col, label=name)
ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(['DD=0\n(static)', 'DD=1\n(dynamic)'])
ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(['MiniMax-H3', 'Wan2.1-Fun', 'LTX-Video-2B'])
ax[1].set_title('per-prompt DD across 12 seeds')
ax[1].grid(alpha=0.3, axis='y')
fig.tight_layout(); fig.savefig('paper/assets/fig2_dd_flip.png'); fig.savefig('paper/assets/fig2_dd_flip.pdf'); plt.close(fig)
print('FIG_OK', f'{h3_flip:.4f}', f'{wan_flip:.4f}', f'{ltx_flip:.4f}')
