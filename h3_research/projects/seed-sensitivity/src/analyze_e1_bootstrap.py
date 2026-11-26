"""E1 strengthening: bootstrap CI for pooled Spearman + human-join figure.

Rebuilds the E1 join offline from cached HF files (human_scores.jsonl,
vt_scores.jsonl), same pairing as e1_human_join.py. Produces:
  - video-level bootstrap 95% CI for Spearman(VT, expert LPF) and (DD, LPF)
  - paper/assets/fig3_human_join.png (scatter + per-metric rho bars)
Descriptive only; no cluster inference claimed.
"""
import json, random, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

root = 'data/scivbench'
human = [json.loads(x) for x in open(root + '/human_scores.jsonl')]
vtrows = [json.loads(x) for x in open(root + '/vt_scores.jsonl')]
hm = {(x['model'], x['video_id']): x for x in human}
vm = {(x['model'], x['video_id']): x for x in vtrows}
rows = []
for k, h in hm.items():
    v = vm.get(k)
    if v and v.get('VT') is not None and h.get('lpf') is not None:
        rows.append((v['VT'], v.get('dynamic_degree'), h['lpf']))
assert rows, 'no joined rows'

vt = [r[0] for r in rows]
dd = [r[1] for r in rows]
lpf = [r[2] for r in rows]

def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r

def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float('nan')

rho_vt = spearman(vt, lpf)
rho_dd = spearman(dd, lpf)

rng = random.Random(0)
B = 2000
def boot(xs):
    n = len(xs); out = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(spearman([xs[i] for i in idx], [lpf[i] for i in idx]))
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]

ci_vt = boot(vt); ci_dd = boot(dd)
print(f'n={len(rows)} rho(VT,LPF)={rho_vt:.4f} CI[{ci_vt[0]:.4f},{ci_vt[1]:.4f}]')
print(f'rho(DD,LPF)={rho_dd:.4f} CI[{ci_dd[0]:.4f},{ci_dd[1]:.4f}]')
json.dump({'n': len(rows), 'spearman_vt_lpf': rho_vt, 'ci_vt': ci_vt,
           'spearman_dd_lpf': rho_dd, 'ci_dd': ci_dd, 'B': B},
          open('data/scivbench/e1_bootstrap.json', 'w'), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
ax[0].scatter(lpf, vt, s=6, alpha=0.15, color='#2b6cb0')
ax[0].set_xlabel('expert LPF (human)'); ax[0].set_ylabel('VT (auto)')
try:
    clu=json.load(open('data/scivbench/e1_cluster_bootstrap.json'))
    lo,hi=clu['cluster_boot_ci']
    ci_txt=f"[cluster 95% CI {lo:.3f}, {hi:.3f}]"
except Exception:
    ci_txt=f"[video 95% CI {ci_vt[0]:.3f}, {ci_vt[1]:.3f}]"
ax[0].set_title(f'ρ={rho_vt:.3f}  {ci_txt}')
ax[0].grid(alpha=0.3)
metrics = json.load(open('data/scivbench/e1_human_join.json'))['metrics']['lpf']
names = {'VT': 'VT', 'dynamic_degree': 'DD', 'subject_consistency': 'SC',
         'background_consistency': 'BC', 'motion_smoothness': 'MS',
         'aesthetic_quality': 'AQ', 'imaging_quality': 'IQ'}
keys = ['VT', 'dynamic_degree', 'subject_consistency', 'background_consistency',
        'motion_smoothness', 'aesthetic_quality', 'imaging_quality']
vals = [metrics[k]['spearman'] for k in keys]
ax[1].bar([names[k] for k in keys], vals,
          color=['#c53030' if v < 0 else '#2b6cb0' for v in vals])
ax[1].axhline(0, color='k', lw=0.8)
ax[1].set_ylabel('Spearman ρ vs expert LPF')
ax[1].set_title('per-metric human correlation (n=2,392)')
ax[1].grid(alpha=0.3, axis='y')
fig.tight_layout(); fig.savefig('paper/assets/fig3_human_join.png'); fig.savefig('paper/assets/fig3_human_join.pdf'); plt.close(fig)
print('FIG3_OK')
