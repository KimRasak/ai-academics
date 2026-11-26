"""Empirical validation of the resolution model with 12 seeds (seeds 0..11).

Correct test: for s in {2,3,4,6,12}, draw random s-subsets of the 12 seeds,
compute the 16-prompt mean VT for each subset, and take the SD across subsets.
Because every subset uses all 16 prompts, between-prompt variance cancels and
the SD isolates seed-averaging noise. Compare to sigma_sp/sqrt(16*s)
(i.i.d. seed noise prediction) using sigma_sp estimated from the same 12 seeds.
"""
import json, glob, random, statistics, math
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'data'

def load(patterns):
    byp = defaultdict(dict)
    for pat in patterns:
        for p in glob.glob(str(pat)):
            d = json.load(open(p)); vid = d.get('video_id', '')
            if '_seed' not in vid: continue
            byp[vid.rsplit('_seed', 1)[0]][int(vid.rsplit('_seed', 1)[1])] = d['VT']
    return byp

def empirical_se_subsets(byp, s, B=500, rng=None):
    rng = rng or random.Random(0)
    means = []
    pids = sorted(byp)
    for _ in range(B):
        sub = rng.sample(range(12), s)
        means.append(statistics.mean(statistics.mean(byp[p][k] for k in sub) for p in pids))
    return statistics.pstdev(means)

def main():
    h3 = load([root / 'scivbench/e3_new_vt/h3_base/*.json',
               root / 'scivbench/vt_full/h3_base/*.json',
               root / 'scivbench/e3_s6_11_vt/h3_base/*.json'])
    wan = load([root / 'wan_cross/vt2/*.json',
                root / 'wan_cross/vt_s6_11/*.json'])
    out = {}
    for name, byp in [('H3', h3), ('Wan', wan)]:
        bad = [p for p, v in byp.items() if len(v) != 12]
        if bad:
            print(f'{name}: incomplete prompts {len(bad)} (need 12 seeds each); skip'); continue
        sigma = statistics.mean(statistics.pstdev(list(v.values())) for v in byp.values())
        rng = random.Random(0)
        res = {}
        for s in (2, 3, 4, 6, 12):
            emp = empirical_se_subsets(byp, s, rng=rng)
            pred = sigma / math.sqrt(16 * s)
            res[s] = {'empirical': round(emp, 5), 'predicted': round(pred, 5),
                      'ratio': round(emp / pred, 3)}
        out[name] = {'sigma_sp_12seeds': round(sigma, 6), 'curves': res}
        print(name, 'sigma_sp(12seeds)=', round(sigma, 5))
        for s, r in res.items():
            print(f'  s={s:>2}: empirical={r["empirical"]:.5f} predicted={r["predicted"]:.5f} ratio={r["ratio"]}')
    json.dump(out, open(root / 'scivbench/se_validation.json', 'w'), indent=2)
    print('WROTE', root / 'scivbench/se_validation.json')

if __name__ == '__main__':
    main()
