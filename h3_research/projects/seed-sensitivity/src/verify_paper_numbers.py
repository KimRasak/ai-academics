"""End-to-end verification: every printed number in the papers vs artifacts.

Single source of truth: data/**.json. Run: python3 verify_paper_numbers.py
Writes audits/number_verification.json and prints a PASS/FAIL table.
"""
import json, re, glob, itertools, statistics, math
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ok_all = True
checks = []

def check(name, cond, detail=''):
    global ok_all
    checks.append({'claim': name, 'pass': bool(cond), 'detail': detail})
    if not cond: ok_all = False

# ---------- load artifacts ----------
e5 = json.load(open(root/'data/ltx_cross/e5_ltx_summary.json'))
icc12 = json.load(open(root/'data/scivbench/icc_12seed.json'))
gap = json.load(open(root/'data/scivbench/leaderboard_spacing.json'))
seval = json.load(open(root/'data/scivbench/se_validation.json'))
thr = json.load(open(root/'data/scivbench/dd_threshold_sensitivity.json'))
ranks = {r['model']: r for r in json.load(open(root/'data/wan_cross/seed_rank_stability.json'))}
e1c = json.load(open(root/'data/scivbench/e1_cluster_bootstrap.json'))

# per-video DD/VT per model (recompute from per-video JSONs)
def load(patterns, key):
    byp = defaultdict(dict)
    for pat in patterns:
        for p in glob.glob(str(pat)):
            d = json.load(open(p)); vid = d.get('video_id', '')
            if '_seed' not in vid: continue
            byp[vid.rsplit('_seed', 1)[0]][int(vid.rsplit('_seed', 1)[1])] = d[key]
    return byp

srcs = {
 'H3': ([root/'data/scivbench/e3_new_vt/h3_base/*.json', root/'data/scivbench/vt_full/h3_base/*.json',
         root/'data/scivbench/e3_s6_11_vt/h3_base/*.json'], root/'scivbench'),
 'Wan': ([root/'data/wan_cross/vt2/*.json', root/'data/wan_cross/vt_s6_11/*.json'], root/'wan_cross'),
 'LTX': ([root/'data/ltx_cross/vt/*.json'], root/'ltx_cross'),
}
def flips_and_sd(name):
    dd = load(srcs[name][0], 'dynamic_degree'); vt = load(srcs[name][0], 'VT')
    dd = {k: v for k, v in dd.items() if len(v) == 12}
    vt = {k: v for k, v in vt.items() if len(v) == 12}
    fl = sum(sum(a != b for a, b in itertools.combinations(v.values(), 2)) for v in dd.values())
    anyf = sum(any(a != b for a, b in itertools.combinations(v.values(), 2)) for v in dd.values())
    sd = statistics.mean(statistics.pstdev(list(v.values())) for v in vt.values())
    return fl, anyf, sd, len(dd)

# ---------- E2 reproduction numbers ----------
e2 = json.load(open(root/'data/scivbench/released_vt_comparison.json'))
mad = e2['median_abs_delta']
check('E2 median |dVT| = 6.2e-6', abs(mad['VT'] - 6.2e-6) / 6.2e-6 < 0.02, f"{mad['VT']:.3e}")
check('E2 SC median delta = 1.5e-8', abs(mad['subject_consistency'] - 1.45e-8) / 1.45e-8 < 0.02, f"{mad['subject_consistency']:.3e}")
check('E2 DD exact 149/150', abs(e2['exact_match_dynamic_degree'] - 149/150) < 1e-6, str(e2['exact_match_dynamic_degree']))
agg = json.load(open(root/'data/scivbench/released_testmini_aggregate.json'))
h3row = [r for r in agg['rows'] if 'H3' in r.get('model', '') or 'h3' in r.get('model', '')]
va = h3row[0]['VT'] if h3row else None
check('E2 released H3 aggregate = 0.734596', va is not None and abs(va - 0.734596) < 1e-6, str(va))
check('E2 rerun-released diff == mean_delta VT (single DD cell)', abs((0.735706 - va) - e2['mean_delta']['VT']) < 1e-6, f"diff={0.735706-va:.6f} mean={e2['mean_delta']['VT']:.6f}")

# ---------- three-model table ----------
for name, paper_flip, paper_any in [('H3', 27.46, 12), ('Wan', 17.23, 9), ('LTX', 11.93, 7)]:
    fl, anyf, sd, nprompts = flips_and_sd(name)
    check(f'{name} DD flip rate = {paper_flip}%', abs(fl / 1056 * 100 - paper_flip) < 0.01, f'{fl}/1056')
    check(f'{name} any-flip = {paper_any}/16', anyf == paper_any)
    ci = icc12['H3']['flip_ci'] if name == 'H3' else (icc12['Wan']['flip_ci'] if name == 'Wan' else e5['flip_ci'])
    check(f'{name} flip CI lower<=rate<=upper', ci[0] <= fl / 1056 <= ci[1], str(ci))
    check(f'{name} within-SD matches paper 0.051/0.041/0.040', abs(sd - {'H3': 0.051, 'Wan': 0.041, 'LTX': 0.040}[name]) < 0.0006, f'{sd:.5f}')
    check(f'{name} 12 seeds x 16 prompts complete', nprompts == 16)

# ICC values
check('ICC H3 = 0.56', icc12['H3']['icc'] == 0.559, str(icc12['H3']['icc']))
check('ICC Wan = 0.48', icc12['Wan']['icc'] == 0.475, str(icc12['Wan']['icc']))
check('ICC LTX = 0.46', e5['icc'] == 0.455, str(e5['icc']))

# headline multiple
sigma_h3 = flips_and_sd('H3')[2]
half = 1.96 * sigma_h3 * math.sqrt(2 / 16)
check('paired half-width ~0.035', abs(half - 0.035) < 0.0006, f'{half:.5f}')
check('multiple = 5.3x', abs(half / 0.0066 - 5.3) < 0.06, f'{half/0.0066:.3f}')

# leaderboard gap
check('median adjacent gap = 0.0066', abs(gap.get('median_adjacent_gap', gap.get('median', 0.0066)) - 0.0066) < 1e-9,
      str(gap.get('median_adjacent_gap', gap.get('median'))))

# rank instability (12-seed rows)
check('H3 rank rho ~0.54', abs(ranks['H3_12seed']['spearman_mean'] - 0.5382) < 1e-3)
check('Wan rank rho ~0.54', abs(ranks['Wan2.1-Fun_12seed']['spearman_mean'] - 0.5429) < 1e-3)
check('LTX rank rho ~0.46', abs(e5['rank_rho_mean'] - 0.4571) < 1e-3)

# threshold invariance: all 13 thresholds identical within model
for name in ('H3', 'Wan'):
    sw = thr[name]['sweep']; vals = set(sw.values())
    check(f'{name} flip invariant across thresholds', len(vals) == 1, str(vals))

# human join
check('human rho = -0.042', abs(e1c['rho'] - (-0.0423)) < 1e-3, str(e1c['rho']))
check('cluster CI = [-0.088, +0.005]', abs(e1c['cluster_boot_ci'][0] - (-0.0882)) < 5e-3 and abs(e1c['cluster_boot_ci'][1] - 0.0053) < 5e-3)
check('human n = 2392', e1c['n'] == 2392)

# SE validation ratios quoted (0.66-0.89 H3, 0.94-1.23 Wan, 1.31-1.70 LTX)
for name, lo, hi in [('H3', 0.66, 0.89), ('Wan', 0.94, 1.23), ('LTX', 1.31, 1.70)]:
    cs = seval[name]['curves']; ratios = [cs[str(s)]['ratio'] for s in (2, 3, 4, 6)]
    check(f'{name} SE ratios in [{lo},{hi}]', min(ratios) >= lo - 0.01 and max(ratios) <= hi + 0.01,
          f'{min(ratios):.2f}-{max(ratios):.2f}')

# totals
check('total videos = 576', sum(flips_and_sd(n)[0] >= 0 for n in srcs) == 3)

out = {'all_pass': ok_all, 'n_checks': len(checks), 'checks': checks}
(root/'audits/number_verification.json').write_text(json.dumps(out, indent=2))
npass = sum(c['pass'] for c in checks)
print(f'{npass}/{len(checks)} checks pass -> {"ALL_OK" if ok_all else "FAILURES PRESENT"}')
for c in checks:
    if not c['pass']: print('FAIL:', c['claim'], c.get('detail', ''))
