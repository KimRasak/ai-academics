"""E4: Wan2.1-Fun cross-model seed-power analysis.

Reads per-video VT JSON results for the 16 prompts x 6 seeds Wan batch and
computes the same statistics as the H3 E3 seed-power analysis:
  - pairwise DD flip rate (same prompt, different seed, DD 0<->1)
  - per-prompt any-flip indicator
  - within-prompt SD of VT and DD
Mirrors analyze_e3_seed_power.py methodology. No training involved.
"""
import json, glob, os, sys, itertools, statistics

res_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/wan_cross/vt'
rows = []
for p in glob.glob(os.path.join(res_dir, '*.json')):
    d = json.load(open(p))
    vid = d.get('video_id', '')
    if '_seed' not in vid:
        continue  # aggregate/summary files have no per-video id
    if '_seed' not in vid:
        continue
    pid, seed = vid.rsplit('_seed', 1)
    rows.append({'pid': pid, 'seed': int(seed),
                 'VT': d['VT'], 'DD': d['dynamic_degree']})

prompts = {}
for r in rows:
    prompts.setdefault(r['pid'], []).append(r)

pairs = list(itertools.combinations(range(6), 2))
n_pairs = len(prompts) * len(pairs)
dd_flips = 0
any_flip_prompts = 0
vt_sds, dd_sds, vt_means = [], [], []
for pid, rs in prompts.items():
    by_seed = {r['seed']: r for r in rs}
    if len(by_seed) < 2:
        continue
    flip = False
    for a, b in pairs:
        if a in by_seed and b in by_seed:
            if by_seed[a]['DD'] != by_seed[b]['DD']:
                dd_flips += 1
                flip = True
    any_flip_prompts += flip
    vt_sds.append(statistics.pstdev([r['VT'] for r in by_seed.values()]))
    dd_sds.append(statistics.pstdev([r['DD'] for r in by_seed.values()]))
    vt_means.append(statistics.mean([r['VT'] for r in by_seed.values()]))

n_eval_pairs = sum(len(pairs) for p in prompts.values() if len(p) >= 2)
out = {
    'n_videos': len(rows),
    'n_prompts': len(prompts),
    'dd_pairwise_flip': dd_flips,
    'n_eval_pairs': n_eval_pairs,
    'dd_flip_rate': dd_flips / n_eval_pairs if n_eval_pairs else None,
    'prompts_any_flip': any_flip_prompts,
    'n_prompts_multi': sum(1 for p in prompts.values() if len(p) >= 2),
    'vt_within_prompt_sd_mean': statistics.mean(vt_sds) if vt_sds else None,
    'vt_within_prompt_sd_max': max(vt_sds) if vt_sds else None,
    'dd_within_prompt_sd_mean': statistics.mean(dd_sds) if dd_sds else None,
    'vt_overall_mean': statistics.mean([r['VT'] for r in rows]),
    'dd_overall_mean': statistics.mean([r['DD'] for r in rows]),
}
json.dump(out, open('data/wan_cross/e4_wan_seed_power.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
