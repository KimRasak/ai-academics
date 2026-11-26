"""Resolution analysis: how many seeds to resolve adjacent-model VT gaps?

Model: paired comparison of two models on the same P prompts with s seeds each.
The prompt main effect cancels in the paired difference; residual variance is
the seed-by-prompt interaction, estimated by the within-prompt across-seed SD
(sigma_sp) measured in E3 (H3) and E4 (Wan).

SE(mean diff) = sigma_sp * sqrt(2 / (P * s))
Resolve a gap g at 95% iff 1.96 * SE <= g  =>  s >= 2 * (1.96 * sigma_sp / g)^2 / P

Uses measured sigma_sp from e3_seed_power.json and e4_wan_seed_power.json.
"""
import json, math

h3 = json.load(open('data/scivbench/e3_seed_power.json'))
e4 = json.load(open('data/wan_cross/e4_wan_seed_power.json'))
lb = json.load(open('data/scivbench/leaderboard_spacing.json'))

sigma_h3 = h3['mdd']['VT']['mean_within_prompt_sd']        # 0.051963
sigma_wan = e4['vt_within_prompt_sd_mean']                  # 0.038396
P = 16
g_med = lb['adjacent_gap_median']                           # 0.0066
g_min = lb['adjacent_gap_min']                              # 0.0029

def se(sigma, P, s):
    return sigma * math.sqrt(2.0 / (P * s))

out = {'paired_SE_per_seed': {}, 'seeds_to_resolve': {}}
for name, sigma in [('H3', sigma_h3), ('Wan', sigma_wan)]:
    per = {s: round(se(sigma, P, s), 5) for s in (1, 2, 4, 6, 16, 32)}
    out['paired_SE_per_seed'][name] = {'sigma_sp': round(sigma, 6), 'SE_by_s': per}
    s_med = 2 * (1.96 * sigma / g_med) ** 2 / P
    s_min = 2 * (1.96 * sigma / g_min) ** 2 / P
    out['seeds_to_resolve'][name] = {'median_gap': g_med, 'seeds': math.ceil(s_med),
                                     'min_gap': g_min, 'seeds_min_gap': math.ceil(s_min)}
    print(f'{name}: sigma_sp={sigma:.4f}')
    for s in (1, 2, 4, 6):
        hw = 1.96 * se(sigma, P, s)
        print(f'  s={s}: SE={se(sigma,P,s):.4f}  95% half-width=±{hw:.4f}  (median gap {g_med:.4f})')
    print(f'  seeds to resolve median gap: {math.ceil(s_med)}; to resolve min gap: {math.ceil(s_min)}')

json.dump(out, open('data/scivbench/resolution_analysis.json', 'w'), indent=2)
print('WROTE data/scivbench/resolution_analysis.json')
