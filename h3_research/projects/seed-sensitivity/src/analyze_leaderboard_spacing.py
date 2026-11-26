"""Leaderboard spacing vs seed noise (uses released vt_scores.jsonl, 16 models).

Computes per-model mean VT, adjacent-rank gaps, and compares against the
measured within-prompt seed SD (H3 0.052, Wan 0.038). Also computes the
gap structure on the common-video subset to avoid coverage bias.
"""
import json, statistics
from collections import defaultdict

rows = [json.loads(l) for l in open('data/scivbench/vt_scores.jsonl')]
by_model = defaultdict(dict)
for r in rows:
    by_model[r['model']][r['video_id']] = r['VT']

means = {m: statistics.mean(d.values()) for m, d in by_model.items()}
order = sorted(means, key=lambda m: -means[m])
print('=== leaderboard (own coverage) ===')
for i, m in enumerate(order, 1):
    gap = means[order[i - 2]] - means[m] if i > 1 else float('nan')
    print(f'{i:>2} {m:<20} VT={means[m]:.4f} n={len(by_model[m]):>5} gap_to_prev={gap:.4f}')

# common-video subset across models with >=1000 videos (stable coverage)
big = [m for m in by_model if len(by_model[m]) >= 1000]
common = set.intersection(*[set(by_model[m]) for m in big])
print(f'\n=== common subset: {len(big)} models x {len(common)} videos ===')
m2 = {m: statistics.mean(by_model[m][v] for v in common) for m in big}
order2 = sorted(m2, key=lambda m: -m2[m])
gaps = []
for i, m in enumerate(order2, 1):
    gap = m2[order2[i - 2]] - m2[m] if i > 1 else None
    if gap is not None: gaps.append(gap)
    print(f'{i:>2} {m:<20} VT={m2[m]:.4f} gap={gap if gap is None else round(gap,4)}')
print('\nadjacent-gap summary: median', statistics.median(gaps),
      'max', max(gaps), 'min', min(gaps))
h3_rank = order2.index('MiniMax-H3') + 1 if 'MiniMax-H3' in order2 else None
print('H3 rank:', h3_rank, 'of', len(order2))

out = {'n_models': len(big), 'n_videos': len(common),
       'ranking': [{'rank': i + 1, 'model': m, 'vt': round(m2[m], 6),
                    'gap_to_prev': None if i == 0 else round(m2[order2[i - 1]] - m2[m], 6)}
                   for i, m in enumerate(order2)],
       'adjacent_gap_median': statistics.median(gaps),
       'adjacent_gap_max': max(gaps),
       'adjacent_gap_min': min(gaps),
       'h3_rank_common': h3_rank,
       'seed_sd_h3': 0.051963, 'seed_sd_wan': 0.038396}
json.dump(out, open('data/scivbench/leaderboard_spacing.json', 'w'), indent=2)
print('WROTE data/scivbench/leaderboard_spacing.json')
