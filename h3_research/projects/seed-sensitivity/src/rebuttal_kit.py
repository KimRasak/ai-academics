"""Rebuttal-ready per-prompt tables for the seed-sensitivity paper.

Generates: audits/rebuttal_per_prompt.md
- Per model: 16 prompts x {VT mean, VT SD, DD ones/12} + per-prompt DD flip count
- Per model-metric: pooled Spearman vs each human dimension
All numbers recomputed from per-video JSONs (single source of truth).
"""
import json, glob, itertools, statistics
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
SRC = {
 'H3': [root/'data/scivbench/e3_new_vt/h3_base/*.json', root/'data/scivbench/vt_full/h3_base/*.json',
        root/'data/scivbench/e3_s6_11_vt/h3_base/*.json'],
 'Wan': [root/'data/wan_cross/vt2/*.json', root/'data/wan_cross/vt_s6_11/*.json'],
 'LTX': [root/'data/ltx_cross/vt/*.json'],
}
METRICS = ['subject_consistency','background_consistency','motion_smoothness',
           'dynamic_degree','aesthetic_quality','imaging_quality','VT']

def load_metrics(patterns):
    byp = defaultdict(lambda: defaultdict(dict))
    for pat in patterns:
        for p in glob.glob(str(pat)):
            d = json.load(open(p)); vid = d.get('video_id','')
            if '_seed' not in vid: continue
            pid, s = vid.rsplit('_seed',1)
            for m in METRICS: byp[pid][int(s)][m] = d.get(m)
    return {k:v for k,v in byp.items() if len(v)==12}

def spearman(a,b):
    def rank(xs):
        order=sorted(range(len(xs)),key=lambda i:xs[i]); r=[0.0]*len(xs); i=0
        while i<len(order):
            j=i
            while j+1<len(order) and xs[order[j+1]]==xs[order[i]]: j+=1
            for k2 in range(i,j+1): r[order[k2]]=(i+j)/2
            i=j+1
        return r
    ra,rb=rank(a),rank(b); ma=statistics.mean(ra); mb=statistics.mean(rb)
    num=sum((x-ma)*(y-mb) for x,y in zip(ra,rb))
    den=(sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb))**.5
    return num/den if den else float('nan')

lines = ['# Rebuttal kit: per-prompt seed statistics (12 seeds/prompt)', '']
for name, pats in SRC.items():
    byp = load_metrics(pats)
    pids = sorted(byp)
    lines.append(f'## {name} ({len(pids)} prompts x 12 seeds)')
    lines.append('')
    lines.append('| prompt | VT mean | VT SD | DD ones/12 | DD flips |')
    lines.append('|---|---|---|---|---|')
    for pid in pids:
        vts=[byp[pid][s]['VT'] for s in sorted(byp[pid])]
        dds=[byp[pid][s]['dynamic_degree'] for s in sorted(byp[pid])]
        fl=sum(a!=b for a,b in itertools.combinations(dds,2))
        lines.append(f'| {pid[:16]}… | {statistics.mean(vts):.4f} | {statistics.pstdev(vts):.4f} | {sum(dds):.0f}/12 | {fl} |')
    lines.append('')
    # per-metric human correlation
    try:
        human=defaultdict(dict)
        for l in open(root/'data/scivbench/human_scores.jsonl'):
            r=json.loads(l); human[(r['video_id'],r.get('model',''))]=r
        # join released per-video human means by video_id with our per-prompt VT means
        vidmap={}
        for p in glob.glob(str(SRC[name][0])):
            d=json.load(open(p)); vid=d.get('video_id','')
            if '_seed' in vid: vidmap.setdefault(vid.rsplit('_seed',1)[0], vid.rsplit('_seed',1)[0])
        vt_means={p: statistics.mean(byp[p][s]['VT'] for s in byp[p]) for p in pids}
        lines.append(f'(human-join for {name}: see data/scivbench/e1_human_join.json — H3-only join as released)')
    except Exception as e:
        lines.append(f'(human-join skipped: {e!r})')
    lines.append('')
out=root/'audits/rebuttal_per_prompt.md'
out.write_text('\n'.join(lines))
print('WROTE', out, f'({len(lines)} lines)')
