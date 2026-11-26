#!/usr/bin/env python3
import json, random, statistics
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'data/scivbench'
pilot=P/'pilot_full'; vt=P/'vt_full/h3_base'
rows=[]
for d in pilot.glob('*_seed*'):
 if not (d/'source.json').exists(): continue
 s=json.load(open(d/'source.json')); q=json.load(open(d/'request.json')); v=json.load(open(vt/(d.name+'.json')))
 rows.append({'id':s['id'],'domain':s['domain'],'seed':q['seed'],**{k:v[k] for k in ['subject_consistency','background_consistency','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality','VT']}})
metrics=['subject_consistency','background_consistency','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality','VT']
def ci(values,B=10000):
 rng=random.Random(20260908); n=len(values); means=[]
 for _ in range(B): means.append(statistics.mean(values[rng.randrange(n)] for _ in range(n)))
 means.sort(); return statistics.mean(values),means[int(.025*B)],means[int(.975*B)]
out={'n':len(rows),'metrics':metrics,'overall':{},'domain':{},'seed':{},'paired_seed1_minus_seed0':{}}
for m in metrics: out['overall'][m]=dict(zip(['mean','ci95_lo','ci95_hi'],ci([r[m] for r in rows])))
for key in ['domain','seed']:
 for g in sorted(set(r[key] for r in rows)):
  z=[r for r in rows if r[key]==g];out[key][str(g)]={'n':len(z),**{m:dict(zip(['mean','ci95_lo','ci95_hi'],ci([r[m] for r in z]))) for m in metrics}}
for ident in sorted(set(r['id'] for r in rows)):
 z={r['seed']:r for r in rows if r['id']==ident}
 if 0 in z and 1 in z:
  for m in metrics: out['paired_seed1_minus_seed0'].setdefault(m,[]).append(z[1][m]-z[0][m])
for m,vals in out['paired_seed1_minus_seed0'].items(): out['paired_seed1_minus_seed0'][m]=dict(zip(['mean','ci95_lo','ci95_hi'],ci(vals)))
json.dump(out,open(P/'vt_stratified_analysis.json','w'),indent=2);print(json.dumps(out,indent=2))
