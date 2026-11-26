#!/usr/bin/env python3
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'data/scivbench'; # select released rows by IDs from local index
ids={json.loads(x)['video_id'] for x in open(root/'index.jsonl') if (lambda r:r.get('testmini') and r.get('model')=='MiniMax-H3')(json.loads(x))}
rel={r['video_id']:r for r in [json.loads(x) for x in open(root/'vt_scores.jsonl')] if r.get('model')=='MiniMax-H3' and r['video_id'] in ids}
loc={p.stem:json.load(open(p)) for p in (root/'released_vt/MiniMax-H3').glob('*.json') if p.name not in ['aggregate.json']}
ms=['subject_consistency','background_consistency','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality','VT']; rows=[]
for k,v in loc.items():
 if k in rel: rows.append({'id':k,**{m:v[m]-rel[k][m] for m in ms}})
res={'n':len(rows),'mean_delta':{m:statistics.mean(r[m] for r in rows) for m in ms},'median_abs_delta':{m:statistics.median(abs(r[m]) for r in rows) for m in ms},'exact_match_dynamic_degree':sum(abs(r['dynamic_degree'])<1e-12 for r in rows)/len(rows)}
json.dump(res,open(root/'released_vt_comparison.json','w'),indent=2);print(json.dumps(res,indent=2))
