#!/usr/bin/env python3
import json,statistics,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'data/scivbench'; base=root/'pilot_full'; new=root/'e3_new_vt/h3_base'; metrics=['subject_consistency','background_consistency','motion_smoothness','dynamic_degree','aesthetic_quality','imaging_quality','VT']; rows=[]
for p in new.glob('*.json'):
 if p.name in ('aggregate.json',):continue
 v=json.load(open(p)); ident=p.stem.rsplit('_seed',1)[0]; seed=int(p.stem.rsplit('_seed',1)[1]); rows.append({'id':ident,'seed':seed,**{m:v[m] for m in metrics}})
for seed in (0,1):
 for d in base.glob(f'*_seed{seed}'):
  ident=d.name.rsplit('_seed',1)[0]; v=json.load(open(root/'vt_full/h3_base'/(d.name+'.json'))); rows.append({'id':ident,'seed':seed,**{m:v[m] for m in metrics}})
ids=sorted(set(r['id'] for r in rows)); out={'n':len(rows),'complete_cases':len(ids),'by_seed':{},'per_prompt':{},'mdd':{}}
for k in range(1,7):
 vals=[]
 for ident in ids:
  z=[r for r in rows if r['id']==ident and r['seed']<k]
  if len(z)==k: vals.append(statistics.mean(r['VT'] for r in z))
 out['by_seed'][k]={'n_prompts':len(vals),'mean':statistics.mean(vals),'sd':statistics.stdev(vals),'se':statistics.stdev(vals)/math.sqrt(len(vals))}
for ident in ids:
 z={r['seed']:r for r in rows if r['id']==ident};out['per_prompt'][ident]={}
 for m in metrics:out['per_prompt'][ident][m]={'mean':statistics.mean(z[s][m] for s in z),'sd':statistics.stdev(z[s][m] for s in z)}
for m in metrics:
 vals=[out['per_prompt'][i][m]['sd'] for i in ids];out['mdd'][m]={'mean_within_prompt_sd':statistics.mean(vals),'max_within_prompt_sd':max(vals)}
flips=pairs=0; anyflip=0
for ident in ids:
 z={r['seed']:r for r in rows if r['id']==ident}; local=False
 for a in z:
  for b in z:
   if a<b:pairs+=1; changed=z[a]['dynamic_degree']!=z[b]['dynamic_degree']; flips+=changed; local|=changed
 anyflip+=local
out['dd_pairwise_flip_rate']={'flips':flips,'pairs':pairs,'rate':flips/pairs}
out['dd_prompt_any_flip_rate']={'prompts_with_flip':anyflip,'prompts':len(ids),'rate':anyflip/len(ids)}
json.dump(out,open(root/'e3_seed_power.json','w'),indent=2);print(json.dumps(out,indent=2))
