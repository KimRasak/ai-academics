#!/usr/bin/env python3
import argparse,json,urllib.request,statistics
from pathlib import Path

def rank_values(x):
 order=sorted(range(len(x)),key=lambda i:x[i]); ranks=[0.0]*len(x); i=0
 while i<len(order):
  j=i+1
  while j<len(order) and x[order[j]]==x[order[i]]: j+=1
  avg=(i+j-1)/2
  for k in range(i,j): ranks[order[k]]=avg
  i=j
 return ranks
def spearman(a,b):
 x=rank_values(a); y=rank_values(b); ma=sum(x)/len(x);mb=sum(y)/len(y); num=sum((u-ma)*(v-mb) for u,v in zip(x,y)); den=(sum((u-ma)**2 for u in x)*sum((v-mb)**2 for v in y))**.5; return num/den if den else 0

def dl(url,p):
 if not p.exists(): p.write_bytes(urllib.request.urlopen(url,timeout=120).read())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args(); root=Path(a.out).parent; root.mkdir(parents=True,exist_ok=True); base='https://huggingface.co/datasets/Sci-VBench/Sci-VBench-Videos/resolve/main/'
 files={k:root/f for k,f in [('human','human_scores.jsonl'),('index','index.jsonl')]}
 for p in files.values(): dl(base+p.name,p)
 human=[json.loads(x) for x in open(files['human'])]; idx=[json.loads(x) for x in open(files['index'])]; mini={x['video_id'] for x in idx if x.get('testmini')}
 vt=[json.loads(x) for x in open(root/'vt_scores.jsonl')] if (root/'vt_scores.jsonl').exists() else []
 hm={(x['model'],x['video_id']):x for x in human}; vm={(x['model'],x['video_id']):x for x in vt}; joined=[]
 for k,h in hm.items():
  if k in vm: joined.append({**h,**{f'vt_{m}':vm[k].get(m) for m in ['VT','dynamic_degree','subject_consistency','background_consistency','motion_smoothness','aesthetic_quality','imaging_quality']}})
 res={'human_rows':len(human),'index_rows':len(idx),'mini_video_ids':len(mini),'joined_rows':len(joined),'metrics':{}}
 for target in ['lpf','pg','scc','sc']:
  res['metrics'][target]={}
  for m in ['VT','dynamic_degree','subject_consistency','background_consistency','motion_smoothness','aesthetic_quality','imaging_quality']:
   z=[r for r in joined if r.get(target) is not None and r.get('vt_'+m) is not None]
   if len(z)>2: res['metrics'][target][m]={'n':len(z),'spearman':spearman([r[target] for r in z],[r['vt_'+m] for r in z])}
 json.dump(res,open(a.out,'w'),indent=2);print(json.dumps(res,indent=2))
if __name__=='__main__':main()
