#!/usr/bin/env python3
import argparse,json,statistics
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--out',required=True);a=ap.parse_args();rows=[]
 for d in sorted(Path(a.root).glob('*_seed*')):
  if not (d/'runner_summary.json').exists(): continue
  src=json.load(open(d/'source.json')); req=json.load(open(d/'request.json')); fin=json.load(open(d/'final_status.json')); run=json.load(open(d/'runner_summary.json'))
  rows.append({'case':d.name,'id':src['id'],'domain':src['domain'],'prompt_chars':len(src['posted_prompt']),'seed':req['seed'],'status':run['status'],'polls':run['polls'],'bytes':run['bytes'],'sha256':run['sha256'],'service_inference_s':fin.get('inference_time_s'),'service_peak_memory_mb':fin.get('peak_memory_mb'),'requested_duration':req['target']['duration_seconds']})
 summary={'cases':len(rows),'completed':sum(x['status']=='completed' for x in rows),'domains':{},'seeds':{},'service_inference_s':{'n':sum(x['service_inference_s'] is not None for x in rows)}}
 for k in ['domain','seed']:
  summary[k+'s']={str(v):sum(x[k]==v for x in rows) for v in sorted(set(x[k] for x in rows))}
 vals=[x['service_inference_s'] for x in rows if x['service_inference_s'] is not None]
 if vals: summary['service_inference_s'].update({'min':min(vals),'median':statistics.median(vals),'max':max(vals),'mean':statistics.mean(vals)})
 json.dump({'summary':summary,'rows':rows},open(a.out,'w'),indent=2); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
