#!/usr/bin/env python3
import argparse,json,time,urllib.request,urllib.error,hashlib,subprocess
from pathlib import Path

def call(url,payload=None,timeout=60):
 data=None if payload is None else json.dumps(payload).encode(); req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'} if data else {},method='POST' if data else 'GET')
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read())
def atomic(p,x):
 t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n'); t.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',required=True);ap.add_argument('--base-url',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seeds',default='0');ap.add_argument('--duration',type=float,default=5);ap.add_argument('--poll',type=float,default=5);ap.add_argument('--deadline',type=float,default=1200);a=ap.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True); seeds=[int(x) for x in a.seeds.split(',')]
 rows=[json.loads(x) for x in open(a.pilot) if x.strip()]; atomic(out/'manifest.json',{'pilot':str(a.pilot),'base_url':a.base_url,'seeds':seeds,'count':len(rows)*len(seeds),'records':[{'id':r['id'],'domain':r['domain'],'prompt_chars':len(r['posted_prompt'])} for r in rows]})
 for r in rows:
  for seed in seeds:
   case=f"{r['id']}_seed{seed}"; d=out/case
   if (d/'runner_summary.json').exists(): continue
   d.mkdir(parents=True,exist_ok=True); payload={'task':'t2va','prompt':r['posted_prompt'],'conditions':[],'target':{'short_edge':768,'aspect_ratio':'16:9','duration_seconds':a.duration},'seed':seed}; atomic(d/'source.json',r);atomic(d/'request.json',payload)
   try:
    submit=call(a.base_url.rstrip('/')+'/v1/videos',payload);atomic(d/'submit_response.json',submit); vid=submit['id']; hist=[]; deadline=time.monotonic()+a.deadline
    while True:
     st=call(a.base_url.rstrip('/')+f'/v1/videos/{vid}'); hist.append(st);atomic(d/'status_history.json',hist); state=str(st.get('status','')).lower()
     if state in {'completed','failed','cancelled','canceled'}:break
     if time.monotonic()>deadline:raise TimeoutError('deadline')
     time.sleep(a.poll)
    atomic(d/'final_status.json',hist[-1])
    if state!='completed': raise RuntimeError(state)
    with urllib.request.urlopen(a.base_url.rstrip('/')+f'/v1/videos/{vid}/content',timeout=300) as q:data=q.read()
    (d/'output.mp4').write_bytes(data); atomic(d/'runner_summary.json',{'case':case,'video_id':vid,'status':state,'polls':len(hist),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
   except Exception as e:
    atomic(d/'error.json',{'case':case,'error':repr(e)}); print('ERROR',case,repr(e),flush=True)
   print('DONE',case,flush=True)
if __name__=='__main__':main()
