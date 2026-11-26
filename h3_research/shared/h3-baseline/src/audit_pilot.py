#!/usr/bin/env python3
import argparse,json,subprocess,hashlib
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args(); root=Path(a.root); rows=[]
 for d in sorted(root.glob('*_seed*')):
  if not d.is_dir(): continue
  row={'case':d.name,'exists':(d/'output.mp4').exists()}
  if row['exists']:
   p=d/'output.mp4'; row['bytes']=p.stat().st_size; row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest(); q=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)],capture_output=True,text=True); row['ffprobe_ok']=q.returncode==0
   if row['ffprobe_ok']:
    x=json.loads(q.stdout);row['duration']=float(x['format']['duration']); row['streams']=[{k:s.get(k) for k in ['codec_type','codec_name','width','height','r_frame_rate','sample_rate','channels']} for s in x['streams']]
  if (d/'error.json').exists(): row['error']=json.load(open(d/'error.json'))
  rows.append(row)
 out={'root':str(root),'cases':len(rows),'completed':sum(r.get('ffprobe_ok',False) for r in rows),'missing':sum(not r['exists'] for r in rows),'bad':sum(r.get('exists',False) and not r.get('ffprobe_ok',False) for r in rows),'rows':rows}; print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
