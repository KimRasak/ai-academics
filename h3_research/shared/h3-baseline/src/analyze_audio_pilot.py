#!/usr/bin/env python3
import argparse,json,subprocess,io
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--out',required=True);a=ap.parse_args(); rows=[]
 for p in sorted(Path(a.root).glob('*.mp4')):
  cmd=['/gluster_hdd_cv/user/jinzili/env/sglang310/bin/ffmpeg','-v','error','-i',str(p),'-vn','-ac','2','-ar','32000','-f','f32le','pipe:1']; q=subprocess.run(cmd,stdout=subprocess.PIPE,check=True); x=np.frombuffer(q.stdout,dtype=np.float32).reshape(-1,2); rms=np.sqrt(np.mean(x*x,axis=0)); peak=np.max(np.abs(x),axis=0); frames=np.sqrt(np.mean(x*x,axis=1)); rows.append({'video':p.stem,'samples':len(x),'duration_s':len(x)/32000,'rms_l':float(rms[0]),'rms_r':float(rms[1]),'peak_l':float(peak[0]),'peak_r':float(peak[1]),'silence_fraction_rms_lt_1e-3':float(np.mean(frames<1e-3)),'lr_corr':float(np.corrcoef(x[:,0],x[:,1])[0,1])})
 summary={'n':len(rows)}
 for k in ['duration_s','rms_l','rms_r','peak_l','peak_r','silence_fraction_rms_lt_1e-3','lr_corr']:
  z=np.array([r[k] for r in rows]);summary[k]={'mean':float(z.mean()),'min':float(z.min()),'max':float(z.max())}
 json.dump({'summary':summary,'rows':rows},open(a.out,'w'),indent=2);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
