import os,sys,runpy,json,subprocess,base64,re
pilot=[json.loads(l) for l in open('/ssd2/jinzili/e3_seed/scivbench/pilot16.jsonl')]
seeds=[int(x) for x in os.environ.get('WAN_SEEDS','0,1,2,3,4,5').split(',')]
outdir=os.environ.get('WAN_OUT','/ssd2/jinzili/wan_batch')
os.makedirs(outdir,exist_ok=True)
tmpl=open('/ssd2/jinzili/wan_fun_src/examples/wan2.1_fun/predict_t2v.py').read()
idx=0
for r in pilot:
    for seed in seeds:
        name=f"{r['id']}_seed{seed}"
        tgt=f"{outdir}/{name}.mp4"
        if os.path.exists(tgt) and os.path.getsize(tgt)>10000:
            print('SKIP',name,flush=True); continue
        s=tmpl
        s=s.replace('GPU_memory_mode     = "sequential_cpu_offload"','GPU_memory_mode     = "model_cpu_offload"')
        s=s.replace('model_name          = "models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP"','model_name          = "/gluster_hdd_cv/user/jinzili/weights/Wan2.1-Fun-V1.1-1.3B-InP"')
        # robust prompt swap: anchor ^prompt\s*= (not negative_prompt), assert it applied
        new_prompt_line='prompt = '+json.dumps(r['posted_prompt'], ensure_ascii=True)
        s2,nsub=re.subn(r'^prompt\s*=\s*".*"', new_prompt_line.replace('\\','\\\\'), s, count=1, flags=re.M)
        if nsub!=1 or new_prompt_line not in s2:
            print('FATAL prompt replace failed',name,flush=True); sys.exit(2)
        s=s2
        s=s.replace('seed                = 43','seed                = '+str(seed))
        s=s.replace('save_path           = "samples/wan-videos-fun-t2v"','save_path           = "/ssd2/jinzili/wan_batch_tmp"')
        os.makedirs('/ssd2/jinzili/wan_batch_tmp',exist_ok=True)
        for f in os.listdir('/ssd2/jinzili/wan_batch_tmp'): os.remove('/ssd2/jinzili/wan_batch_tmp/'+f)
        open('/tmp/wan_batch_exec.py','w').write(s)
        # exec script lives in /tmp so the official path bootstrap can't find videox_fun; inject src root directly
        s=s.replace("for project_root in project_roots:","for project_root in project_roots:\n    sys.path.insert(0, '/ssd2/jinzili/wan_fun_src') if '/ssd2/jinzili/wan_fun_src' not in sys.path else None")
        open('/tmp/wan_batch_exec.py','w').write(s)
        print('RUN',name,flush=True)
        env=dict(os.environ); env['PYTHONPATH']='/ssd2/jinzili/wan_fun_src:'+env.get('PYTHONPATH','')
        rc=subprocess.run([sys.executable,'-u','/tmp/wan_batch_exec.py'],env=env,cwd='/ssd2/jinzili/wan_fun_src').returncode
        outs=[f for f in os.listdir('/ssd2/jinzili/wan_batch_tmp') if f.endswith('.mp4')]
        if rc==0 and outs:
            os.replace('/ssd2/jinzili/wan_batch_tmp/'+outs[0],tgt)
            print('DONE',name,os.path.getsize(tgt),flush=True)
        else:
            print('FAIL',name,'rc=',rc,flush=True)
        idx+=1
print('ALL_DONE',idx,flush=True)
