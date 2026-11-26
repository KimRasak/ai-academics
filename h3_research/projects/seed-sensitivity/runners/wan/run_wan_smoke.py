import os,sys,runpy
p="/ssd2/jinzili/wan_fun_src/examples/wan2.1_fun/predict_t2v.py"
s=open(p).read()
s=s.replace('GPU_memory_mode     = "sequential_cpu_offload"','GPU_memory_mode     = "model_cpu_offload"')
s=s.replace('model_name          = "models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP"','model_name          = "/gluster_hdd_cv/user/jinzili/weights/Wan2.1-Fun-V1.1-1.3B-InP"')
# Full protocol: keep official defaults (81 frames, 50 steps, fps 16, skip 5) => ~5s @16FPS
# Smoke protocol (WAN_FULL unset): reduced 17 frames / 4 steps / skip 2
FULL = os.environ.get("WAN_FULL","0") == "1"
if not FULL:
    s=s.replace('video_length        = 81','video_length        = 17')
    s=s.replace('num_inference_steps = 50','num_inference_steps = 4')
    s=s.replace('num_skip_start_steps = 5','num_skip_start_steps = 2')
s=s.replace('seed                = 43','seed                = '+os.environ.get("WAN_SEED","0"))
s=s.replace('save_path           = "samples/wan-videos-fun-t2v"','save_path           = os.environ.get("WAN_OUT","/ssd2/jinzili/wan_smoke")')
open('/tmp/wan_smoke_exec.py','w').write(s)
runpy.run_path('/tmp/wan_smoke_exec.py',run_name='__main__')
