#!/usr/bin/env python3
"""LTX-Video batch runner: 16 Sci-VBench pilot prompts x seeds, mirrors run_wan_batch.py protocol.
Env: WAN_FULL=1 WAN_SEEDS=csv WAN_OUT=dir  (defaults: smoke mode 1 video)
Resume-safe: skips (id,seed) whose mp4 already exists. Asserts prompt content non-empty.
"""
import os, sys, json, re, gc, hashlib, time
import torch
from pathlib import Path
sys.path.insert(0, '/ssd2/jinzili/ltx_src')
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

SEEDS = [int(x) for x in os.environ.get('WAN_SEEDS', '0').split(',') if x != '']
OUT = Path(os.environ.get('WAN_OUT', '/ssd2/jinzili/ltx_smoke'))
FULL = os.environ.get('WAN_FULL', '0') == '1'
OUT.mkdir(parents=True, exist_ok=True)
MODEL = '/ssd2/jinzili/ltx_video'

pilot = [json.loads(x) for x in open('/ssd2/jinzili/e3_seed/scivbench/pilot16.jsonl') if x.strip()]
prompts = {r['id']: r['posted_prompt'] for r in pilot}
assert len(prompts) == 16, f'expected 16 prompts, got {len(prompts)}'

if not FULL:
    prompts = {list(prompts)[0]: list(prompts.values())[0]}
    SEEDS = [int(os.environ.get('WAN_SEED', '0'))]

pipe = LTXPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
print('LTX loaded', flush=True)

total = fail = 0
for pid, prompt in prompts.items():
    vid = hashlib.sha1(pid.encode()).hexdigest()[:40]
    assert re.search(r'\w', prompt) and len(prompt) > 20, f'prompt suspicious for {pid}'
    for seed in SEEDS:
        f = OUT / f'{vid}_seed{seed}.mp4'
        total += 1
        if f.exists() and f.stat().st_size > 10000:
            print('SKIP', f.name, flush=True); continue
        t0 = time.time()
        try:
            g = torch.Generator(device='cpu').manual_seed(seed)
            frames = pipe(prompt=prompt, width=832, height=480, num_frames=81,
                          num_inference_steps=40, guidance_scale=3.0,
                          generator=g).frames[0]
            export_to_video(frames, str(f), fps=16)
            print(f'DONE {f.name} {f.stat().st_size}B {time.time()-t0:.0f}s', flush=True)
        except Exception as e:
            fail += 1
            print(f'FAIL {f.name}: {e!r}', flush=True)
        gc.collect(); torch.cuda.empty_cache()
print('ALL_DONE' if fail == 0 else f'FAILURES={fail}', flush=True)
sys.exit(2 if not FULL and fail else 0)
