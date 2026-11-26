#!/usr/bin/env python3
import os,json,sys
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration,AutoProcessor
model_path='/gluster_hdd_cv/user/jinzili/weights/reward_models/Qwen2-VL-7B-Instruct'
video=sys.argv[1]; out=Path(sys.argv[2]); frames=sorted(Path(video).parent.glob('*.jpg'))
if not frames: raise SystemExit('need extracted jpg frames beside video')
frames=frames[:4]
model=Qwen2VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map='cuda',trust_remote_code=True)
proc=AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
content=[{'type':'image'} for _ in frames]+[{'type':'text','text':'Assess this generated video frame sequence. Answer exactly one label: coherent, partially_coherent, or incoherent. Then colon and confidence 0-100. Format LABEL:SCORE'}]
msg=[{'role':'user','content':content}]; text=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True); inputs=proc(text=[text],images=[Image.open(f).convert('RGB') for f in frames],return_tensors='pt');inputs={k:v.to('cuda') if hasattr(v,'to') else v for k,v in inputs.items()};gen=model.generate(**inputs,max_new_tokens=32,do_sample=False);verdict=proc.batch_decode(gen[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip();json.dump({'model':model_path,'video':video,'n_frames':len(frames),'verdict':verdict},open(out,'w'),indent=2);print(verdict)
