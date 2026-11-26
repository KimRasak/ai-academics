import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
print("import ok", torch.__version__, flush=True)
m = Qwen2VLForConditionalGeneration.from_pretrained(
    "/gluster_hdd_cv/user/jinzili/weights/reward_models/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
p = AutoProcessor.from_pretrained("/gluster_hdd_cv/user/jinzili/weights/reward_models/Qwen2-VL-7B-Instruct", trust_remote_code=True)
print("load ok", m.dtype, flush=True)
