#!/usr/bin/env bash
# Extract 8 frames + judge for both v3 eval runs (container GPU 7).
set -u
export PATH=/gluster_hdd_cv/user/jinzili/env/sglang310/bin:$PATH
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
P=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python
rm -rf frames_lora3 frames_lora3s judge_lora3_results.jsonl judge_lora3s_results.jsonl
$P extract8_frames.py outputs_lora3 frames_lora3
$P extract8_frames.py outputs_lora3_s1234 frames_lora3s
H3_JUDGE_FRAMES=/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_lora3 \
H3_JUDGE_OUT=/gluster_hdd_cv/user/jinzili/tmp/pilot/judge_lora3_results.jsonl \
CUDA_VISIBLE_DEVICES=7 $P judge8.py
H3_JUDGE_FRAMES=/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_lora3s \
H3_JUDGE_OUT=/gluster_hdd_cv/user/jinzili/tmp/pilot/judge_lora3s_results.jsonl \
CUDA_VISIBLE_DEVICES=7 $P judge8.py
echo "ALL_JUDGE_DONE"
