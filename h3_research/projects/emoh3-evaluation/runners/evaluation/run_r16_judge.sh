#!/usr/bin/env bash
# Round 16 post-processing: extract 8 frames + judge (8-frame, greedy) for all new outputs.
set -u
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
PY=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python

proc() { # $1=src $2=framesdst $3=outjsonl
  echo "[$(date +%H:%M)] EXTRACT $1"
  $PY extract8_frames.py "$1" "$2" > r16_extract_$(basename $1).log 2>&1
  echo "[$(date +%H:%M)] JUDGE $2"
  H3_JUDGE_FRAMES="$2" H3_JUDGE_OUT="$3" $PY judge8.py > r16_judge_$(basename $1).log 2>&1
  echo "[$(date +%H:%M)] DONE $(basename $1)"
}

proc /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_x02sweep_base /gluster_hdd_cv/user/jinzili/tmp/pilot/frames_x02sweep_base /gluster_hdd_cv/user/jinzili/tmp/pilot/judge_x02sweep_base.jsonl
proc /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_x02sweep_lora /gluster_hdd_cv/user/jinzili/tmp/pilot/frames_x02sweep_lora /gluster_hdd_cv/user/jinzili/tmp/pilot/judge_x02sweep_lora.jsonl
proc /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_refswap_base_s42 /gluster_hdd_cv/user/jinzili/tmp/pilot/frames_refswap_base_s42 /gluster_hdd_cv/user/jinzili/tmp/pilot/judge_refswap_base_s42.jsonl
proc /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_refswap_lora_s42 /gluster_hdd_cv/user/jinzili/tmp/pilot/frames_refswap_lora_s42 /gluster_hdd_cv/user/jinzili/tmp/pilot/judge_refswap_lora_s42.jsonl
proc /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_refswap_base_s1234 /gluster_hdd_cv/user/jinzili/tmp/pilot/frames_refswap_base_s1234 /gluster_hdd_cv/user/jinzili/tmp/pilot/judge_refswap_base_s1234.jsonl
proc /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_refswap_lora_s1234 /gluster_hdd_cv/user/jinzili/tmp/pilot/frames_refswap_lora_s1234 /gluster_hdd_cv/user/jinzili/tmp/pilot/judge_refswap_lora_s1234.jsonl

echo "[$(date +%H:%M)] ALL_POST_DONE"
