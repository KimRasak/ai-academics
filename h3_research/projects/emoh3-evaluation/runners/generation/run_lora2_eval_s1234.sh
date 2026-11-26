#!/usr/bin/env bash
# EmoH3 LoRA2 variance pass: all 8 cross-emotion cases, seed 1234.
set -u
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
rm -f lora2s_store
export H3_STORE_FILE=/gluster_hdd_cv/user/jinzili/tmp/pilot/lora2s_store
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
PY=/gluster_hdd_cv/user/jinzili/env/h3ft/bin/python
CKPT=/gluster_hdd_cv/user/jinzili/tmp/pilot/runs/lora2/checkpoint-600.pt
ARGS="--jobs jobs_ref2va_emo.json --outdir /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_lora2_s1234 --lora $CKPT --lora-alpha 32 --width 768 --height 768 --num-frames 124 --steps 8 --seed 1234"
for i in 0 1 2 3 4 5 6 7; do
  RANK=$i LOCAL_RANK=$i WORLD_SIZE=8 $PY infer_lora_ref2va.py $ARGS > lora2s_infer_rank$i.log 2>&1 &
done
wait
echo "ALL_WORKERS_DONE exit=$?"
