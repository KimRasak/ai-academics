#!/usr/bin/env bash
# EmoH3 inference-step scaling test: x05+x07 at 16 steps, seed 42, LoRA v2.
set -u
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
rm -f lora3s_store
export H3_STORE_FILE=/gluster_hdd_cv/user/jinzili/tmp/pilot/lora3s_store
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
PY=/gluster_hdd_cv/user/jinzili/env/h3ft/bin/python
CKPT=/gluster_hdd_cv/user/jinzili/tmp/pilot/runs/lora3/checkpoint-900.pt
ARGS="--jobs jobs_lora3.json --outdir /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_lora3_s1234 --lora $CKPT --lora-alpha 32 --width 768 --height 768 --num-frames 124 --steps 16 --seed 1234"
for i in 0 1 2 3 4 5 6 7; do
  RANK=$i LOCAL_RANK=$i WORLD_SIZE=8 $PY infer_lora_ref2va.py $ARGS > lora3s_infer_rank$i.log 2>&1 &
done
wait
echo "ALL_WORKERS_DONE exit=$?"
