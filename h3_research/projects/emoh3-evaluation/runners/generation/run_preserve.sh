#!/usr/bin/env bash
# EmoH3 preservation-video generation (BASE model, no LoRA) — 10 emotion targets.
set -u
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
rm -f pres_infer_store
export H3_STORE_FILE=/gluster_hdd_cv/user/jinzili/tmp/pilot/pres_infer_store
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
PY=/gluster_hdd_cv/user/jinzili/env/h3ft/bin/python
ARGS="--jobs jobs_preserve.json --outdir /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_preserve --width 768 --height 768 --num-frames 124 --steps 8"
for i in 0 1 2 3 4 5 6 7; do
  RANK=$i LOCAL_RANK=$i WORLD_SIZE=8 $PY infer_lora_ref2va.py $ARGS > pres_infer_rank$i.log 2>&1 &
done
wait
echo "ALL_WORKERS_DONE exit=$?"
