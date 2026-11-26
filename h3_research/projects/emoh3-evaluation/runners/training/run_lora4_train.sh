#!/usr/bin/env bash
# EmoH3 LoRA retrain on expanded 18-sample clean preservation data (8xH100).
set -u
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
cd /gluster_hdd_cv/user/jinzili/tmp/h3ft
OUT=/gluster_hdd_cv/user/jinzili/tmp/pilot/runs/lora4
CACHE=/gluster_hdd_cv/user/jinzili/tmp/pilot/traindata4/cache
rm -f "$OUT/dist_store" "$CACHE/dist_store" 2>/dev/null
export H3_STORE_FILE="$OUT/dist_store"
/gluster_hdd_cv/user/jinzili/env/h3ft/bin/deepspeed --num_gpus 8 train.py \
  --model /gluster_hdd_cv/user/jinzili/weights/MiniMax-H3 \
  --variant ref2va \
  --cache "$CACHE" \
  --output "$OUT" \
  --max-steps 900 --trainable lora --strategy deepspeed \
  --video-shift 12.0 --audio-shift 3.0 \
  --val-every 200 --ckpt-every 100 \
  > /gluster_hdd_cv/user/jinzili/tmp/pilot/lora4_train.log 2>&1
echo "TRAIN_DONE exit=$?"
