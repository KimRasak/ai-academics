#!/usr/bin/env bash
# EmoH3 cache encode for 18-sample expanded training data (single GPU).
set -u
export CUDA_VISIBLE_DEVICES=0
export PATH=/gluster_hdd_cv/user/jinzili/env/sglang310/bin:/usr/bin:$PATH
cd /gluster_hdd_cv/user/jinzili/tmp/h3ft
rm -rf /gluster_hdd_cv/user/jinzili/tmp/pilot/traindata2/cache
/gluster_hdd_cv/user/jinzili/env/h3ft/bin/python prepare_cache.py \
  --model /gluster_hdd_cv/user/jinzili/weights/MiniMax-H3 \
  --metadata /gluster_hdd_cv/user/jinzili/tmp/pilot/traindata2/train.jsonl \
  --output /gluster_hdd_cv/user/jinzili/tmp/pilot/traindata2/cache \
  --encode-text --encode-audio \
  > /gluster_hdd_cv/user/jinzili/tmp/pilot/h3ft_cache2.log 2>&1
echo "CACHE_DONE exit=$?"
ls /gluster_hdd_cv/user/jinzili/tmp/pilot/traindata2/cache/ 2>/dev/null | wc -l
