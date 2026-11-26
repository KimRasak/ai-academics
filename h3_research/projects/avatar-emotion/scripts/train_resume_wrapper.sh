#!/bin/bash
# Auto-resume training wrapper: restart on crash, resuming LoRA from newest checkpoint.
set -u
ENV_PY=/ssd2/jinzili/envs/echomimic/bin/python
export PATH=/ssd2/jinzili/ffmpeg-7.0.2-amd64-static:$PATH
cd /ssd2/jinzili/echomimic

STEPS=${STEPS:-8000}
OUT=${OUT:-/ssd2/jinzili/avdata/lora_vA2}
GPU=${GPU:-5}
LR=${LR:-5e-5}
R=${R:-32}
ALPHA=${ALPHA:-32}
MAXTRIES=${MAXTRIES:-40}

attempt=0
while [ $attempt -lt $MAXTRIES ]; do
  attempt=$((attempt+1))
  newest=$(ls -d "$OUT"/step* 2>/dev/null | sort -t p -k3 -n | tail -1)
  INIT=""
  if [ -n "$newest" ] && [ -f "$newest/denoising_lora.pt" ]; then
    INIT="--init_lora $newest"
    echo "[wrapper] attempt $attempt resuming from $newest $(date +%H:%M:%S)"
  else
    echo "[wrapper] attempt $attempt fresh start $(date +%H:%M:%S)"
  fi
  CUDA_VISIBLE_DEVICES=$GPU $ENV_PY /ssd2/jinzili/avdata/train_emolora.py \
    --steps $STEPS --save_every 500 --out "$OUT" --variant VA \
    --r $R --alpha $ALPHA --lr $LR $INIT
  code=$?
  last=$(ls -d "$OUT"/step* 2>/dev/null | sort -t p -k3 -n | tail -1)
  if [ "$(basename "$last" 2>/dev/null)" = "step$STEPS" ]; then
    echo "[wrapper] reached step$STEPS — done after $attempt attempts"
    break
  fi
  echo "[wrapper] exited code=$code at $(basename "$last" 2>/dev/null) — retrying"
  sleep 10
done
echo WRAPPER_DONE
