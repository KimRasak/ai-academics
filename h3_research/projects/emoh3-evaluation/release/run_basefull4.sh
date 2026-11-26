#!/usr/bin/env bash
# Round-7 decisive control (4-GPU, resumable): BASE model on all 8 tasks x 2 seeds, 16 steps,
# inference settings identical to run_lora3_eval.sh.
#
# Two hard-won constraints shape this script:
#  * The node is shared and this container's nvidia-smi does NOT show foreign tenants' memory, so a
#    "free" GPU can still OOM. We therefore TRY several GPU quadruples per case and keep the one that
#    actually produces an mp4, instead of trusting a memory query.
#  * Two concurrent runs would rendezvous into the same NCCL FileStore and hang, so a single-instance
#    lock guards the whole pass. Liveness is judged by lock mtime (the launcher uses another pid ns).
set -u
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
exec >> /gluster_hdd_cv/user/jinzili/tmp/pilot/basefull.log 2>&1

LOCK=/gluster_hdd_cv/user/jinzili/tmp/pilot/.basefull4.lock
# The lock records its owner. Without that, killing one instance would fire its EXIT trap and delete a
# lock now held by a *different* live instance -- which is exactly how two runs ended up interleaved.
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -z "$(find "$LOCK" -maxdepth 0 -mmin +9 2>/dev/null)" ]; then
    echo "LOCKED: another run touched the lock within 9 min"; exit 0
  fi
  echo "stale lock (>9 min), taking over"
fi
echo $$ > "$LOCK/owner"
touch "$LOCK"
mine() { [ "$(cat "$LOCK/owner" 2>/dev/null)" = "$$" ]; }
trap 'mine && rm -rf "$LOCK"' EXIT
hold() { mine || { echo "LOST LOCK to $(cat "$LOCK/owner" 2>/dev/null), exiting"; exit 0; }; touch "$LOCK"; }
echo "START4 $(date -u) pid=$$"

export NCCL_DEBUG=WARN PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/gluster_hdd_cv/user/jinzili/env/h3ft/bin/python
IDX=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python
# Prefer the four emptiest GPUs as reported right now (the view is imperfect but usually right when a
# foreign tenant is large), then fall back to fixed quads.
best4() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | sort -t, -k2 -n | head -4 | cut -d, -f1 | paste -sd, -; }
# Every fixed quad below is tried in turn; the list deliberately covers subsets that avoid each pair of
# GPUs, because a foreign tenant can sit on any two cards and then only some quads can fit.
QUADS="$(best4) 2,3,4,5 0,2,3,4 0,3,4,5 3,4,5,7 0,2,4,5 2,4,5,7 4,5,6,7 0,1,2,3 2,3,6,7 0,1,4,5 1,2,3,7 0,1,6,7"

for SEED in 42 1234; do
  OUT=/gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_basefull_s$SEED
  mkdir -p "$OUT"
  for I in 1 2 3 4 5 6 7 8; do
    CASE=$($IDX -c "import json; j=json.load(open('jobs_lora3.json')); print([x['case_id'] for x in j if x['index']==$I][0])" 2>/dev/null)
    [ -n "$CASE" ] || continue
    if ls "$OUT"/${CASE}*.mp4 >/dev/null 2>&1; then echo "HAVE s$SEED $CASE"; continue; fi
    $IDX -c "import json; j=json.load(open('jobs_lora3.json')); json.dump([x for x in j if x['index']==$I], open('jobs_one_$I.json','w'))"
    ARGS="--jobs jobs_one_$I.json --outdir $OUT --width 768 --height 768 --num-frames 124 --steps 16 --seed $SEED"
    for Q in $QUADS; do
      hold
      rm -f bf4_store_${SEED}_$I
      export H3_STORE_FILE=/gluster_hdd_cv/user/jinzili/tmp/pilot/bf4_store_${SEED}_$I
      export CUDA_VISIBLE_DEVICES=$Q
      echo "RUN4 s$SEED $CASE gpus=$Q $(date -u)"
      for r in 0 1 2 3; do
        RANK=$r LOCAL_RANK=$r WORLD_SIZE=4 $PY infer_lora_ref2va.py $ARGS > bf4_s${SEED}_i${I}_r$r.log 2>&1 &
      done
      wait
      if ls "$OUT"/${CASE}*.mp4 >/dev/null 2>&1; then
        echo "OK4 s$SEED $CASE gpus=$Q"; break
      fi
      if grep -qils "OutOfMemoryError" bf4_s${SEED}_i${I}_r*.log; then
        echo "OOM4 s$SEED $CASE gpus=$Q -> next quad"
      else
        echo "FAIL4 s$SEED $CASE gpus=$Q (no mp4, no OOM) -> next quad"
      fi
    done
  done
done
echo "ALL_BASEFULL4_DONE s42=$(ls outputs_basefull_s42 | grep -c mp4) s1234=$(ls outputs_basefull_s1234 | grep -c mp4)"
