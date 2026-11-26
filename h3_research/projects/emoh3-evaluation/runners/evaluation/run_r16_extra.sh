#!/usr/bin/env bash
# Round 16 extra evidence: (M1) x02 multi-seed sweep for base+v3; (M4) reference-swap mechanism test.
# Sequential 8-proc FSDP2 runs. ~1.5-2.5h total on 8xH100.
set -u
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
export H3_STORE_FILE=/gluster_hdd_cv/user/jinzili/tmp/pilot/r16_infer_store
export PYTHONFAULTHANDLER=1
PY=/gluster_hdd_cv/user/jinzili/env/h3ft/bin/python
CKPT=/gluster_hdd_cv/user/jinzili/tmp/pilot/runs/lora3/checkpoint-900.pt

run() {
  local name=$1 lora=$2 jobs=$3 outdir=$4 seed=$5
  echo "[$(date +%H:%M)] START $name (seed $seed, jobs $jobs)"
  rm -f r16_infer_store
  local ARGS="--jobs $jobs --outdir $outdir $lora --lora-alpha 32 --width 768 --height 768 --num-frames 124 --steps 16 --seed $seed"
  for i in 0 1 2 3 4 5 6 7; do
    RANK=$i LOCAL_RANK=$i WORLD_SIZE=8 $PY infer_lora_ref2va.py $ARGS > r16_${name}_rank$i.log 2>&1 &
  done
  wait
  echo "[$(date +%H:%M)] DONE $name exit=$?"
}

# ---- M1: x02 seed sweep (5 new seeds each) ----
run x02sweep_base "" jobs_x02sweep.json /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_x02sweep_base 777
run x02sweep_lora "--lora $CKPT" jobs_x02sweep.json /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_x02sweep_lora 777

# ---- M4: reference swap, prompt fixed "She looks sad.", 4 emotion refs of char 1534125952 ----
run refswap_base_s42 "" jobs_refswap.json /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_refswap_base_s42 42
run refswap_lora_s42 "--lora $CKPT" jobs_refswap.json /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_refswap_lora_s42 42
run refswap_base_s1234 "" jobs_refswap.json /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_refswap_base_s1234 1234
run refswap_lora_s1234 "--lora $CKPT" jobs_refswap.json /gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_refswap_lora_s1234 1234

echo "[$(date +%H:%M)] ALL_EXTRA_DONE"
