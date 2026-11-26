#!/usr/bin/env bash
set -u
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
echo "=== SEED 42 ==="
bash run_lora3_eval.sh
echo "=== SEED 1234 ==="
bash run_lora3_eval_s1234.sh
echo "ALL_EVAL_DONE"
