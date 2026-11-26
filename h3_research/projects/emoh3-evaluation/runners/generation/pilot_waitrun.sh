#!/bin/bash
# Wait for H3 Ref2VA server on 034 to be ready, then run the pilot generation.
BASE="http://osa00-h100-034.osa00.bdjp.io:30020"
PY=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python
for i in $(seq 1 180); do
  out=$(curl -s -m 6 --noproxy="*" "$BASE/get_model_info" 2>/dev/null)
  if [ -n "$out" ]; then
    echo "[wait] ready after ~$((i*10))s"
    echo "$out" | head -c 200
    echo
    break
  fi
  sleep 10
  if [ $((i % 6)) -eq 0 ]; then echo "[wait] still loading... $(date +%H:%M:%S)"; fi
done
echo "[run] starting pilot at $(date +%H:%M:%S)"
cd /gluster_hdd_cv/user/jinzili/tmp/pilot || exit 1
H3_BASE="$BASE" "$PY" pilot_expr.py
echo "[done] pilot finished at $(date +%H:%M:%S)"
