#!/usr/bin/env bash
# Complete the full-span matrix for the three fast judges first.
#
# Rationale: Qwen3.8-27B runs at minutes per clip in plain transformers (it emits a long
# deliberation before the WORD:SCORE line), so putting it ahead of the fast judges starves the
# analyses that only need q2/q25/q3vl. Those three finish the 97-config full-span matrix in well
# under an hour and already give three of the six inter-judge reliability pairs.
set -u
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
exec >> /gluster_hdd_cv/user/jinzili/tmp/pilot/judge_full3.log 2>&1
PY=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python
export VARIANT=orig
for J in q2 q25 q3vl; do
  export JUDGE=$J FRAMESET=frames_full
  case $J in
    q3vl) export CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
    *)    export CUDA_VISIBLE_DEVICES=0 ;;
  esac
  echo "=== frames_full x $J on GPUs $CUDA_VISIBLE_DEVICES $(date -u) ==="
  $PY judge_frameset.py
done
echo JUDGE_FULL3_DONE
