#!/usr/bin/env bash
set -u
P=/gluster_hdd_cv/user/jinzili/tmp/pilot
cd $P
exec >> $P/judge_gtlast.log 2>&1
PY=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python
echo "=== extract last-second frames $(date -u) ==="
$PY extract8_last.py $P/clips_gt $P/frames_gtlast
echo "GTLAST=$(ls frames_gtlast | wc -l)"
export VARIANT=orig FRAMESET=frames_gtlast
for J in q2 q25 q3vl; do
  export JUDGE=$J
  case $J in q3vl) export CUDA_VISIBLE_DEVICES=0,1,2,3 ;; *) export CUDA_VISIBLE_DEVICES=0 ;; esac
  echo "=== frames_gtlast x $J $(date -u) ==="
  $PY judge_frameset.py
done
echo JUDGE_GTLAST_DONE
