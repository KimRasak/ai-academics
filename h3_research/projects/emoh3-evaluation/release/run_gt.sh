#!/usr/bin/env bash
# ACCURACY test with external labels (no annotation work by us).
#
# The 22 training rows point at 14 distinct source clips from the same production batch as the
# known-emotion portraits the paper already uses as ground truth (expressions_0825). Their emotion is
# in the filename, so these clips let us ask the question reliability cannot answer: does judging the
# whole clip recover the *labelled* emotion more often than judging its first second?
set -u
P=/gluster_hdd_cv/user/jinzili/tmp/pilot
cd $P
exec >> $P/judge_gt.log 2>&1
PY=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python
PYFT=/gluster_hdd_cv/user/jinzili/env/h3ft/bin/python

echo "=== stage clips $(date -u) ==="
mkdir -p clips_gt
$PYFT - <<'PYX'
import json, shutil, os
rows=[json.loads(l) for l in open("/gluster_hdd_cv/user/jinzili/tmp/pilot/traindata3/train.jsonl")]
seen=set()
for r in rows:
    p=r["target_video"]
    if p in seen or not os.path.exists(p): continue
    seen.add(p)
    dst="/gluster_hdd_cv/user/jinzili/tmp/pilot/clips_gt/"+os.path.basename(p)
    if not os.path.exists(dst): shutil.copy(p, dst)
print("staged", len(seen))
PYX
echo "CLIPS=$(ls clips_gt | grep -c mp4)"

echo "=== extract both protocols ==="
$PY extract8_frames.py $P/clips_gt $P/frames_gt1s
$PY extract8_full.py   $P/clips_gt $P/frames_gtfull
echo "GT1S=$(ls frames_gt1s | wc -l) GTFULL=$(ls frames_gtfull | wc -l)"

export VARIANT=orig
for FS in frames_gt1s frames_gtfull; do
  export FRAMESET=$FS
  for J in q2 q25 q3vl; do
    export JUDGE=$J
    case $J in
      q3vl) export CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
      *)    export CUDA_VISIBLE_DEVICES=0 ;;
    esac
    echo "=== $FS x $J $(date -u) ==="
    $PY judge_frameset.py
  done
done
echo JUDGE_GT_DONE
