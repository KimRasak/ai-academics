#!/bin/bash
# Follow-up: judge the 3 appendix dirs missing from the main re-judge (v2 s1234 4fr; v2 16s s42/s1234).
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
PY=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python
rm -rf frames_extra
mkdir -p frames_extra
for d in frames_lora2_s1234 frames_s16 frames_s16b; do
  [ -d "$d" ] || { echo "SKIP missing $d"; continue; }
  for c in "$d"/*; do
    [ -d "$c" ] || continue
    ln -s "../$d/$(basename "$c")" "frames_extra/${d}__$(basename "$c")"
  done
done
export H3_JUDGE_FRAMES=/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_extra
echo "extra cases: $(ls frames_extra | wc -l)"
export H3_JUDGE_MAXFR=8
export H3_JUDGE_OUT=/gluster_hdd_cv/user/jinzili/tmp/pilot/judge15_extra_8fr.jsonl
rm -f judge15_extra_8fr.jsonl
$PY judge15.py > judge15_extra_8fr.log 2>&1
echo "extra 8fr done: $(wc -l < judge15_extra_8fr.jsonl)"
export H3_JUDGE_MAXFR=4
export H3_JUDGE_OUT=/gluster_hdd_cv/user/jinzili/tmp/pilot/judge15_extra_4fr.jsonl
rm -f judge15_extra_4fr.jsonl
$PY judge15.py > judge15_extra_4fr.log 2>&1
echo "extra 4fr done: $(wc -l < judge15_extra_4fr.jsonl)"
echo EXTRA_DONE
