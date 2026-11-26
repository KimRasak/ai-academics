#!/bin/bash
# 15-class re-judge: pass1 = all cases @ 8fr, pass2 = all cases @ 4fr, pass3 = judge sanity on references.
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
PY=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python
export H3_JUDGE_FRAMES=/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_all

echo "PASS1 8fr start $(date)"
export H3_JUDGE_MAXFR=8
export H3_JUDGE_OUT=/gluster_hdd_cv/user/jinzili/tmp/pilot/judge15_8fr.jsonl
rm -f judge15_8fr.jsonl
$PY judge15.py > judge15_8fr.log 2>&1
echo "PASS1 8fr done: $(wc -l < judge15_8fr.jsonl)"

echo "PASS2 4fr start $(date)"
export H3_JUDGE_MAXFR=4
export H3_JUDGE_OUT=/gluster_hdd_cv/user/jinzili/tmp/pilot/judge15_4fr.jsonl
rm -f judge15_4fr.jsonl
$PY judge15.py > judge15_4fr.log 2>&1
echo "PASS2 4fr done: $(wc -l < judge15_4fr.jsonl)"

echo "PASS3 sanity start $(date)"
export H3_SANITY_OUT=/gluster_hdd_cv/user/jinzili/tmp/pilot/judge_sanity15.jsonl
rm -f judge_sanity15.jsonl
$PY judge_reference_sanity.py > sanity15.log 2>&1
echo "PASS3 sanity done: $(wc -l < judge_sanity15.jsonl)"
echo ALLDONE $(date)
