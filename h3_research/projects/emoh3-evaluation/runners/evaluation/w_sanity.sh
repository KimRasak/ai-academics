#!/bin/bash
# Judge sanity: classify 21 known-label reference portraits with the 15-class judge.
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
PY=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python
export H3_SANITY_OUT=/gluster_hdd_cv/user/jinzili/tmp/pilot/judge_sanity15.jsonl
rm -f judge_sanity15.jsonl sanity15.log
$PY judge_reference_sanity.py > sanity15.log 2>&1
echo "sanity done: $(wc -l < judge_sanity15.jsonl)"
echo SANITY_DONE
