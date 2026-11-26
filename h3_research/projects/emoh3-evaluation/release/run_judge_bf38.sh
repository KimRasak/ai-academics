#!/usr/bin/env bash
# Qwen3.8-27B on the 16-cell paired base matrix only (0-1 s protocol).
# Bounded pass: this judge runs at minutes per clip in plain transformers because it emits a long
# deliberation before its WORD:SCORE line, so the 97-config full-span pass is deliberately not attempted.
set -u
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
exec >> /gluster_hdd_cv/user/jinzili/tmp/pilot/judge_bf38.log 2>&1
PY=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python
export VARIANT=orig JUDGE=q38 FRAMESET=frames_bf16 CUDA_VISIBLE_DEVICES=0,1,2,3
echo "=== frames_bf16 x q38 $(date -u) ==="
$PY judge_frameset.py
echo JUDGE_BF38_DONE
