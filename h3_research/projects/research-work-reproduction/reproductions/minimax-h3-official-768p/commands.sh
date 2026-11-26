#!/usr/bin/env bash
set -euo pipefail

# Documentation-only official argument record. Use
# commands/launch_h3_base_service.sh for a logged, ownership-gated launch.
MODEL=/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3
SGLANG=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/sglang

# FL2VA service for both T2VA and FL2VA requests (official endpoint 30010).
# The launch wrapper sets an explicitly audited CUDA_VISIBLE_DEVICES list and
# archives the server log; do not run both services concurrently on one GPU set.
"$SGLANG" serve \
  --model-path "$MODEL" \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30010 \
  --model-variant fl2va

# Ref2VA is a separate service and must not be started concurrently on the
# same four GPUs (official endpoint 30011):
"$SGLANG" serve \
  --model-path "$MODEL" \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30011 \
  --model-variant ref2va
