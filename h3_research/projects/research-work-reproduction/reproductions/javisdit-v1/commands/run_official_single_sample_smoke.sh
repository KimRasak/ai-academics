#!/usr/bin/env bash
set -euo pipefail

# Released-checkpoint smoke only. This is deliberately separate from the
# JavisBench-mini protocol and never computes an evaluation metric.

code_root=/gluster_hdd_cv/user/jinzili/code/research-work-reproduction/javisdit-v1/JavisDiT
env_python=/gluster_hdd_cv/user/jinzili/env/javisdit-v1-py310-cu121/bin/python
data_root=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/javisdit-v1
weights_root=/gluster_hdd_cv/user/jinzili/weights/javisdit-v1-official
run_id=javisdit_v1_official_smoke_gpu4_20260920T125200Z
run_root="$data_root/outputs/$run_id"
run_work="$data_root/runtime/$run_id"
log_file="$data_root/logs/${run_id}.log"

if test -e "$run_root" || test -e "$run_work"; then
  echo "refusing to reuse smoke output/runtime directory" >&2
  exit 2
fi

mkdir -p "$run_root" "$run_work/checkpoints" "$data_root/logs" "$data_root/manifests"
ln -s "$weights_root/Wan2.1-T2V-1.3B" "$run_work/checkpoints/Wan2.1-T2V-1.3B"
ln -s "$weights_root/audioldm2" "$run_work/checkpoints/audioldm2"

gpu4_memory_before=$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits)
if test "$gpu4_memory_before" -gt 128; then
  echo "GPU 4 preflight has ${gpu4_memory_before} MiB in use; refusing launch" >&2
  exit 3
fi

prompt='Realistic locked-off shot. Ocean waves roll onto a quiet rocky shore at dawn, foam spreading over dark stones before sliding back into the sea. The camera holds still as pale morning light grows over the horizon. The soundtrack is layered with surf, gulls calling far away, and a steady coastal wind.'

cd "$run_work"
CUDA_VISIBLE_DEVICES=4 "$env_python" "$code_root/scripts/inference.py" \
  "$code_root/configs/javisdit-v1-0/inference/sample.py" \
  --model-path "$weights_root/JavisDiT-v1.0-jav" \
  --prompt "$prompt" \
  --save-dir "$run_root" \
  --verbose 2 \
  2>&1 | tee "$log_file"

find "$run_root" -type f \( -name '*.mp4' -o -name '*.wav' \) -print0 | sort -z | xargs -0 sha256sum \
  > "$data_root/manifests/${run_id}.media.sha256"
find "$run_root" -type f \( -name '*.mp4' -o -name '*.wav' \) -print0 | sort -z | xargs -0 -r -n1 ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \
  > "$data_root/manifests/${run_id}.ffprobe.txt"
