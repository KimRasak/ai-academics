#!/usr/bin/env bash
set -euo pipefail

# Read-only ownership gate for a local four-GPU MiniMax-H3 Base service.
# It neither starts a server nor changes GPU/process state. Pass exactly four
# physical GPU indices, for example: check_h3_launch_preflight.sh 0 1 2 4
if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU0 GPU1 GPU2 GPU3" >&2
  exit 64
fi

gpus=("$@")
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU index: $gpu" >&2; exit 64; }
done

if [[ $(printf '%s\n' "${gpus[@]}" | sort -n | uniq | wc -l) -ne 4 ]]; then
  echo "GPU indices must be distinct" >&2
  exit 64
fi

echo "checked_at_utc=$(date -u +%FT%TZ)"
echo "requested_gpus=${gpus[*]}"

ports_busy=0
for port in 30010 30011; do
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
    echo "port_${port}=LISTENING"
    ports_busy=1
  else
    echo "port_${port}=free"
  fi
done

busy=0
for gpu in "${gpus[@]}"; do
  # Compute-process table is authoritative for ownership; memory alone may
  # include driver allocations. Print every process record for auditing.
  rows=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || true)
  if [[ -n "$rows" ]]; then
    printf 'gpu_%s=BUSY\n%s\n' "$gpu" "$rows"
    busy=1
  else
    printf 'gpu_%s=free_of_compute_processes\n' "$gpu"
  fi
done

if [[ "$ports_busy" -ne 0 || "$busy" -ne 0 ]]; then
  echo "decision=DO_NOT_START_H3_SERVICE"
  exit 2
fi

echo "decision=SAFE_TO_START_H3_SERVICE_SUBJECT_TO_HUMAN_OWNERSHIP_CONFIRMATION"
