#!/usr/bin/env bash
set -euo pipefail

# Read-only turnover audit after an H3 server operator has stopped a service.
# It never sends a signal and never frees a port/GPU by force.
if [[ $# -ne 5 ]]; then
  echo "usage: $0 fl2va|ref2va GPU0 GPU1 GPU2 GPU3" >&2
  exit 64
fi

variant=$1
shift
gpus=("$@")
case "$variant" in
  fl2va) port=30010 ;;
  ref2va) port=30011 ;;
  *) echo "unknown model variant: $variant" >&2; exit 64 ;;
esac

printf 'checked_at_utc='; date -u +%FT%TZ
printf 'variant=%s\nport=%s\nrequested_gpus=%s\n' "$variant" "$port" "${gpus[*]}"

listening=0
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
  echo "port_${port}=LISTENING"
  listening=1
else
  echo "port_${port}=free"
fi

busy=0
for gpu in "${gpus[@]}"; do
  rows=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || true)
  if [[ -n "$rows" ]]; then
    printf 'gpu_%s=BUSY\n%s\n' "$gpu" "$rows"
    busy=1
  else
    printf 'gpu_%s=free_of_compute_processes\n' "$gpu"
  fi
done

if [[ "$listening" -eq 0 && "$busy" -eq 0 ]]; then
  echo "decision=SERVICE_TERMINAL_AND_SELECTED_GPUS_FREE_OF_COMPUTE_PROCESSES"
  exit 0
fi
echo "decision=DO_NOT_START_NEXT_SERVICE_OR_RELEASE_GPU_CLAIM"
exit 2
