#!/usr/bin/env bash
set -euo pipefail

# Start one official local H3-Base service only after its four GPUs pass the
# read-only ownership gate. This command is intentionally never called by any
# preflight or case runner: invoking it consumes four GPUs.
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

ROOT=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/minimax-h3-official-768p
MODEL=/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3
SGLANG=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/sglang
GATE="$ROOT/tools/check_h3_launch_preflight.sh"
LOG_DIR="$ROOT/server_logs/${variant}_$(date -u +%Y%m%dT%H%M%SZ)"

[[ -x "$GATE" ]] || { echo "missing ownership gate: $GATE" >&2; exit 65; }
[[ -x "$SGLANG" ]] || { echo "missing sglang executable: $SGLANG" >&2; exit 65; }
[[ -d "$MODEL/$([[ "$variant" == fl2va ]] && echo FL2VA || echo Ref2VA)" ]] || {
  echo "missing checkpoint partition for $variant" >&2; exit 65;
}

"$GATE" "${gpus[@]}"
mkdir -p "$LOG_DIR"
{
  printf 'started_at_utc='; date -u +%FT%TZ
  printf 'variant=%s\nport=%s\ngpus=%s\nmodel=%s\n' "$variant" "$port" "${gpus[*]}" "$MODEL"
  sha256sum "$ROOT/R0_FREEZE_MANIFEST.json" "$ROOT/tools/check_h3_launch_preflight.sh"
  for gpu in "${gpus[@]}"; do
    nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits -i "$gpu"
  done
} > "$LOG_DIR/launch_metadata.txt"

export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gpus[*]}")
# The official four-GPU FL2VA load reached the transformer shard stage but
# failed with 3.35 GiB PyTorch-reserved/unallocated memory and a 294 MiB
# allocation request.  This allocator option is a runtime fragmentation
# compatibility setting; it does not alter model weights, request payloads,
# sampling arguments, routing, or output protocol.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:+${PYTORCH_CUDA_ALLOC_CONF},}expandable_segments:True"
printf 'service_log=%s\n' "$LOG_DIR/service.log"
exec "$SGLANG" serve \
  --model-path "$MODEL" \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port "$port" \
  --model-variant "$variant" \
  > "$LOG_DIR/service.log" 2>&1
