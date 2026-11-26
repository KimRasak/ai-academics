#!/usr/bin/env bash
set -euo pipefail

# Execute exactly one frozen official H3 768p request after the service has
# passed the ownership gate. This wrapper does not start/stop SGLang; that
# remains an explicit operator action because it consumes four GPUs.
if [[ $# -ne 1 ]]; then
  echo "usage: H3_BASE_URL=http://host:port $0 t2va|fl2va|ref2va" >&2
  exit 64
fi

case_name=$1
case "$case_name" in
  t2va|fl2va) default_port=30010 ;;
  ref2va) default_port=30011 ;;
  *) echo "unknown frozen case: $case_name" >&2; exit 64 ;;
esac

ROOT=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/minimax-h3-official-768p
MODEL=/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3
RUNNER="$ROOT/tools/run_frozen_official_request.py"
SCRIPT="$MODEL/scripts/readme/reproducible-768p-${case_name}-request.sh"
OUT="$ROOT/outputs/${case_name}"
# H3_BASE_URL makes a pre-existing, separately launcher-provisioned service
# explicit.  It never starts, stops or reconfigures the service.  The default
# preserves the historical local routing for a local service.
BASE_URL="${H3_BASE_URL:-http://127.0.0.1:${default_port}}"

[[ -x "$RUNNER" || -f "$RUNNER" ]] || { echo "missing durable runner: $RUNNER" >&2; exit 65; }
[[ -f "$SCRIPT" ]] || { echo "missing official script: $SCRIPT" >&2; exit 65; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite existing output: $OUT" >&2; exit 73; }

curl --fail --silent --show-error "$BASE_URL/health" >/dev/null
mkdir -p "$OUT"
sha256sum "$SCRIPT" > "$OUT/official_script.sha256"
cp "$SCRIPT" "$OUT/official_request.sh"

python3 "$RUNNER" \
  --script "$SCRIPT" \
  --base-url "$BASE_URL" \
  --output-dir "$OUT" \
  --poll-seconds 5 \
  --deadline-seconds 1800

sha256sum "$OUT/output.mp4" > "$OUT/output.sha256"
ffprobe -v error \
  -show_entries format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,sample_rate,channels \
  -of json "$OUT/output.mp4" > "$OUT/ffprobe.json"

python3 "$ROOT/tools/audit_frozen_case_output.py" --case "$case_name"
