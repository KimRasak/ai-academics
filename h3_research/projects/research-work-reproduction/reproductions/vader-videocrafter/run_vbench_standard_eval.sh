#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT {base|pickscore-lora}" >&2
  exit 2
fi

RUN_ROOT=$1
CONDITION=$2
case "$CONDITION" in
  base|pickscore-lora) ;;
  *)
    echo "unsupported condition: $CONDITION" >&2
    exit 2
    ;;
esac

VBENCH_ROOT=/gluster_hdd_cv/user/jinzili/code/research-work-reproduction/VBench
VBENCH_ENV=/gluster_hdd_cv/user/jinzili/env/vbench-temporal
VBENCH_CACHE=/gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/vbench
VIDEOS_PATH=$RUN_ROOT/$CONDITION/standard
OUTPUT_PATH=$RUN_ROOT/$CONDITION/evaluation

if [[ $(find "$VIDEOS_PATH" -maxdepth 1 -name '*.mp4' | wc -l) -ne 360 ]]; then
  echo "$VIDEOS_PATH does not contain exactly 360 MP4 files" >&2
  exit 1
fi

mkdir -p "$OUTPUT_PATH"
cd "$VBENCH_ROOT"

VBENCH_CACHE_DIR="$VBENCH_CACHE" \
"$VBENCH_ENV/bin/python" evaluate.py \
  --output_path "$OUTPUT_PATH" \
  --full_json_dir "$VBENCH_ROOT/vbench/VBench_full_info.json" \
  --videos_path "$VIDEOS_PATH" \
  --dimension dynamic_degree motion_smoothness \
  --load_ckpt_from_local True \
  --mode vbench_standard
