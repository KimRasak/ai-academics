#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 FAKE_SUBDIR EVAL_RUN_ID GPU" >&2
  exit 64
fi
fake_subdir=$1
eval_run_id=$2
gpu=$3
case "$fake_subdir" in original|sr_mp4) ;; *) exit 65 ;; esac

MMROOT=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/mm-diffusion
SAMPLE_RUN=ddpm_ddim25_metric_run03_officialrng_gpu4567_20260920
export CUDA_VISIBLE_DEVICES="$gpu"
export LD_LIBRARY_PATH="$MMROOT/conda-env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$MMROOT/conda-env/bin:$PATH"
export PYTHONPATH="$MMROOT/code"
export HOME="$MMROOT/runtime-home"

exec "$MMROOT/conda-env/bin/mpiexec" -n 1 "$MMROOT/env/bin/python" \
  "$MMROOT/code/py_scripts/eval.py" --devices 0 --sample_num 2048 \
  --ref_dir "$MMROOT/raw/extracted/landscape/train" \
  --fake_dir "$MMROOT/outputs/$SAMPLE_RUN/landscape.pt/$fake_subdir" \
  --output_dir "$MMROOT/outputs/$eval_run_id"
