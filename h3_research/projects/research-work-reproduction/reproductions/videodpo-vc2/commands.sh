#!/usr/bin/env bash
set -euo pipefail

git ls-remote https://github.com/CIntellifusion/VideoDPO.git HEAD

audit_root="$(mktemp -d /tmp/videodpo-hf-audit.XXXXXX)"
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://huggingface.co/datasets/CodeGoat24/VideoDPO \
  "$audit_root/dataset"
git -C "$audit_root/dataset" rev-parse HEAD
sed -n '1,4p' "$audit_root/dataset/winvideos.zip"
sed -n '1,4p' "$audit_root/dataset/losevideos.zip"

# Do not run configs/vc2_dpo/run.sh unchanged:
# - it selects a nonexistent config;
# - the README names a nonexistent directory;
# - max_epochs=10 conflicts with the paper's 3000 steps;
# - dupbeta=0 disables OmniScore pair reweighting.
