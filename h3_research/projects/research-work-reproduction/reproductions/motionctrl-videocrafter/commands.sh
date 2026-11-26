#!/usr/bin/env bash
set -euo pipefail

# Read-only source identity.
git ls-remote https://github.com/TencentARC/MotionCtrl.git \
  HEAD refs/heads/main refs/heads/animatediff

# Clone checkpoint metadata without downloading the 4–10 GB LFS objects.
audit_root="$(mktemp -d /tmp/motionctrl-hf-audit.XXXXXX)"
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://huggingface.co/TencentARC/MotionCtrl \
  "$audit_root/repo"
git -C "$audit_root/repo" rev-parse HEAD
sed -n '1,4p' "$audit_root/repo/motionctrl.pth"

# H100-compatible single-condition checkpoint smoke.
CUDA_VISIBLE_DEVICES=2 \
  /gluster_hdd_cv/user/jinzili/env/motionctrl-h100-adapted/bin/python \
  smoke_generate.py \
  --repo /gluster_hdd_cv/user/jinzili/code/research-work-reproduction/motionctrl-videocrafter \
  --checkpoint /gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/motionctrl/motionctrl.pth \
  --output-root /gluster_hdd_cv/user/jinzili/data/research-work-reproduction/motionctrl/outputs/camera-L-seed1234-v1

# Do not run the released training scripts unchanged:
# - their 8-GPU settings are overwritten by one-GPU debug settings;
# - train_omcm_sparse.sh selects the dense config;
# - released configs use 1,000,000 steps rather than 50k/20k/20k.
