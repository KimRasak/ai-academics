# MotionCtrl environments

## Release-resolution environment

Remote path:
`/gluster_hdd_cv/user/jinzili/env/motionctrl-release-20250219`

The repository's requirements resolve to Torch 1.13.1/CUDA 11.7 and xFormers
0.0.16 after restoring period-compatible unpinned libraries. Imports pass, but
the environment is not runnable on H100:

- Torch reports that the wheel contains no `sm_90` kernels;
- xFormers attention fails with `Expected is_sm8x || is_sm75`.

The 121-line initial-resolution freeze is stored remotely at
`weights/research-work-reproduction/motionctrl/environment.freeze.txt`, SHA-256
`31ce1199ec709baae55e099502e11d3c47b3dffc77378a6b1994869f2a601bc0`.

## H100-compatible adapted environment

Remote path:
`/gluster_hdd_cv/user/jinzili/env/motionctrl-h100-adapted`

Compatibility changes are explicit:

- Torch 1.13.1/CUDA 11.7 → Torch 2.3.0/CUDA 12.1;
- xFormers 0.0.16 → 0.0.26.post1;
- unpinned libraries are frozen to mutually compatible versions.

Imports and a CUDA xFormers attention smoke pass on H100. The 81-line freeze is
stored remotely at
`weights/research-work-reproduction/motionctrl/environment.h100-adapted.freeze.txt`,
SHA-256
`c6cf9f7fd48b43a6bc006e11afc0d7b96da2708ee5c82ff49696e48afa59f945`.

This environment is a compatibility adaptation, not an exact reconstruction of
the authors' V100 runtime.
