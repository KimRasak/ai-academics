# MotionCtrl outputs

Remote media:

`/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/motionctrl/outputs/camera-L-seed1234-v1`

The first smoke uses the released `test_camera_L` condition and matching
coastline prompt with seed 1234. The official checkpoint loads with zero missing
and zero unexpected keys. Its single H.264 video passes full decode.

Visual inspection of frames 0/5/10/15 shows a coherent coastline with visible
lateral camera motion and minor boundary deformation. This is deployment
evidence only; it is not CamMC, FID or FVD reproduction.
