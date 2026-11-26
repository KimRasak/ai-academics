"""Minimal distributed-device binding gate for the MM-Diffusion compatibility patch."""

import os
import time

import torch as th

from mm_diffusion import dist_util


def main():
    dist_util.setup_dist(os.environ["MM_DIFFUSION_GPU_LIST"])
    rank = int(os.environ["RANK"])
    print(
        "MM_DIFFUSION_BINDING"
        f" rank={rank}"
        f" pid={os.getpid()}"
        f" visible={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        f" current_device={th.cuda.current_device()}"
        f" device_uuid={th.cuda.get_device_properties(0).uuid}"
        f" device_name={th.cuda.get_device_name(0)}",
        flush=True,
    )
    # This allocation and pause make the worker-to-physical-GPU mapping
    # independently observable in nvidia-smi before the collective gate.
    _allocation = th.empty((256, 1024, 1024), device="cuda:0", dtype=th.float16)
    time.sleep(int(os.environ.get("MM_DIFFUSION_BINDING_HOLD_SECONDS", "0")))
    th.distributed.barrier()
    th.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
