#!/usr/bin/env bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > /gluster_hdd_cv/user/jinzili/tmp/pilot/gpu_lora_infer.txt 2>&1
