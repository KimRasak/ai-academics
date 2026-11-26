#!/usr/bin/env bash
ps aux 2>/dev/null | grep -E "infer_lora_ref2va|run_lora_infer" | grep -v grep > /gluster_hdd_cv/user/jinzili/tmp/pilot/procs.txt 2>&1
ls -la /gluster_hdd_cv/user/jinzili/tmp/pilot/lora_infer_rank*.log >> /gluster_hdd_cv/user/jinzili/tmp/pilot/procs.txt 2>&1
