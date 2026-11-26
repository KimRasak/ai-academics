#!/usr/bin/env bash
free -g > /gluster_hdd_cv/user/jinzili/tmp/pilot/free_gpu.txt 2>&1
ps aux --sort=-%mem 2>/dev/null | head -15 > /gluster_hdd_cv/user/jinzili/tmp/pilot/ps_mem.txt 2>&1
