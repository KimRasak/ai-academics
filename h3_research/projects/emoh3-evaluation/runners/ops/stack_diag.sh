#!/usr/bin/env bash
which py-spy 2>/dev/null > /gluster_hdd_cv/user/jinzili/tmp/pilot/which_diag.txt
which gdb 2>/dev/null >> /gluster_hdd_cv/user/jinzili/tmp/pilot/which_diag.txt
which strace 2>/dev/null >> /gluster_hdd_cv/user/jinzili/tmp/pilot/which_diag.txt
PID=$(pgrep -f "infer_lora_ref2va.py" | head -1)
echo "target pid=$PID" >> /gluster_hdd_cv/user/jinzili/tmp/pilot/which_diag.txt
cat /proc/$PID/status 2>/dev/null | grep -E "State|VmRSS|Threads" >> /gluster_hdd_cv/user/jinzili/tmp/pilot/which_diag.txt
ls /proc/$PID/task 2>/dev/null | wc -l >> /gluster_hdd_cv/user/jinzili/tmp/pilot/which_diag.txt
cat /proc/$PID/stack 2>/dev/null >> /gluster_hdd_cv/user/jinzili/tmp/pilot/which_diag.txt
