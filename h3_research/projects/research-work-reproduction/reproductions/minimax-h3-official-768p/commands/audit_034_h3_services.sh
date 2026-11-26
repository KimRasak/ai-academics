#!/usr/bin/env bash
set -euo pipefail

# This is executed only by the project-owned gpu_launcher on node 034. It is
# read-only: identify every resident MiniMax-H3 SGLang service before deciding
# whether an older project service can be retired for a new auditable Base run.

OUT=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/minimax-h3-official-768p/server_logs/audit_034_h3_services_20260922.txt
mkdir -p "$(dirname "$OUT")"

{
  echo "checked_at_utc=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname)"
  echo "--- launcher-host H3 SGLang processes ---"
  ps -eo pid,ppid,pgid,user,stat,etimes,args | grep '[s]glang serve' | grep 'MiniMax-H3' || true
  echo "--- H3-related process tree ---"
  ps -eo pid,ppid,pgid,user,stat,etimes,args | grep -E '[s]glang|[m]inimax_h3|[M]iniMax-H3' || true
  echo "--- service ports ---"
  ss -ltnp 2>/dev/null | grep -E ':(30010|30011|30020|30030|30031)\b' || true
  echo "--- GPU compute applications ---"
  nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
} | tee "$OUT"

echo "audit_file=$OUT"
