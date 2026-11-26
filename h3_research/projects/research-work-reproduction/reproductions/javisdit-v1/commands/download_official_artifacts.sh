#!/usr/bin/env bash
set -euo pipefail

# Download only official released artifacts needed by the preregistered
# JavisDiT-v1 JavisBench-mini checkpoint-verification protocol.  This script
# does not launch inference, evaluation, or training.

artifact_root=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/javisdit-v1
weights_root=/gluster_hdd_cv/user/jinzili/weights/javisdit-v1-official
logs_dir="$artifact_root/logs"
manifest_dir="$artifact_root/manifests"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)

mkdir -p "$artifact_root/benchmark" "$weights_root" "$logs_dir" "$manifest_dir"

hf download JavisVerse/JavisDiT-v1.0-jav \
  --local-dir "$weights_root/JavisDiT-v1.0-jav" \
  --max-workers 4 \
  2>&1 | tee "$logs_dir/download_javisdit_${timestamp}.log"

hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir "$weights_root/Wan2.1-T2V-1.3B" \
  --max-workers 4 \
  2>&1 | tee "$logs_dir/download_wan_${timestamp}.log"

hf download cvssp/audioldm2 \
  --local-dir "$weights_root/audioldm2" \
  --max-workers 4 \
  2>&1 | tee "$logs_dir/download_audioldm2_${timestamp}.log"

hf download JavisVerse/JavisBench \
  --repo-type dataset \
  --local-dir "$artifact_root/benchmark/JavisBench" \
  --max-workers 4 \
  2>&1 | tee "$logs_dir/download_javisbench_${timestamp}.log"

find "$weights_root/JavisDiT-v1.0-jav" "$weights_root/Wan2.1-T2V-1.3B" \
  "$weights_root/audioldm2" "$artifact_root/benchmark/JavisBench" \
  -type f -not -path '*/.cache/*' -print0 | sort -z | xargs -0 sha256sum \
  > "$manifest_dir/official_artifacts_${timestamp}.sha256"

{
  printf 'download_completed_at=%s\n' "$timestamp"
  printf 'javisdit_model_repo=JavisVerse/JavisDiT-v1.0-jav\n'
  printf 'wan_repo=Wan-AI/Wan2.1-T2V-1.3B\n'
  printf 'audioldm2_repo=cvssp/audioldm2\n'
  printf 'benchmark_repo=JavisVerse/JavisBench\n'
  printf 'sha256_inventory=%s\n' "$manifest_dir/official_artifacts_${timestamp}.sha256"
} > "$manifest_dir/official_artifacts_${timestamp}.provenance.txt"
