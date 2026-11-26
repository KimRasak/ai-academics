#!/usr/bin/env bash
set -euo pipefail

# Audit a completed MM-Diffusion metric-run directory without modifying media.
# It writes derived evidence only: media inventory, ffprobe failures/streams,
# checksums and checksum-duplicate reports.
if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT AUDIT_DIRECTORY" >&2
  exit 64
fi

run_root=$1
audit_dir=$2
model_root="$run_root/landscape.pt"

for dir in "$model_root/original" "$model_root/sr_mp4" "$model_root/audios"; do
  if [[ ! -d "$dir" ]]; then
    echo "missing expected media directory: $dir" >&2
    exit 65
  fi
done

mkdir -p "$audit_dir"
find "$model_root/original" -type f -name '*.mp4' -print | LC_ALL=C sort > "$audit_dir/original_mp4_paths.txt"
find "$model_root/sr_mp4" -type f -name '*.mp4' -print | LC_ALL=C sort > "$audit_dir/sr_mp4_paths.txt"
find "$model_root/audios" -type f -name '*.wav' -print | LC_ALL=C sort > "$audit_dir/wav_paths.txt"

{
  printf 'original_mp4='; wc -l < "$audit_dir/original_mp4_paths.txt"
  printf 'sr_mp4='; wc -l < "$audit_dir/sr_mp4_paths.txt"
  printf 'wav='; wc -l < "$audit_dir/wav_paths.txt"
} > "$audit_dir/counts.txt"

cat "$audit_dir/original_mp4_paths.txt" "$audit_dir/sr_mp4_paths.txt" "$audit_dir/wav_paths.txt" > "$audit_dir/all_media_paths.txt"
while IFS= read -r media; do
  sha256sum "$media"
done < "$audit_dir/all_media_paths.txt" > "$audit_dir/media.sha256"
cut -d' ' -f1 "$audit_dir/media.sha256" | LC_ALL=C sort | uniq -cd > "$audit_dir/byte_identical_duplicate_hashes.txt"

: > "$audit_dir/ffprobe_failures.txt"
: > "$audit_dir/ffprobe_streams.txt"
while IFS= read -r media; do
  if ! ffprobe -v error -show_entries stream=codec_type,codec_name,width,height,sample_rate,channels \
      -of compact=p=0:nk=1 "$media" >> "$audit_dir/ffprobe_streams.txt" 2>> "$audit_dir/ffprobe_failures.txt"; then
    printf '%s\n' "$media" >> "$audit_dir/ffprobe_failures.txt"
  fi
done < "$audit_dir/all_media_paths.txt"

printf 'ffprobe_failure_lines=' > "$audit_dir/summary.txt"
wc -l < "$audit_dir/ffprobe_failures.txt" >> "$audit_dir/summary.txt"
cat "$audit_dir/counts.txt" >> "$audit_dir/summary.txt"
printf 'byte_identical_duplicate_hash_lines=' >> "$audit_dir/summary.txt"
wc -l < "$audit_dir/byte_identical_duplicate_hashes.txt" >> "$audit_dir/summary.txt"
