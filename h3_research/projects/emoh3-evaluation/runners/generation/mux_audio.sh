#!/usr/bin/env bash
# Mux the per-case wav into the video-only preservation mp4s.
set -u
export PATH=/gluster_hdd_cv/user/jinzili/env/sglang310/bin:/usr/bin:$PATH
D=/gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_preserve
for w in "$D"/pres_*.wav; do
  base=${w%.wav}
  mp4=${base}.mp4
  [ -f "$mp4" ] || { echo "NO MP4: $mp4"; continue; }
  ffmpeg -y -loglevel error -i "$mp4" -i "$w" -map 0:v -map 1:a -c:v copy -c:a aac -shortest "$mp4.muxed.mp4" && mv "$mp4.muxed.mp4" "$mp4" && echo "MUXED $mp4"
done
echo "MUX_DONE"
