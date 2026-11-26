#!/usr/bin/env python3
"""Extract 8 evenly-sampled frames from specified mp4s into a dir (frame-count check)."""
import subprocess, sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_lora2")
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_lora2_8")
CASES = sys.argv[3:] if len(sys.argv) > 3 else None
FFMPEG = "/gluster_hdd_cv/user/jinzili/env/sglang310/bin/ffmpeg"
DST.mkdir(parents=True, exist_ok=True)
n = 8
for mp4 in sorted(SRC.glob("*.mp4")):
    if CASES and not any(c in mp4.stem for c in CASES):
        continue
    outdir = DST / mp4.stem
    outdir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        t = i / max(n - 1, 1)
        out = outdir / ("f%02d.jpg" % i)
        subprocess.run(
            [FFMPEG, "-v", "error", "-ss", "%.2f" % t, "-i", str(mp4),
             "-frames:v", "1", "-q:v", "2", "-y", str(out)],
            capture_output=True,
        )
    print(mp4.stem, len(list(outdir.glob("*.jpg"))), flush=True)
