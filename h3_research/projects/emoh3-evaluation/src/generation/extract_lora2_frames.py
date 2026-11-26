#!/usr/bin/env python3
"""Extract 4 evenly-sampled frames from outputs_lora2/*.mp4 into frames_lora2/<stem>/."""
import subprocess
from pathlib import Path

OUT = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_lora2")
FRAMES = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_lora2")
FFMPEG = "/gluster_hdd_cv/user/jinzili/env/sglang310/bin/ffmpeg"
FRAMES.mkdir(parents=True, exist_ok=True)
n = 4
for mp4 in sorted(OUT.glob("*.mp4")):
    stem = mp4.stem
    outdir = FRAMES / stem
    outdir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        t = i / max(n - 1, 1)
        out = outdir / ("f%02d.jpg" % i)
        subprocess.run(
            [FFMPEG, "-v", "error", "-ss", "%.2f" % t, "-i", str(mp4),
             "-frames:v", "1", "-q:v", "2", "-y", str(out)],
            capture_output=True,
        )
    print(stem, len(list(outdir.glob("*.jpg"))), flush=True)
