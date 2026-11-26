#!/usr/bin/env python3
"""Extract 4 evenly-sampled frames from <OUT>/*.mp4 into <FRAMES>/<stem>/.
Paths from env H3_FRAMES_OUT / H3_FRAMES_DIR (defaults: outputs_lora -> frames_lora).
"""
import os, subprocess
from pathlib import Path

OUT = Path(os.environ.get("H3_FRAMES_OUT", "/gluster_hdd_cv/user/jinzili/tmp/pilot/outputs_lora"))
FRAMES = Path(os.environ.get("H3_FRAMES_DIR", "/gluster_hdd_cv/user/jinzili/tmp/pilot/frames_lora"))
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
