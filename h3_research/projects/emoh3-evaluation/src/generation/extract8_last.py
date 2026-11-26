#!/usr/bin/env python3
"""Control for the frame-span result: 8 frames from the LAST second only.

If accuracy under last-second sampling matches full-span sampling and both beat first-second sampling,
the effect is that the expression *develops late*, not that wider coverage helps per se. This
distinguishes the two explanations the frame-span axis leaves open.
"""
import subprocess, sys, json
from pathlib import Path

SRC = Path(sys.argv[1]); DST = Path(sys.argv[2])
FFMPEG = "/gluster_hdd_cv/user/jinzili/env/sglang310/bin/ffmpeg"
FFPROBE = "/gluster_hdd_cv/user/jinzili/env/sglang310/bin/ffprobe"
DST.mkdir(parents=True, exist_ok=True)
n = 8
for mp4 in sorted(SRC.glob("*.mp4")):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                        "-of", "json", str(mp4)], capture_output=True, text=True)
    try:
        dur = float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        print("SKIP", mp4.stem, flush=True); continue
    t0 = max(0.0, dur - 1.05)
    outdir = DST / mp4.stem
    outdir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        t = t0 + (dur - 0.05 - t0) * i / (n - 1)
        subprocess.run([FFMPEG, "-v", "error", "-ss", "%.3f" % t, "-i", str(mp4),
                        "-frames:v", "1", "-q:v", "2", "-y", str(outdir / ("f%02d.jpg" % i))],
                       check=False)
    print("OK", mp4.stem, "t=%.2f..%.2f" % (t0, dur - 0.05), flush=True)
