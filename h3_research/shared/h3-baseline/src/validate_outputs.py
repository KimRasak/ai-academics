#!/usr/bin/env python3
"""Validate pilot outputs: ffprobe + frame extraction + basic motion stats."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

OUT_DIR = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/outputs")
RESULTS = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/results.jsonl")
FRAMES = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/frames")
FRAMES.mkdir(parents=True, exist_ok=True)
FFPROBE = "/gluster_hdd_cv/user/jinzili/env/sglang310/bin/ffprobe"

def ffprobe(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                        "stream=codec_type,codec_name,width,height,nb_frames,duration",
                        "-of", "json", str(path)], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"error": r.stderr[:200]}

def extract_frames(mp4, n=6):
    stem = mp4.stem
    outdir = FRAMES / stem
    outdir.mkdir(parents=True, exist_ok=True)
    # sample n frames evenly using ffmpeg
    for i in range(n):
        t = i / max(n - 1, 1)
        out = outdir / ("f%02d.jpg" % i)
        subprocess.run([FFPROBE.replace("ffprobe", "ffmpeg"), "-v", "error",
                        "-ss", "%.2f" % t, "-i", str(mp4), "-frames:v", "1",
                        "-q:v", "2", "-y", str(out)], capture_output=True)
    return sorted(outdir.glob("*.jpg"))

summary = []
for mp4 in sorted(OUT_DIR.glob("*.mp4")):
    info = ffprobe(mp4)
    streams = info.get("streams", []) if isinstance(info, dict) else []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    frames = extract_frames(mp4)
    rec = {
        "file": mp4.name, "size": mp4.stat().st_size,
        "video": v, "audio": a, "n_frames_extracted": len(frames),
        "ffprobe_error": info.get("error"),
    }
    summary.append(rec)
    print(json.dumps(rec, ensure_ascii=False))

print("TOTAL", len(summary))
