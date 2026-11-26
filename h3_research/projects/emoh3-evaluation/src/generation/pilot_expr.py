#!/usr/bin/env python3
"""Pilot: cross-emotion anime expression generation via H3 Ref2VA on node 034."""
from __future__ import annotations
import json, os, shutil, time, urllib.error, urllib.request
from pathlib import Path

BASE = os.environ.get("H3_BASE", "http://osa00-h100-034.osa00.bdjp.io:30020")
REF_DIR = Path("/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3/inputs/expressions_0825")
OUT_DIR = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/outputs")
CASES = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/pilot_cases.json")
RESULTS = Path("/gluster_hdd_cv/user/jinzili/tmp/pilot/results.jsonl")
OUT_DIR.mkdir(parents=True, exist_ok=True)
POLL_S, CASE_TIMEOUT_S = 5, 1800
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def api(path, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE+path, data=data,
        headers={"Content-Type":"application/json"} if data else {},
        method="POST" if data else "GET")
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read())

def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

cases = json.loads(CASES.read_text())
done = set()
if RESULTS.exists():
    for line in RESULTS.read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("status") == "completed":
                done.add(r["case_id"])
        except Exception:
            pass

log("target=%s cases=%d already_done=%d" % (BASE, len(cases), len(done)))
ok = 0
for case in cases:
    cid = case["case_id"]
    if cid in done:
        log("skip %s (done)" % cid)
        continue
    ref = REF_DIR / case["ref"]
    if not ref.is_file():
        log("MISSING ref %s" % ref)
        continue
    payload = {"task": "ref2va", "prompt": case["prompt"],
        "conditions": [{"type": "image", "uri": "file://%s" % ref, "role": "reference"}],
        "target": {"short_edge": 768, "aspect_ratio": case["aspect_ratio"],
                   "duration_seconds": case["seconds"]},
        "seed": case["seed"]}
    try:
        sub = api("/v1/videos", payload)
    except Exception as e:
        log("%s submit_failed %r" % (cid, e))
        RESULTS.open("a").write(json.dumps({"case_id": cid, "status": "submit_failed", "error": repr(e)})+"\n")
        continue
    vid = sub.get("id")
    if not vid:
        log("%s submit_failed no id: %s" % (cid, str(sub)[:200]))
        RESULTS.open("a").write(json.dumps({"case_id": cid, "status": "submit_failed", "resp": str(sub)[:200]})+"\n")
        continue
    info = {}
    t0 = time.time()
    while True:
        time.sleep(POLL_S)
        try:
            info = api("/v1/videos/%s" % vid, timeout=30)
        except Exception as e:
            log("  poll %s err %r" % (cid, e))
            continue
        st = info.get("status")
        if st in ("completed", "failed"):
            break
        if time.time()-t0 > CASE_TIMEOUT_S:
            info = {"status": "timeout"}
            break
    rec = {"case_id": cid, "video_id": vid, "status": info.get("status"),
           "emotion": case["emotion"], "prompt": case["prompt"], "char": case["char"]}
    if info.get("status") == "completed":
        vp = info.get("video") or {}
        src = Path(str(vp.get("path") or ""))
        if not src.is_file():
            src = Path("/gluster_hdd_cv/user/jinzili/code/anime_style/model_demo/minimax_h3/outputs/webdemo_scratch") / ("%s.mp4" % vid)
        if src.is_file():
            dst = OUT_DIR / ("%s.mp4" % cid)
            shutil.copy(src, dst)
            rec["output"] = str(dst)
            rec["size"] = src.stat().st_size
        else:
            rec["note"] = "video_path_not_found"
        ok += 1
    RESULTS.open("a").write(json.dumps(rec)+"\n")
    log("%s: %s -> %s" % (cid, rec["status"], rec.get("output", "-")))
log("done: %d/%d" % (ok, len(cases)))
