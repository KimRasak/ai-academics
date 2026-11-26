#!/usr/bin/env python3
"""Generate a resume-safe Sci-VBench testmini H3-Base set from author settings.

The caller supplies a manifest where each row joins one public testmini ID to
its released MiniMax-H3 settings sidecar. Outputs stay per-ID and preserve all
request/poll/media provenance. This runner deliberately does no scoring.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def api(url, payload=None, timeout=90):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def complete(directory):
    media = directory / "output.mp4"
    summary = directory / "runner_summary.json"
    final = directory / "final_status.json"
    if not (media.is_file() and summary.is_file() and final.is_file()):
        return False
    try:
        return json.loads(final.read_text()).get("status") == "completed" and media.stat().st_size > 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--deadline-seconds", type=float, default=1800)
    parser.add_argument("--limit", type=int, default=0, help="0 means every row")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "run_manifest.json", {
        "manifest": args.manifest, "base_url": args.base_url, "count": len(rows),
        "protocol": "published MiniMax-H3 settings per matching Sci-VBench testmini ID",
    })
    terminal = {"completed", "failed", "cancelled", "canceled"}
    for ordinal, row in enumerate(rows):
        case_dir = output_root / row["id"]
        if complete(case_dir):
            print(f"SKIP complete {ordinal + 1}/{len(rows)} {row['id']}", flush=True)
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        setting = row["settings"]
        payload = {
            "task": setting["task"], "prompt": setting["prompt"], "conditions": [],
            "target": {"short_edge": setting["short_edge"], "aspect_ratio": setting["aspect_ratio"],
                       "duration_seconds": setting["duration_seconds"]}, "seed": setting["seed"],
        }
        atomic_json(case_dir / "source.json", row)
        atomic_json(case_dir / "request.json", payload)
        submitted = api(args.base_url.rstrip("/") + "/v1/videos", payload)
        atomic_json(case_dir / "submit_response.json", submitted)
        video_id = submitted["id"]
        history = []
        deadline = time.monotonic() + args.deadline_seconds
        while True:
            status = api(args.base_url.rstrip("/") + f"/v1/videos/{video_id}")
            history.append({"observed_at": time.time(), "response": status})
            atomic_json(case_dir / "status_history.json", history)
            state = str(status.get("status", "")).lower()
            if state in terminal:
                atomic_json(case_dir / "final_status.json", status)
                if state != "completed":
                    raise RuntimeError(f"{row['id']} terminal state {state}")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{row['id']} exceeded {args.deadline_seconds} seconds")
            time.sleep(args.poll_seconds)
        with urllib.request.urlopen(args.base_url.rstrip("/") + f"/v1/videos/{video_id}/content", timeout=300) as response:
            media = response.read()
        (case_dir / "output.mp4").write_bytes(media)
        atomic_json(case_dir / "runner_summary.json", {
            "ordinal": ordinal, "id": row["id"], "video_id": video_id, "status": "completed",
            "bytes": len(media), "sha256": hashlib.sha256(media).hexdigest(), "polls": len(history),
        })
        print(f"DONE {ordinal + 1}/{len(rows)} {row['id']} {len(media)} bytes", flush=True)


if __name__ == "__main__":
    main()
