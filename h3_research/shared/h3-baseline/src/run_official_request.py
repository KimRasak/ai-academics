#!/usr/bin/env python3
"""Run an archived official MiniMax-H3 request with durable polling evidence."""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def request_json(url, *, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_payload(script_path):
    text = Path(script_path).read_text(encoding="utf-8")
    match = re.search(r"<<'JSON'\n(.*?)\nJSON", text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON heredoc not found in {script_path}")
    return json.loads(match.group(1))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--deadline-seconds", type=float, default=1800)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = extract_payload(args.script)
    atomic_json(out / "request.json", payload)

    created = request_json(args.base_url.rstrip("/") + "/v1/videos", payload=payload)
    atomic_json(out / "submit_response.json", created)
    video_id = created["id"]
    status_url = args.base_url.rstrip("/") + f"/v1/videos/{video_id}"
    history = []
    deadline = time.monotonic() + args.deadline_seconds
    terminal = {"completed", "failed", "cancelled", "canceled"}
    while True:
        status = request_json(status_url)
        history.append({
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "response": status,
        })
        atomic_json(out / "status_history.json", history)
        state = str(status.get("status", "")).lower()
        if state in terminal:
            atomic_json(out / "final_status.json", status)
            if state != "completed":
                raise SystemExit(f"generation ended with status={state}")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"generation did not finish within {args.deadline_seconds}s")
        time.sleep(args.poll_seconds)

    content_url = status_url + "/content"
    with urllib.request.urlopen(content_url, timeout=300) as response:
        video = response.read()
    (out / "output.mp4").write_bytes(video)
    atomic_json(out / "runner_summary.json", {
        "script": str(args.script),
        "base_url": args.base_url,
        "video_id": video_id,
        "status": "completed",
        "polls": len(history),
        "output_bytes": len(video),
    })


if __name__ == "__main__":
    main()
