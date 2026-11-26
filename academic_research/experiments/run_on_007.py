#!/usr/bin/env python3
"""Clean helper to dispatch a command to osa00-h100-007.osa00.bdjp.io:8659 via jump host h100-006-v2."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time


def dispatch(cmd: str, log_path: str | None = None, wait_sec: int = 0) -> None:
    cmd_b64 = base64.b64encode(cmd.encode("utf-8")).decode("ascii")
    if log_path:
        wrapped = f"bash -c \"echo {cmd_b64} | base64 -d | bash; echo __JOB_EXIT_\\$?__\" > {log_path} 2>&1"
    else:
        wrapped = f"echo {cmd_b64} | base64 -d | bash"
    payload_b64 = base64.b64encode(json.dumps({"cmd": wrapped}).encode("utf-8")).decode("ascii")
    remote_py = (
        "import base64, urllib.request; "
        f"data = base64.b64decode('{payload_b64}'); "
        "op = urllib.request.build_opener(urllib.request.ProxyHandler({})); "
        "req = urllib.request.Request('http://osa00-h100-007.osa00.bdjp.io:8659/run', data=data, "
        "headers={'Content-Type': 'application/json'}, method='POST'); "
        "print(op.open(req, timeout=20).read().decode())"
    )
    out = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "h100-006-v2", f"python3 -c {json.dumps(remote_py)}"],
        text=True,
    )
    print(out.strip(), flush=True)

    if log_path and wait_sec > 0:
        deadline = time.time() + wait_sec
        while time.time() < deadline:
            time.sleep(3)
            chk = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "h100-006-v2", f"tail -n 30 {log_path} 2>/dev/null || true"],
                capture_output=True,
                text=True,
            )
            if "__JOB_EXIT_" in chk.stdout:
                print("\n=== LOG OUTPUT ===")
                full = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", "h100-006-v2", f"cat {log_path}"],
                    capture_output=True,
                    text=True,
                )
                print(full.stdout)
                return
        print(f"[Still running after {wait_sec}s; log at {log_path}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--wait", type=int, default=0)
    args = ap.parse_args()
    dispatch(args.cmd, args.log, args.wait)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
