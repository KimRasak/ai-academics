#!/usr/bin/env python3
"""Exp 0: Inspect osa00-h100-007 environment, GPUs, Python envs, Blender, and cached open-weight models."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess


def sh(cmd: str) -> str:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=15)
        return out.strip()
    except Exception as exc:
        return f"[ERR] {exc}"


def main() -> None:
    print("=== HOST & GPUS ===")
    print(sh("hostname"))
    print(sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader"))

    print("\n=== PYTHON ENVS & BLENDER ===")
    print("which python3:", shutil.which("python3"))
    print("which blender:", shutil.which("blender"))
    for p in sorted(glob.glob("/gluster_hdd_cv/user/jinzili/env/*")):
        print("  env:", p)

    print("\n=== MODEL CACHES (HF / GLUSTER) ===")
    search_roots = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/gluster_hdd_cv/user/jinzili/models",
        "/gluster_hdd_cv/user/jinzili/data",
        "/gluster_hdd_cv/models",
        "/data/models",
    ]
    keywords = ("qwen", "wan", "ltx", "flux", "depth", "vggt", "moge", "ic-light", "relight", "ccip", "dino", "siglip", "internvl", "dreamx")
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        print(f"Scanning {root}:")
        try:
            for entry in sorted(os.listdir(root)):
                low = entry.lower()
                if any(k in low for k in keywords) or root.endswith("hub"):
                    print(f"  {os.path.join(root, entry)}")
        except Exception as exc:
            print(f"  [ERR] {exc}")

    print("\n=== NETWORK / HF REACHABILITY ===")
    print(sh("curl -sI -m 5 https://huggingface.co | head -n 1 || echo HF_UNREACHABLE"))


if __name__ == "__main__":
    main()
