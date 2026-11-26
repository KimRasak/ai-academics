#!/usr/bin/env python3
"""Read-only integrity verifier for the MiniMax-H3 R0 freeze manifest."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(label: str, path: Path, expected: str) -> bool:
    if not path.is_file():
        print(f"FAIL {label}: missing {path}")
        return False
    actual = sha256(path)
    if actual != expected:
        print(f"FAIL {label}: sha256={actual}, expected={expected}")
        return False
    print(f"PASS {label}: {actual}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("t2va", "fl2va", "ref2va"), required=True)
    parser.add_argument("--root", default="/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/minimax-h3-official-768p")
    parser.add_argument("--model-root", default="/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3")
    args = parser.parse_args()

    root = Path(args.root)
    model = Path(args.model_root)
    manifest_path = root / "R0_FREEZE_MANIFEST.json"
    if not manifest_path.is_file():
        print(f"FAIL manifest missing: {manifest_path}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["official_request_scripts"][args.case]
    passed = True
    passed &= check("official README", model / "README.md", manifest["official_release"]["readme_sha256"])
    passed &= check(
        f"official {args.case} request",
        model / "scripts" / "readme" / f"reproducible-768p-{args.case}-request.sh",
        case["sha256"],
    )
    for relative, expected in manifest["local_execution_tools"].items():
        passed &= check(relative, root / relative, expected)

    print(
        "ROUTING "
        f"case={args.case} partition={case['partition']} port={case['port']} seed={case['seed']}"
    )
    print("decision=" + ("FREEZE_VERIFIED" if passed else "DO_NOT_SUBMIT"))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
