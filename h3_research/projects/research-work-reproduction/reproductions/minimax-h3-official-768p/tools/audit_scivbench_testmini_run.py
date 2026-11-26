#!/usr/bin/env python3
"""Read-only closeout audit for a completed MiniMax-H3 Sci-VBench testmini run.

The generator keeps one directory per published Sci-VBench ID.  This audit
checks the complete directory contract before any evaluator is allowed to see
the videos.  It does not score, alter, copy, or regenerate media.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REQUIRED = (
    "source.json", "request.json", "submit_response.json", "status_history.json",
    "final_status.json", "runner_summary.json", "output.mp4",
)


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ffprobe(path, binary):
    command = [binary, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        return None, result.stderr.strip() or f"ffprobe exited {result.returncode}"
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"invalid ffprobe JSON: {exc}"


def expected_request(row):
    setting = row["settings"]
    return {
        "task": setting["task"], "prompt": setting["prompt"], "conditions": [],
        "target": {"short_edge": setting["short_edge"], "aspect_ratio": setting["aspect_ratio"],
                   "duration_seconds": setting["duration_seconds"]}, "seed": setting["seed"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    expected = {row["id"]: row for row in rows}
    if len(expected) != len(rows):
        raise SystemExit("manifest has duplicate IDs")

    report = {"schema": 1, "kind": "read_only_scivbench_testmini_h3_media_audit",
              "manifest": str(args.manifest), "output_root": str(args.output_root),
              "expected_cases": len(rows), "cases": [], "global_failures": [],
              "byte_identical_duplicate_hashes": []}
    found_dirs = {path.name for path in args.output_root.iterdir() if path.is_dir()} if args.output_root.exists() else set()
    extra = sorted(found_dirs - set(expected))
    if extra:
        report["global_failures"].append({"extra_case_directories": extra})

    hashes = Counter()
    failures = 0
    for case_id, row in expected.items():
        case_dir = args.output_root / case_id
        case = {"id": case_id, "directory": str(case_dir), "failures": []}
        missing = [name for name in REQUIRED if not (case_dir / name).is_file()]
        if missing:
            case["failures"].append({"missing_required_files": missing})
        if not missing:
            try:
                case["source_matches_manifest"] = load_json(case_dir / "source.json") == row
                if not case["source_matches_manifest"]:
                    case["failures"].append("source.json differs from frozen manifest row")
                request = load_json(case_dir / "request.json")
                case["request_matches_author_settings"] = request == expected_request(row)
                if not case["request_matches_author_settings"]:
                    case["failures"].append("request.json differs from author settings payload")
                final = load_json(case_dir / "final_status.json")
                case["final_status"] = final.get("status")
                if final.get("status") != "completed":
                    case["failures"].append("final status is not completed")
                history = load_json(case_dir / "status_history.json")
                case["poll_records"] = len(history) if isinstance(history, list) else None
                if not isinstance(history, list) or not history:
                    case["failures"].append("status history is empty or not a list")
                summary = load_json(case_dir / "runner_summary.json")
                media = case_dir / "output.mp4"
                actual_hash, actual_bytes = sha256(media), media.stat().st_size
                case["sha256"] = actual_hash
                case["bytes"] = actual_bytes
                hashes[actual_hash] += 1
                if summary.get("id") != case_id or summary.get("status") != "completed":
                    case["failures"].append("runner summary ID or status mismatch")
                if summary.get("sha256") != actual_hash or summary.get("bytes") != actual_bytes:
                    case["failures"].append("runner summary media hash or byte count mismatch")
                probe, error = ffprobe(media, args.ffprobe)
                case["ffprobe"] = probe
                if error:
                    case["failures"].append({"ffprobe": error})
                else:
                    streams = probe.get("streams", [])
                    video = [s for s in streams if s.get("codec_type") == "video"]
                    audio = [s for s in streams if s.get("codec_type") == "audio"]
                    case["media_contract"] = {
                        "video_streams": len(video), "audio_streams": len(audio),
                        "video": {key: video[0].get(key) for key in ("codec_name", "width", "height", "r_frame_rate")}
                                 if len(video) == 1 else None,
                        "audio": {key: audio[0].get(key) for key in ("codec_name", "sample_rate", "channels")}
                                 if len(audio) == 1 else None,
                        "duration_seconds": probe.get("format", {}).get("duration"),
                    }
                    if len(video) != 1 or len(audio) != 1:
                        case["failures"].append("expected exactly one video and one audio stream")
                    elif (video[0].get("codec_name"), video[0].get("width"), video[0].get("height"),
                          video[0].get("r_frame_rate"), audio[0].get("codec_name"),
                          audio[0].get("sample_rate"), audio[0].get("channels")) != ("h264", 1344, 768, "24/1", "aac", "32000", 2):
                        case["failures"].append("media stream contract differs from expected 1344x768/24fps H.264 + 32kHz stereo AAC")
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                case["failures"].append(f"audit parsing exception: {type(exc).__name__}: {exc}")
        if case["failures"]:
            failures += 1
        report["cases"].append(case)

    report["byte_identical_duplicate_hashes"] = [digest for digest, count in hashes.items() if count > 1]
    if report["byte_identical_duplicate_hashes"]:
        report["global_failures"].append("byte-identical MP4 duplicates found")
    report["summary"] = {"case_failures": failures, "global_failures": len(report["global_failures"]),
                         "audited_cases": len(report["cases"]), "pass": failures == 0 and not report["global_failures"]}
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))
    raise SystemExit(0 if report["summary"]["pass"] else 1)


if __name__ == "__main__":
    main()
