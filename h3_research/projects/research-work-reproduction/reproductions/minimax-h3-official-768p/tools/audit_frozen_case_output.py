#!/usr/bin/env python3
"""Audit one completed H3 R0 output without asserting perceptual equivalence.

This validates execution provenance and media contract only.  It deliberately
does not use a newly chosen visual/audio similarity model or threshold: those
would be a new evaluation protocol rather than an official-example deployment
reproduction.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def script_payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"<<'JSON'\n(.*?)\nJSON", text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON heredoc missing from {path}")
    return json.loads(match.group(1))


def ffprobe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,sample_rate,channels",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("t2va", "fl2va", "ref2va"), required=True)
    parser.add_argument("--root", default="/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/minimax-h3-official-768p")
    parser.add_argument("--model-root", default="/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3")
    args = parser.parse_args()

    root = Path(args.root)
    model_root = Path(args.model_root)
    case_dir = root / "outputs" / args.case
    output = case_dir / "output.mp4"
    copied_script = case_dir / "official_request.sh"
    official_script = model_root / "scripts" / "readme" / f"reproducible-768p-{args.case}-request.sh"
    manifest = json.loads((root / "R0_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    expected = manifest["official_request_scripts"][args.case]
    media_contract = json.loads((root / "official_reference_media_audit.json").read_text(encoding="utf-8"))["expected_local_base_media"]

    problems = []
    for required in (output, copied_script, case_dir / "request.json", case_dir / "submit_response.json", case_dir / "status_history.json", case_dir / "final_status.json", case_dir / "runner_summary.json"):
        if not required.is_file():
            problems.append(f"missing:{required.name}")
    if problems:
        print(json.dumps({"case": args.case, "decision": "INVALID", "problems": problems}, indent=2))
        return 2

    official_hash = sha256(official_script)
    copied_hash = sha256(copied_script)
    request = json.loads((case_dir / "request.json").read_text(encoding="utf-8"))
    official_payload = script_payload(official_script)
    final = json.loads((case_dir / "final_status.json").read_text(encoding="utf-8"))
    streams = ffprobe(output)
    video = next((x for x in streams.get("streams", []) if x.get("codec_type") == "video"), None)
    audio = next((x for x in streams.get("streams", []) if x.get("codec_type") == "audio"), None)

    if official_hash != expected["sha256"]:
        problems.append("official_script_hash_mismatch")
    if copied_hash != expected["sha256"]:
        problems.append("copied_script_hash_mismatch")
    if request != official_payload:
        problems.append("request_payload_differs_from_frozen_official_script")
    if final.get("status") != "completed":
        problems.append(f"terminal_status={final.get('status')}")
    if not video:
        problems.append("missing_video_stream")
    elif (video.get("width"), video.get("height"), video.get("avg_frame_rate")) != (
        media_contract["width"], media_contract["height"], media_contract["video_frame_rate"],
    ):
        problems.append("video_contract_mismatch")
    if not audio:
        problems.append("missing_audio_stream")
    elif (str(audio.get("sample_rate")), audio.get("channels")) != (
        str(media_contract["audio_sample_rate_hz"]), media_contract["audio_channels"],
    ):
        problems.append("audio_contract_mismatch")

    record = {
        "case": args.case,
        "official_script_sha256": official_hash,
        "copied_script_sha256": copied_hash,
        "output_sha256": sha256(output),
        "output_ffprobe": streams,
        "request_matches_frozen_official_payload": request == official_payload,
        "final_status": final.get("status"),
        "declared_conditions": expected.get("conditions", []),
        "content_equivalence": "not_scored; no official or pre-registered perceptual threshold is claimed for stochastic deployment reproduction",
        "decision": "PASS_PROVENANCE_AND_MEDIA_CONTRACT" if not problems else "INVALID",
        "problems": problems,
    }
    (case_dir / "postrun_audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
