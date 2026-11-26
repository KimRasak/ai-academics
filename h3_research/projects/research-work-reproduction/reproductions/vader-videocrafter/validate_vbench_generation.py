#!/usr/bin/env python3
"""Validate a completed paired VADER/VBench generation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


CONDITIONS = ("base", "pickscore-lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, str]:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(output)["streams"][0]


def load_manifests(root: Path) -> dict[tuple[str, int], dict]:
    manifests = {}
    for condition in CONDITIONS:
        for shard in (0, 1):
            path = root / condition / f"shard-{shard}" / "project" / "prompt_manifest.json"
            manifests[(condition, shard)] = json.loads(path.read_text(encoding="utf-8"))
    return manifests


def main() -> None:
    args = parse_args()
    manifests = load_manifests(args.root)
    errors: list[str] = []
    summary: dict[str, object] = {"root": str(args.root), "conditions": {}}

    for condition in CONDITIONS:
        condition_rows = []
        condition_media = []
        for shard in (0, 1):
            manifest = manifests[(condition, shard)]
            if manifest.get("status") != "success":
                errors.append(f"{condition} shard {shard}: status={manifest.get('status')}")
            if manifest.get("prompt_count") != 36 or manifest.get("run_count") != 180:
                errors.append(f"{condition} shard {shard}: unexpected protocol cardinality")
            if manifest.get("prompts_consumed") != 180:
                errors.append(f"{condition} shard {shard}: prompts_consumed != 180")
            if len(manifest.get("rows", [])) != 180 or len(manifest.get("media", [])) != 180:
                errors.append(f"{condition} shard {shard}: rows/media length != 180")
            condition_rows.extend(manifest.get("rows", []))
            condition_media.extend(manifest.get("media", []))

        standard_dir = args.root / condition / "standard"
        standard_files = sorted(standard_dir.glob("*.mp4"))
        if len(standard_files) != 360:
            errors.append(f"{condition}: standard file count={len(standard_files)}, expected 360")

        prompt_samples = Counter(
            (row["official_prompt_index"], row["prompt"], row["sample_index"])
            for row in condition_rows
        )
        if len(prompt_samples) != 360 or any(count != 1 for count in prompt_samples.values()):
            errors.append(f"{condition}: prompt/sample keys are incomplete or duplicated")
        if Counter(row["sample_index"] for row in condition_rows) != Counter({i: 72 for i in range(5)}):
            errors.append(f"{condition}: sample indices are not balanced 72 times each")

        hashes = {}
        specifications = Counter()
        inode_mismatches = 0
        for media in condition_media:
            source = Path(media["source_path"])
            standard = Path(media["standard_path"])
            if not source.is_file() or not standard.is_file():
                errors.append(f"{condition}: missing media for run {media['run_index']}")
                continue
            actual_hash = sha256(source)
            hashes[(media["official_prompt_index"], media["sample_index"])] = actual_hash
            if actual_hash != media["sha256"] or sha256(standard) != actual_hash:
                errors.append(f"{condition}: hash mismatch for {standard.name}")
            if source.stat().st_ino != standard.stat().st_ino:
                inode_mismatches += 1
            info = probe(standard)
            specifications[
                (
                    int(info["width"]),
                    int(info["height"]),
                    int(info["nb_read_frames"]),
                    info["r_frame_rate"],
                )
            ] += 1
        if inode_mismatches:
            errors.append(f"{condition}: {inode_mismatches} standard files are not hard links")
        expected_specification = Counter({(512, 320, 24, "10/1"): 360})
        if specifications != expected_specification:
            errors.append(f"{condition}: unexpected video specifications {dict(specifications)}")
        summary["conditions"][condition] = {
            "rows": len(condition_rows),
            "media": len(condition_media),
            "standard_files": len(standard_files),
            "specifications": [
                {
                    "width": key[0],
                    "height": key[1],
                    "frames": key[2],
                    "frame_rate": key[3],
                    "count": count,
                }
                for key, count in specifications.items()
            ],
            "hashes": hashes,
        }

    for shard in (0, 1):
        base_rows = manifests[("base", shard)].get("rows", [])
        lora_rows = manifests[("pickscore-lora", shard)].get("rows", [])
        pairing_fields = ("run_index", "seed", "official_prompt_index", "prompt", "sample_index")
        if [
            tuple(row[field] for field in pairing_fields) for row in base_rows
        ] != [
            tuple(row[field] for field in pairing_fields) for row in lora_rows
        ]:
            errors.append(f"shard {shard}: base/LoRA prompt-seed pairing differs")

    base_hashes = summary["conditions"]["base"].pop("hashes")
    lora_hashes = summary["conditions"]["pickscore-lora"].pop("hashes")
    identical_pairs = sorted(key for key in base_hashes if base_hashes[key] == lora_hashes.get(key))
    if identical_pairs:
        errors.append(f"base/LoRA have {len(identical_pairs)} byte-identical paired videos")

    summary["paired_videos"] = len(set(base_hashes) & set(lora_hashes))
    summary["identical_pairs"] = identical_pairs
    summary["errors"] = errors
    summary["status"] = "success" if not errors else "failed"
    encoded = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
