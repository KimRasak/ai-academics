#!/usr/bin/env python3
"""Validate a complete, sharded T2V-Turbo VBench generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import av


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=5)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    started = time.time()
    issues: list[str] = []

    source_lines = [
        line.strip()
        for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompts = list(dict.fromkeys(source_lines))
    expected: dict[str, dict] = {}
    for prompt_index, prompt in enumerate(prompts):
        for sample_index in range(args.samples_per_prompt):
            filename = f"{prompt}-{sample_index}.mp4"
            expected[filename] = {
                "prompt": prompt,
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "seed": prompt_index * args.samples_per_prompt + sample_index,
                "shard_index": prompt_index % args.num_shards,
            }

    configs = []
    records = []
    manifest_summaries = []
    for shard_index in range(args.num_shards):
        config_path = args.output_dir / f"generation_config.shard_{shard_index:02d}.json"
        manifest_path = args.output_dir / f"media_manifest.shard_{shard_index:02d}.jsonl"
        if not config_path.is_file():
            issues.append(f"missing config: {config_path}")
            continue
        if not manifest_path.is_file():
            issues.append(f"missing manifest: {manifest_path}")
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configs.append(config)
        shard_records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records.extend(shard_records)
        manifest_summaries.append(
            {
                "shard_index": shard_index,
                "path": str(manifest_path.resolve()),
                "records": len(shard_records),
                "sha256": sha256(manifest_path),
            }
        )

    expected_count = len(prompts) * args.samples_per_prompt
    if len(records) != expected_count:
        issues.append(f"manifest records: expected {expected_count}, found {len(records)}")

    records_by_filename: dict[str, dict] = {}
    duplicate_manifest_filenames = []
    for record in records:
        filename = record["filename"]
        if filename in records_by_filename:
            duplicate_manifest_filenames.append(filename)
        records_by_filename[filename] = record
    if duplicate_manifest_filenames:
        issues.append(
            f"duplicate manifest filenames: {duplicate_manifest_filenames[:10]}"
        )

    actual_mp4 = {path.name: path for path in args.output_dir.glob("*.mp4")}
    missing_files = sorted(set(expected) - set(actual_mp4))
    unexpected_files = sorted(set(actual_mp4) - set(expected))
    missing_records = sorted(set(expected) - set(records_by_filename))
    unexpected_records = sorted(set(records_by_filename) - set(expected))
    if missing_files:
        issues.append(f"missing media: {missing_files[:10]}")
    if unexpected_files:
        issues.append(f"unexpected media: {unexpected_files[:10]}")
    if missing_records:
        issues.append(f"missing manifest records: {missing_records[:10]}")
    if unexpected_records:
        issues.append(f"unexpected manifest records: {unexpected_records[:10]}")

    total_bytes = 0
    runtime_seconds = []
    codec_counts: dict[str, int] = {}
    for index, (filename, expected_row) in enumerate(expected.items(), 1):
        path = actual_mp4.get(filename)
        record = records_by_filename.get(filename)
        if path is None or record is None:
            continue
        for key, value in expected_row.items():
            if record.get(key) != value:
                issues.append(
                    f"{filename}: manifest {key}={record.get(key)!r}, expected {value!r}"
                )
        if record.get("prompt") != expected_row["prompt"]:
            issues.append(f"{filename}: prompt mismatch")
        stat_bytes = path.stat().st_size
        total_bytes += stat_bytes
        if record.get("output_bytes") != stat_bytes:
            issues.append(f"{filename}: byte count mismatch")
        actual_hash = sha256(path)
        if record.get("output_sha256") != actual_hash:
            issues.append(f"{filename}: SHA256 mismatch")
        runtime_seconds.append(record["runtime_seconds"])

        try:
            with av.open(str(path)) as container:
                video_streams = container.streams.video
                if len(video_streams) != 1:
                    issues.append(f"{filename}: video stream count {len(video_streams)}")
                    continue
                stream = video_streams[0]
                codec_name = stream.codec_context.name
                codec_counts[codec_name] = codec_counts.get(codec_name, 0) + 1
                if (stream.width, stream.height) != (512, 320):
                    issues.append(
                        f"{filename}: resolution {stream.width}x{stream.height}"
                    )
                if stream.average_rate is None or float(stream.average_rate) != 16.0:
                    issues.append(f"{filename}: FPS {stream.average_rate}")
                decoded_frames = sum(1 for _ in container.decode(video=0))
                if decoded_frames != 16:
                    issues.append(f"{filename}: decoded frames {decoded_frames}")
        except Exception as exc:  # noqa: BLE001 - validator must retain the failure.
            issues.append(f"{filename}: decode failed: {type(exc).__name__}: {exc}")

        if index % 250 == 0:
            print(
                json.dumps(
                    {
                        "status": "validating",
                        "checked": index,
                        "expected": expected_count,
                        "issues": len(issues),
                    }
                ),
                flush=True,
            )

    shared_artifact_fields = [
        "code_revision",
        "runner_sha256",
        "base_checkpoint_sha256",
        "lora_checkpoint_sha256",
    ]
    if len(configs) == args.num_shards:
        reference_artifacts = configs[0]["artifacts"]
        for shard_index, config in enumerate(configs):
            if config.get("jobs_in_shard") != expected_count // args.num_shards:
                issues.append(
                    f"config shard {shard_index}: jobs_in_shard={config.get('jobs_in_shard')}"
                )
            if config.get("shard_index") != shard_index:
                issues.append(f"config shard {shard_index}: shard index mismatch")
            for field in shared_artifact_fields:
                if config["artifacts"].get(field) != reference_artifacts.get(field):
                    issues.append(f"config shard {shard_index}: artifact {field} mismatch")

    report = {
        "status": "success" if not issues else "failed",
        "output_dir": str(args.output_dir.resolve()),
        "prompt_file": str(args.prompt_file.resolve()),
        "prompt_file_sha256": sha256(args.prompt_file),
        "source_prompt_lines": len(source_lines),
        "unique_prompts": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "expected_videos": expected_count,
        "actual_videos": len(actual_mp4),
        "manifest_records": len(records),
        "unique_manifest_filenames": len(records_by_filename),
        "seed_range": [
            min(row["seed"] for row in expected.values()),
            max(row["seed"] for row in expected.values()),
        ],
        "total_bytes": total_bytes,
        "codec_counts": codec_counts,
        "media_contract": {
            "width": 512,
            "height": 320,
            "fps": 16,
            "decoded_frames": 16,
            "full_decode": "passed" if not any("decode" in item for item in issues) else "failed",
        },
        "runtime_seconds": {
            "minimum": min(runtime_seconds) if runtime_seconds else None,
            "maximum": max(runtime_seconds) if runtime_seconds else None,
            "mean": (
                sum(runtime_seconds) / len(runtime_seconds)
                if runtime_seconds
                else None
            ),
        },
        "manifest_summaries": manifest_summaries,
        "config_artifacts": configs[0]["artifacts"] if configs else None,
        "validation_seconds": time.time() - started,
        "issues": issues,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
