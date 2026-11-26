#!/usr/bin/env python3
"""Run VADER/VideoCrafter on a pinned VBench prompt shard.

This wrapper does not modify the upstream VADER checkout. It injects a prompt
iterator at runtime because the released CLI only supports random prompt-file
sampling and its `from_str` helper has an incompatible call signature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vader-root", type=Path, required=True)
    parser.add_argument("--vbench-json", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--standard-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=["base", "pickscore-lora"], required=True)
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=5)
    parser.add_argument("--seed", type=int, default=30000)
    return parser.parse_args()


def main() -> None:
    wrapper_args = parse_args()
    if not 0 <= wrapper_args.shard_index < wrapper_args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if wrapper_args.condition == "pickscore-lora" and not wrapper_args.lora_checkpoint:
        raise ValueError("pickscore-lora requires --lora-checkpoint")

    videocrafter_root = wrapper_args.vader_root / "VADER-VideoCrafter"
    sys.path[:0] = [
        str(wrapper_args.vader_root),
        str(videocrafter_root),
        str(videocrafter_root / "scripts" / "main"),
    ]

    import train_t2v_lora  # pylint: disable=import-error,import-outside-toplevel
    from pytorch_lightning import seed_everything  # pylint: disable=import-outside-toplevel

    vbench_rows = json.loads(wrapper_args.vbench_json.read_text(encoding="utf-8"))
    dynamic_rows = [
        row for row in vbench_rows if "dynamic_degree" in row["dimension"]
    ]
    selected = [
        (official_index, row["prompt_en"])
        for official_index, row in enumerate(dynamic_rows)
        if official_index % wrapper_args.num_shards == wrapper_args.shard_index
    ]
    sequence = [
        {
            "run_index": run_index,
            "seed": wrapper_args.seed + run_index,
            "official_prompt_index": official_index,
            "prompt": prompt,
            "sample_index": sample_index,
        }
        for run_index, (official_index, prompt, sample_index) in enumerate(
            (official_index, prompt, sample_index)
            for official_index, prompt in selected
            for sample_index in range(wrapper_args.samples_per_prompt)
        )
    ]

    manifest_path = wrapper_args.project_dir / "prompt_manifest.json"
    manifest = {
        "status": "prepared",
        "condition": wrapper_args.condition,
        "vbench_json": str(wrapper_args.vbench_json),
        "vbench_json_sha256": sha256(wrapper_args.vbench_json),
        "dimension": "dynamic_degree",
        "shard_index": wrapper_args.shard_index,
        "num_shards": wrapper_args.num_shards,
        "samples_per_prompt": wrapper_args.samples_per_prompt,
        "prompt_count": len(selected),
        "run_count": len(sequence),
        "rows": sequence,
    }
    write_json(manifest_path, manifest)

    cursor = 0

    def vbench_manifest(_nouns_file=None, _activities_file=None):
        nonlocal cursor
        if cursor >= len(sequence):
            raise RuntimeError("prompt manifest exhausted")
        prompt = sequence[cursor]["prompt"]
        cursor += 1
        return prompt, {}

    train_t2v_lora.prompts_file.vbench_manifest = vbench_manifest

    argv = [
        "--seed",
        str(wrapper_args.seed),
        "--mode",
        "base",
        "--ckpt_path",
        str(wrapper_args.base_checkpoint),
        "--config",
        str(wrapper_args.config),
        "--height",
        "320",
        "--width",
        "512",
        "--unconditional_guidance_scale",
        "12.0",
        "--ddim_steps",
        "25",
        "--ddim_eta",
        "1.0",
        "--frames",
        "24",
        "--prompt_fn",
        "vbench_manifest",
        "--val_batch_size",
        "1",
        "--num_val_runs",
        str(len(sequence)),
        "--lora_rank",
        str(wrapper_args.lora_rank),
        "--inference_only",
        "true",
        "--project_dir",
        str(wrapper_args.project_dir),
        "--is_sample_preview",
        "true",
        "--use_wandb",
        "false",
    ]
    if wrapper_args.lora_checkpoint:
        argv.extend(["--lora_ckpt_path", str(wrapper_args.lora_checkpoint)])

    upstream_args = train_t2v_lora.get_parser().parse_args(argv)
    seed_everything(upstream_args.seed)

    try:
        train_t2v_lora.run_training(upstream_args)
        media = []
        wrapper_args.standard_dir.mkdir(parents=True, exist_ok=True)
        for row in sequence:
            source = (
                wrapper_args.project_dir
                / "samples"
                / "step_0"
                / f"{row['run_index']}_0_0001.mp4"
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            target = (
                wrapper_args.standard_dir
                / f"{row['prompt']}-{row['sample_index']}.mp4"
            )
            if target.exists():
                raise FileExistsError(target)
            os.link(source, target)
            media.append(
                {
                    **row,
                    "source_path": str(source),
                    "standard_path": str(target),
                    "bytes": source.stat().st_size,
                    "sha256": sha256(source),
                }
            )
        manifest["status"] = "success"
        manifest["media"] = media
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["prompts_consumed"] = cursor
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
