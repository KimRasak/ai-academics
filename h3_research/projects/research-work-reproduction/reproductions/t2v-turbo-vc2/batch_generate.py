#!/usr/bin/env python3
"""Auditable, non-GUI inference for the released T2V-Turbo VC2 LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-id", default="smoke_0000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--samples-per-prompt", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--origin-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def save_video(video: torch.Tensor, path: Path, fps: int) -> None:
    frames = video.detach().cpu().float().clamp(-1.0, 1.0)
    frames = ((frames.permute(1, 0, 2, 3) + 1.0) * 127.5)
    frames = frames.to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(
        str(path), frames, fps=fps, video_codec="h264", options={"crf": "10"}
    )


def load_jobs(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if args.prompt_file is None:
        prompt = args.prompt or "An astronaut riding a horse."
        return [
            {
                "prompt": prompt,
                "prompt_id": args.prompt_id,
                "prompt_index": 0,
                "sample_index": 0,
                "seed": args.seed,
                "filename": f"{args.prompt_id}__seed_{args.seed}.mp4",
            }
        ], {"mode": "single_prompt"}

    if args.prompt is not None:
        raise ValueError("--prompt and --prompt-file are mutually exclusive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    if args.samples_per_prompt < 1:
        raise ValueError("--samples-per-prompt must be positive")

    source_lines = [
        line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()
    ]
    all_unique_prompts = list(dict.fromkeys(source_lines))
    if args.max_prompts is not None and args.max_prompts < 1:
        raise ValueError("--max-prompts must be positive")
    prompts = (
        all_unique_prompts[: args.max_prompts]
        if args.max_prompts is not None
        else all_unique_prompts
    )
    jobs = []
    for prompt_index, prompt in enumerate(prompts):
        if prompt_index % args.num_shards != args.shard_index:
            continue
        if "/" in prompt or "\\" in prompt:
            raise ValueError(f"prompt cannot be used as a VBench filename: {prompt!r}")
        for sample_index in range(args.samples_per_prompt):
            seed = args.seed_base + prompt_index * args.samples_per_prompt + sample_index
            filename = f"{prompt}-{sample_index}.mp4"
            if len(filename.encode()) > 255:
                raise ValueError(f"VBench filename exceeds 255 bytes: {filename!r}")
            jobs.append(
                {
                    "prompt": prompt,
                    "prompt_id": f"vbench_{prompt_index:04d}",
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "seed": seed,
                    "filename": filename,
                }
            )
    metadata = {
        "mode": "vbench_standard",
        "prompt_file": str(args.prompt_file.resolve()),
        "prompt_file_sha256": sha256(args.prompt_file),
        "source_prompt_lines": len(source_lines),
        "source_unique_prompts": len(all_unique_prompts),
        "duplicate_prompt_lines": len(source_lines) - len(all_unique_prompts),
        "selected_unique_prompts": len(prompts),
        "max_prompts": args.max_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "seed_rule": "seed_base + prompt_index * samples_per_prompt + sample_index",
        "seed_base": args.seed_base,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "jobs_in_shard": len(jobs),
    }
    return jobs, metadata


def main() -> None:
    args = parse_args()
    jobs, job_metadata = load_jobs(args)
    code_root = args.code_root.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    lora_checkpoint = args.lora_checkpoint.resolve()
    runner_path = Path(__file__).resolve()
    for artifact in (base_checkpoint, lora_checkpoint):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / f"generation_config.shard_{args.shard_index:02d}.json"
    config_path.write_text(
        json.dumps(
            {
                **job_metadata,
                "artifacts": {
                    "code_root": str(code_root),
                    "code_revision": git_revision(code_root),
                    "runner_path": str(runner_path),
                    "runner_sha256": sha256(runner_path),
                    "base_checkpoint": str(base_checkpoint),
                    "base_checkpoint_bytes": base_checkpoint.stat().st_size,
                    "base_checkpoint_sha256": sha256(base_checkpoint),
                    "lora_checkpoint": str(lora_checkpoint),
                    "lora_checkpoint_bytes": lora_checkpoint.stat().st_size,
                    "lora_checkpoint_sha256": sha256(lora_checkpoint),
                },
                "generation": {
                    "width": args.width,
                    "height": args.height,
                    "frames": args.frames,
                    "fps": args.fps,
                    "inference_steps": args.steps,
                    "lcm_origin_steps": args.origin_steps,
                    "guidance_scale": args.guidance_scale,
                    "dtype": args.dtype,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    if args.plan_only:
        print(
            json.dumps(
                {
                    "status": "planned",
                    **job_metadata,
                    "config_path": str(config_path.resolve()),
                },
                ensure_ascii=False,
            )
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    sys.path.insert(0, str(code_root))
    os.chdir(code_root)

    from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline
    from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
    from utils.common_utils import load_model_checkpoint
    from utils.lora import collapse_lora, monkeypatch_remove_lora
    from utils.lora_handler import LoraHandler
    from utils.utils import instantiate_from_config

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    config = OmegaConf.load("configs/inference_t2v_512_v2.0.yaml")
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(pretrained_t2v, str(base_checkpoint))

    unet_config = model_config["params"]["unet_config"]
    unet_config["params"]["time_cond_proj_dim"] = 256
    unet = instantiate_from_config(unet_config)
    unet.load_state_dict(pretrained_t2v.model.diffusion_model.state_dict(), strict=False)

    lora_manager = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=True,
        save_for_webui=True,
        unet_replace_modules=["UNetModel"],
    )
    lora_manager.add_lora_to_model(
        True,
        unet,
        lora_manager.unet_replace_modules,
        lora_path=str(lora_checkpoint),
        dropout=0.1,
        r=64,
    )
    collapse_lora(unet, lora_manager.unet_replace_modules)
    monkeypatch_remove_lora(unet)
    unet.eval()
    # `UNetModel.dtype` is a plain attribute initialized from `use_fp16`;
    # `nn.Module.to(dtype=...)` does not update it. The current official app
    # sets this explicitly before inference.
    unet.dtype = dtype

    pretrained_t2v.model.diffusion_model = unet
    scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )
    pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)
    # The paper-day app used the older `torch_device` / `torch_dtype` keyword
    # names. Diffusers 0.30 accepts `device` / `dtype`; the old names are
    # silently ignored and leave the registered model on CPU.
    pipeline.to(device="cuda", dtype=dtype)

    manifest_name = (
        f"media_manifest.shard_{args.shard_index:02d}.jsonl"
        if args.prompt_file
        else "media_manifest.jsonl"
    )
    manifest_path = args.output_dir / manifest_name
    for job_number, job in enumerate(jobs):
        output_path = args.output_dir / job["filename"]
        if args.skip_existing and output_path.is_file() and output_path.stat().st_size:
            print(
                json.dumps(
                    {
                        "status": "skipped_existing",
                        "job_number": job_number,
                        "output_path": str(output_path),
                    }
                ),
                flush=True,
            )
            continue

        seed = job["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        started = time.time()
        result = pipeline(
            prompt=job["prompt"],
            height=args.height,
            width=args.width,
            frames=args.frames,
            fps=args.fps,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            lcm_origin_steps=args.origin_steps,
            num_videos_per_prompt=1,
        )
        elapsed = time.time() - started
        save_video(result[0], output_path, args.fps)

        record = {
            **job,
            "output_path": str(output_path.resolve()),
            "output_bytes": output_path.stat().st_size,
            "output_sha256": sha256(output_path),
            "generation": {
                "width": args.width,
                "height": args.height,
                "frames": args.frames,
                "fps": args.fps,
                "inference_steps": args.steps,
                "lcm_origin_steps": args.origin_steps,
                "guidance_scale": args.guidance_scale,
                "dtype": args.dtype,
                "rng": "python_numpy_torch_global_seed_per_video",
            },
            "runtime_seconds": elapsed,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_device": torch.cuda.get_device_name(),
            "shard_index": args.shard_index,
        }
        with manifest_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "status": "success",
                    "job_number": job_number,
                    "jobs_in_shard": len(jobs),
                    "output_path": str(output_path),
                    "runtime_seconds": elapsed,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
