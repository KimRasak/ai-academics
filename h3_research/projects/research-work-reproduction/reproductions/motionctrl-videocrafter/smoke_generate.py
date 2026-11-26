#!/usr/bin/env python3
"""Generate one provenance-preserving MotionCtrl released-example smoke."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning import seed_everything

    from main.evaluation.motionctrl_inference import (
        load_camera_pose,
        motionctrl_sample,
        save_results,
    )
    from utils.utils import instantiate_from_config

    seed_everything(args.seed, workers=True)

    config_path = repo / "configs/inference/config_both.yaml"
    pose_path = repo / "examples/camera_poses/test_camera_L.json"
    prompt = "coastline, rocks, storm weather, wind, waves, lightning"

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.pop("model"))
    state = torch.load(checkpoint, map_location="cpu")["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    model = model.to(args.device).eval()

    camera_poses, _ = load_camera_pose(
        str(repo / "examples"), ["test_camera_L"]
    )
    camera = torch.stack(camera_poses).to(args.device)
    noise_shape = [1, model.channels, model.temporal_length, 32, 32]

    with torch.inference_mode():
        samples = motionctrl_sample(
            model,
            [prompt],
            noise_shape,
            camera_poses=camera,
            n_samples=1,
            unconditional_guidance_scale=7.5,
            ddim_steps=50,
            ddim_eta=1.0,
            cond_T=800,
        )

    video_dir = output_root / "videos"
    video_dir.mkdir(exist_ok=True)
    filename = "camera_L_coastline_seed1234"
    save_results(samples[0], filename, str(video_dir), fps=10)
    video_path = video_dir / f"{filename}.mp4"

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = {
        "kind": "released_example_single_condition_smoke",
        "evidence_scope": "deployment_smoke_not_paper_metric_reproduction",
        "repo_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
            "state_tensors": len(state),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        },
        "condition": {
            "type": "camera_motion",
            "name": "test_camera_L",
            "path": str(pose_path),
            "sha256": sha256(pose_path),
            "prompt": prompt,
        },
        "generation": {
            "seed": args.seed,
            "resolution": [256, 256],
            "frames": int(model.temporal_length),
            "fps": 10,
            "samples": 1,
            "ddim_steps": 50,
            "ddim_eta": 1.0,
            "guidance": 7.5,
            "cond_T": 800,
            "device": args.device,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "output": {
            "path": str(video_path),
            "bytes": video_path.stat().st_size,
            "sha256": sha256(video_path),
            "ffprobe": json.loads(probe.stdout),
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
