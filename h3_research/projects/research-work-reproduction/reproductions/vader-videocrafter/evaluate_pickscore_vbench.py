#!/usr/bin/env python3
"""Evaluate VBench-named videos with the pinned PickScore checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


PROCESSOR_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PROCESSOR_REVISION = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"
MODEL_ID = "yuvalkirstain/PickScore_v1"
MODEL_REVISION = "a4e4367c6dfa7288a00c550414478f865b875800"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def prompt_from_name(path: Path) -> tuple[str, int]:
    prompt, sample = path.stem.rsplit("-", 1)
    return prompt, int(sample)


def decode(path: Path) -> np.ndarray:
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 320, 512, 3)
    if len(frames) != 24:
        raise ValueError(f"{path}: decoded {len(frames)} frames, expected 24")
    return frames


def main() -> None:
    args = parse_args()
    videos = sorted(args.videos_dir.glob("*.mp4"))
    if len(videos) != 360:
        raise ValueError(f"expected 360 videos, found {len(videos)}")

    processor = AutoProcessor.from_pretrained(
        PROCESSOR_ID,
        revision=PROCESSOR_REVISION,
        local_files_only=True,
    )
    model = (
        AutoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        .eval()
        .to("cuda", dtype=torch.float16)
    )

    rows = []
    with torch.inference_mode():
        for index, path in enumerate(videos):
            prompt, sample_index = prompt_from_name(path)
            text_inputs = processor(
                text=[prompt],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_features = model.get_text_features(
                input_ids=text_inputs["input_ids"].to("cuda"),
                attention_mask=text_inputs["attention_mask"].to("cuda"),
            )
            text_features = text_features / torch.linalg.vector_norm(
                text_features, dim=-1, keepdim=True
            )

            frames = decode(path)
            frame_scores = []
            for start in range(0, len(frames), args.batch_size):
                images = [
                    Image.fromarray(frame)
                    for frame in frames[start : start + args.batch_size]
                ]
                image_inputs = processor(images=images, return_tensors="pt")
                image_features = model.get_image_features(
                    pixel_values=image_inputs["pixel_values"].to(
                        "cuda", dtype=torch.float16
                    )
                )
                image_features = image_features / torch.linalg.vector_norm(
                    image_features, dim=-1, keepdim=True
                )
                scores = model.logit_scale.exp() * text_features @ image_features.T
                frame_scores.extend(float(value) for value in scores[0].cpu())
            rows.append(
                {
                    "filename": path.name,
                    "path": str(path),
                    "prompt": prompt,
                    "sample_index": sample_index,
                    "frame_scores": frame_scores,
                    "video_mean": float(np.mean(frame_scores)),
                }
            )
            print(
                f"{index + 1}/360\t{rows[-1]['video_mean']:.6f}\t{path.name}",
                flush=True,
            )

    video_means = np.array([row["video_mean"] for row in rows])
    payload = {
        "protocol": {
            "condition": args.condition,
            "processor_id": PROCESSOR_ID,
            "processor_revision": PROCESSOR_REVISION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "decode": "FFmpeg raw RGB24",
            "frames_per_video": 24,
            "frame_batch_size": args.batch_size,
            "aggregation": "mean frame score per video; mean across videos",
        },
        "summary": {
            "videos": len(rows),
            "mean": float(video_means.mean()),
            "standard_deviation": float(video_means.std(ddof=1)),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
