#!/usr/bin/env python
"""8xH100 FSDP2 Ref2VA generation with the trained emotion LoRA (EmoH3).
Uses the pinned diffusers (h3ft env) MiniMaxH3Ref2VAModularPipeline.

All ranks process every job in lockstep (identical geometry => aligned FSDP2
collectives); only rank 0 saves outputs.
"""
import argparse
import faulthandler
import gc
import json
import os
import signal
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np

faulthandler.enable()
WATCHDOG_SEC = 2400


def _watchdog(_signum, _frame):
    print("WATCHDOG FIRED: dumping tracebacks", flush=True)
    faulthandler.dump_traceback(file=sys.stderr)
    raise SystemExit("watchdog fired: stuck for %ds" % WATCHDOG_SEC)


def _phase(msg):
    print(f"[phase {os.environ.get('RANK','?')}] {msg}", flush=True)


def apply_fsdp2_to_transformer(transformer, mesh):
    from torch.distributed.fsdp import fully_shard

    token_refiner = transformer.token_refiner
    for block in token_refiner.refiner_blocks:
        fully_shard(block, mesh=mesh, reshard_after_forward=True)
    fully_shard(token_refiner, mesh=mesh, reshard_after_forward=True)
    for block in transformer.transformer_blocks:
        fully_shard(block, mesh=mesh, reshard_after_forward=True)
    for name in ("proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                 "norm_out", "proj_out", "audio_proj_out"):
        fully_shard(getattr(transformer, name), mesh=mesh, reshard_after_forward=True)
    fully_shard(transformer, mesh=mesh, reshard_after_forward=True)


def apply_fsdp2_to_text_encoder_model(te_model, mesh):
    from torch.distributed.fsdp import fully_shard

    language_model = te_model.language_model
    for layer in language_model.layers:
        fully_shard(layer, mesh=mesh, reshard_after_forward=True)
    fully_shard(te_model, mesh=mesh, reshard_after_forward=True)

    def reshard_conditioner_root(module, _inputs, output):
        module.reshard()
        return output

    te_model.register_forward_hook(reshard_conditioner_root)


def load_lora_into_transformer(transformer, ckpt_path, lora_alpha, lora_scale):
    from peft import LoraConfig

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    rank = next(
        v.shape[0] for k, v in sd.items() if k.endswith(".lora_A.default.weight")
    )
    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            init_lora_weights=False,
            target_modules=["to_qkv", "to_out.0", "linear_1", "linear_2"],
        )
    )
    adapter_params = {
        n: p
        for n, p in transformer.named_parameters()
        if ".lora_A." in n or ".lora_B." in n
    }
    missing = sorted(adapter_params.keys() - sd.keys())
    unexpected = sorted(sd.keys() - adapter_params.keys())
    mismatch = [
        (n, tuple(sd[n].shape), tuple(adapter_params[n].shape))
        for n in adapter_params
        if n in sd and sd[n].shape != adapter_params[n].shape
    ]
    if missing or unexpected or mismatch:
        raise RuntimeError(
            f"LoRA incompatible: missing={missing[:3]} unexpected={unexpected[:3]} "
            f"shape={mismatch[:3]}"
        )
    incompatible = transformer.load_state_dict(sd, strict=False)
    missing_lora = [
        k for k in incompatible.missing_keys if ".lora_A." in k or ".lora_B." in k
    ]
    if incompatible.unexpected_keys or missing_lora:
        raise RuntimeError(
            f"LoRA load did not consume adapters: missing={missing_lora[:3]} "
            f"unexpected={incompatible.unexpected_keys[:3]}"
        )
    transformer.set_adapters("default", weights=lora_scale)
    transformer.requires_grad_(False)
    transformer.eval()
    print(
        f"[rank {os.environ.get('RANK','?')}] LoRA loaded: tensors={len(sd)} "
        f"rank={rank} alpha={lora_alpha} scale={lora_scale}",
        flush=True,
    )


def save_video_audio(videos, audio, sr, out_path, fps=24):
    import av

    arr = videos[0] if isinstance(videos, (list, tuple)) else videos
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    print(f"[save] videos shape={arr.shape} dtype={arr.dtype}", flush=True)
    # squeeze leading singleton dims -> (T, H, W, C)
    while arr.ndim > 4:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    T, H, W, C = arr.shape
    if C != 3:
        arr = arr[..., :3]

    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    for t in range(T):
        frame = av.VideoFrame.from_ndarray(arr[t], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    if audio is not None:
        import wave

        a = audio[0] if isinstance(audio, (list, tuple)) else audio
        if isinstance(a, torch.Tensor):
            n = a.numel()
            a = a.detach().cpu().numpy()
        else:
            a = np.asarray(a)
            n = a.size
        print(f"[save] audio shape={a.shape} sr={sr}", flush=True)
        if n > 0:
            a = np.squeeze(a)
            if a.ndim > 1:
                a = a[0]  # mono
            a = (a * 32767).clip(-32768, 32767).astype(np.int16)
            wav_path = str(out_path).rsplit(".", 1)[0] + ".wav"
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(sr))
                wf.writeframes(a.tobytes())
    print(f"[rank {os.environ.get('RANK','?')}] saved {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/gluster_hdd_cv/user/jinzili/weights/MiniMax-H3")
    ap.add_argument("--lora", default="")
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--video-shift", type=float, default=12.0)
    ap.add_argument("--audio-shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--seeds",
        default="",
        help="optional comma-separated seeds; loads the model once and writes seed<N>/ subdirectories",
    )
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    signal.signal(signal.SIGALRM, _watchdog)
    signal.alarm(WATCHDOG_SEC)

    store = os.environ.get(
        "H3_STORE_FILE", os.path.join(args.outdir, "dist_store")
    )
    _phase("dist init (FileStore)")
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{store}",
        rank=rank,
        world_size=world,
    )
    from torch.distributed.device_mesh import init_device_mesh

    _phase("init_device_mesh")
    fsdp_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",))

    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3Ref2VABlocks,
        MiniMaxH3Reference,
    )

    _phase("build pipeline")
    pipe = MiniMaxH3Ref2VABlocks().init_pipeline(args.model)
    _phase("load components")
    pipe.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path=args.model,
    )
    _phase("set shifts")
    pipe.scheduler.set_shift(args.video_shift)
    pipe.audio_scheduler.set_shift(args.audio_shift)

    transformer = pipe.transformer_ref
    transformer.requires_grad_(False)
    transformer.eval()
    if args.lora:
        _phase("load lora")
        load_lora_into_transformer(
            transformer, args.lora, args.lora_alpha, args.lora_scale
        )

    _phase("fsdp2 text encoder")
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    apply_fsdp2_to_text_encoder_model(pipe.text_encoder.model, fsdp_mesh)
    _phase("fsdp2 transformer")
    apply_fsdp2_to_transformer(transformer, fsdp_mesh)

    _phase("move unsharded components")
    fsdp_ids = {id(transformer), id(pipe.text_encoder)}
    for comp in pipe.components.values():
        if isinstance(comp, torch.nn.Module) and id(comp) not in fsdp_ids:
            comp.to(device)

    print(
        f"[rank {rank}/{world}] FSDP2 ready; transformer={type(transformer).__name__} "
        f"device={device}",
        flush=True,
    )

    jobs = json.load(open(args.jobs))
    os.makedirs(args.outdir, exist_ok=True)
    tag = "lora" if args.lora else "base"
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else [args.seed]
    )
    if not seeds:
        raise ValueError("--seeds did not contain any integers")

    for seed in seeds:
        seed_outdir = (
            os.path.join(args.outdir, f"seed{seed}")
            if len(seeds) > 1
            else args.outdir
        )
        os.makedirs(seed_outdir, exist_ok=True)
        for j in jobs:
            _phase(f"generate seed={seed} {j['case_id']}")
            signal.alarm(WATCHDOG_SEC)
            refs = [MiniMaxH3Reference(image=p) for p in j["references"]]
            gen = torch.Generator().manual_seed(seed + j["index"])
            with torch.inference_mode():
                res = pipe(
                    prompt=j["prompt"],
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.steps + 1,
                    generator=gen,
                    output_type="np",
                    output=["videos", "audio", "sampling_rate"],
                    references=refs,
                )
            if rank == 0:
                out_path = os.path.join(
                    seed_outdir, f"{j['case_id']}_{tag}.mp4"
                )
                save_video_audio(
                    res["videos"], res["audio"], res["sampling_rate"], out_path
                )
            del res
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            signal.alarm(0)
            dist.barrier()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
