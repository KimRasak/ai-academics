#!/usr/bin/env python3
"""
Exp 39v2 (Clean Foreground-Masked Gloom + Specular-Only Eye Highlight Extinction Training Cache):

Directly fixes the TWO visual defects discovered during Exp 40 pixel inspection (`cmp40_03_G1`, `cmp40_03_G2`):
  1. Foreground-clipped forehead gloom: uses border-connected background flood-fill (`_foreground_mask`) so the
     translucent indigo anime gloom shadow + vertical hatching NEVER spills onto the white studio background!
  2. Specular-only eye highlight removal (`_clean_specular_off`): extinguishes ONLY the small bright white
     specular highlight dots (`luma > 0.72` / high-contrast peaks inside the eye diff mask) while gently
     matting iris contrast (`0.82x`) — preserving iris color & white sclera so extreme close-up eyes (e.g. `fc_03`)
     NEVER collapse into pitch-black voids!

Symlinks unchanged `anger_vein_on`, `sweat_drop_on`, and 3D-motion `anchor_*` `.pt` files from Exp 39,
and re-encodes ONLY the 96 cleaned (`eye_highlight_off`, `forehead_shadow_on`, `dead_eyes_and_shadow`) clips!
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp31b_video_dit_anime_vs_real_diagnostic as e31b
import exp39_build_2d_grammar_training_cache as e39


def _foreground_mask(orig_rgb: np.ndarray, tol: float = 20.0) -> np.ndarray:
    """Border-connected flood-fill from image edges to separate bright studio background (0) from character foreground (1)."""
    H, W, _ = orig_rgb.shape
    # Estimate studio background RGB from the 4 corners (median)
    corners = np.concatenate([
        orig_rgb[:12, :12].reshape(-1, 3),
        orig_rgb[:12, -12:].reshape(-1, 3),
        orig_rgb[-12:, :12].reshape(-1, 3),
        orig_rgb[-12:, -12:].reshape(-1, 3),
    ], axis=0)
    bg_rgb = np.median(corners, axis=0)
    is_bg_cand = np.linalg.norm(orig_rgb - bg_rgb[None, None, :], axis=-1) < tol

    # Downsample 2x for instant BFS flood-fill from outer border
    small = is_bg_cand[::2, ::2]
    sh, sw = small.shape
    visited = np.zeros((sh, sw), dtype=np.bool_)
    q = deque()
    for x in range(sw):
        if small[0, x]:
            visited[0, x] = True
            q.append((0, x))
        if small[sh - 1, x]:
            visited[sh - 1, x] = True
            q.append((sh - 1, x))
    for y in range(sh):
        if small[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            q.append((y, 0))
        if small[y, sw - 1] and not visited[y, sw - 1]:
            visited[y, sw - 1] = True
            q.append((y, sw - 1))
    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < sh and 0 <= nx < sw and small[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    fg_small = (~visited).astype(np.float32)
    fg_pil = Image.fromarray((fg_small * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    fg_pil = fg_pil.filter(ImageFilter.GaussianBlur(radius=1.5))
    return np.clip(np.asarray(fg_pil, dtype=np.float32) / 255.0, 0.0, 1.0)


def _clean_eye_highlight_off(orig_rgb: np.ndarray, raw_var_rgb: np.ndarray) -> np.ndarray:
    """
    Replace ONLY bright white specular highlight dots inside the eye diff mask with natural matte iris color,
    preserving colorful irises & white sclera (NEVER turning close-up eyes into pitch-black voids!).
    """
    diff = np.abs(raw_var_rgb - orig_rgb).max(axis=-1)  # [H, W]
    eye_roi = diff > 10.0
    if eye_roi.sum() < 10:
        return raw_var_rgb.copy()

    # Compute pixel luma and saturation on orig_rgb
    o01 = orig_rgb / 255.0
    luma = 0.299 * o01[..., 0] + 0.587 * o01[..., 1] + 0.114 * o01[..., 2]
    cmax = o01.max(axis=-1)
    cmin = o01.min(axis=-1)
    sat = (cmax - cmin) / np.maximum(cmax, 1e-4)

    # Compute local iris median color inside the eye ROI (excluding extreme bright white highlights)
    iris_cand = eye_roi & (luma > 0.14) & (luma < 0.72)
    if iris_cand.sum() >= 15:
        matte_iris = np.median(orig_rgb[iris_cand], axis=0) * 0.72
    else:
        matte_iris = np.array([62.0, 66.0, 78.0], dtype=np.float32)

    out = orig_rgb.copy()
    # 1. Specular highlight dots inside the edited eye region: bright & low-sat
    spec_mask = eye_roi & (luma > 0.74) & (sat < 0.34)
    # Soften specular mask edges slightly
    spec_f = np.asarray(
        Image.fromarray((spec_mask.astype(np.uint8) * 255)).filter(ImageFilter.GaussianBlur(radius=1.0)),
        dtype=np.float32,
    ) / 255.0

    # 2. Gentle matte dulling on the rest of the iris (capped so color stays readable, NEVER black void!)
    # Blend toward a floor-clamped version of raw_var_rgb (minimum 0.55x of original brightness!)
    safe_var = np.maximum(raw_var_rgb, orig_rgb * 0.55)
    # Desaturate iris slightly (20%) for authentic "hollow lifeless matte" anime look
    gray_o = (luma * 255.0)[..., None]
    matte_var = 0.80 * safe_var + 0.20 * (gray_o * 0.78)
    roi_f = np.clip((diff - 6.0) / 14.0, 0.0, 1.0)[..., None]

    out = (1.0 - roi_f) * orig_rgb + roi_f * matte_var
    # Paint over bright specular dots with matte iris color (keep tiny 0.15 blend of orig so anti-aliasing is crisp)
    sf = spec_f[..., None]
    out = (1.0 - sf) * out + sf * (0.85 * matte_iris[None, None, :] + 0.15 * out)
    return np.clip(out, 0.0, 255.0)


def _clean_forehead_shadow(orig_rgb: np.ndarray, raw_var_rgb: np.ndarray) -> np.ndarray:
    """Clip forehead gloom shadow strictly to character foreground mask so white studio background stays 100% pure!"""
    fg = _foreground_mask(orig_rgb)[..., None]
    return np.clip(fg * raw_var_rgb + (1.0 - fg) * orig_rgb, 0.0, 255.0)


def _synthesize_clean_clip(
    orig_im: Image.Image, var_im: Image.Image, variant: str, T: int = 49, size: int = 512, eye_im: Image.Image | None = None
) -> list[Image.Image]:
    o = np.asarray(orig_im.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32)
    v_raw = np.asarray(var_im.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32)

    if variant == "eye_highlight_off":
        v_clean = _clean_eye_highlight_off(o, v_raw)
        return e39._synthesize_grammar_clip(orig_im, Image.fromarray(v_clean.astype(np.uint8)), variant, T=T, size=size)

    if variant == "forehead_shadow_on":
        v_clean = _clean_forehead_shadow(o, v_raw)
        return e39._synthesize_grammar_clip(orig_im, Image.fromarray(v_clean.astype(np.uint8)), variant, T=T, size=size)

    if variant == "dead_eyes_and_shadow":
        e_raw = np.asarray(eye_im.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32) if eye_im is not None else o
        e_clean = _clean_eye_highlight_off(o, e_raw)
        # Apply foreground clipping to combined shadow+dead-eyes target, while keeping eye regions non-void!
        v_fg = _clean_forehead_shadow(o, v_raw)
        # Ensure eyes don't collapse to pitch-black void in combined target either
        eye_diff = np.abs(e_raw - o).max(axis=-1) > 10.0
        v_clean = v_fg.copy()
        v_clean[eye_diff] = np.maximum(v_fg[eye_diff], e_clean[eye_diff] * 0.72)
        return e39._synthesize_grammar_clip(
            orig_im,
            Image.fromarray(v_clean.astype(np.uint8)),
            variant,
            T=T,
            size=size,
            eye_im=Image.fromarray(e_clean.astype(np.uint8)),
        )

    raise ValueError(variant)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale48-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--src-cache-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp39_2d_grammar_train/latent_cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp39v2_clean_train"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    cache_dir = args.out_dir / "latent_cache"
    qc_dir = args.out_dir / "qc_samples"
    cache_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Shard 0 symlinks unchanged anger_vein_on, sweat_drop_on, and anchor_* .pt files immediately!
    if args.shard == 0:
        linked = 0
        for pt in sorted(args.src_cache_dir.glob("*.pt")):
            if any(k in pt.name for k in ("__anger_vein_on.pt", "__sweat_drop_on.pt", "__anchor_")):
                dst = cache_dir / pt.name
                if not dst.exists():
                    os.symlink(pt, dst)
                    linked += 1
        print(f"[exp39v2 s0] symlinked {linked} unchanged vein/sweat/anchor cache files", flush=True)

    by_char = e39._load_train_char_map(args.scale48_manifest)
    cids = sorted(by_char.keys())
    tasks = []
    for ci, cid in enumerate(cids):
        vmap = by_char[cid]
        for vi, var in enumerate(["eye_highlight_off", "forehead_shadow_on", "dead_eyes_and_shadow"]):
            if var not in vmap or "orig" not in vmap:
                continue
            prompts = e39.PROMPT_TEMPLATES[var]
            prompt = prompts[(ci + vi) % len(prompts)]
            tasks.append((cid, var, vmap["orig"], vmap[var], vmap.get("eye_highlight_off"), prompt))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (cache_dir / f"{t[0]}__{t[1]}.pt").exists()]
    if not todo:
        print(f"[exp39v2 s{args.shard}] all clean clips complete!", flush=True)
        return 0

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    pipe = WanImageToVideoPipeline.from_pretrained(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, transformer=None, transformer_2=None, torch_dtype=torch.bfloat16
    )
    pipe.text_encoder.to("cuda").eval()

    latents_mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=torch.float32).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std, device="cuda", dtype=torch.float32).view(1, vae.config.z_dim, 1, 1, 1)

    prompt_cache: dict[str, torch.Tensor] = {}
    for t in todo:
        pr = t[5]
        if pr not in prompt_cache:
            pe, _ = pipe.encode_prompt(
                prompt=pr, do_classifier_free_guidance=False, num_videos_per_prompt=1,
                device=torch.device("cuda"), dtype=torch.bfloat16,
            )
            prompt_cache[pr] = pe.cpu()

    qc_saved = 0
    for cid, var, orig_p, var_p, eye_p, prompt in todo:
        orig_im = Image.open(orig_p).convert("RGB")
        var_im = Image.open(var_p).convert("RGB")
        eye_im = Image.open(eye_p).convert("RGB") if eye_p else None
        pil_frames = _synthesize_clean_clip(orig_im, var_im, var, T=49, size=512, eye_im=eye_im)

        if args.shard == 0 and qc_saved < 6:
            e31b._save_strip(pil_frames, qc_dir / f"v2_qc_{cid}__{var}.png", f"CLEAN v2 TARGET | {cid} | {var}", n_show=6, thumb_h=170)
            qc_saved += 1

        arr = np.stack([np.asarray(f, dtype=np.float32) / 127.5 - 1.0 for f in pil_frames], axis=0)
        vid = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).to("cuda", dtype=torch.float32)
        x0_lat = (vae.encode(vid).latent_dist.mode() - latents_mean) * latents_std

        img0 = vid[:, :, :1]
        vid_cond = torch.cat([img0, torch.zeros_like(vid[:, :, 1:])], dim=2)
        y0_lat = (vae.encode(vid_cond).latent_dist.mode() - latents_mean) * latents_std

        T_f, H_lat, W_lat = 49, 64, 64
        mask = torch.ones(1, 1, T_f, H_lat, W_lat, device="cuda", dtype=torch.float32)
        mask[:, :, 1:] = 0.0
        f0_m = torch.repeat_interleave(mask[:, :, 0:1], dim=2, repeats=4)
        mask = torch.cat([f0_m, mask[:, :, 1:]], dim=2)
        mask = mask.view(1, -1, 4, H_lat, W_lat).transpose(1, 2)
        cond = torch.cat([mask, y0_lat], dim=1)

        item = {
            "char_id": cid, "variant": var, "kind": "grammar", "prompt": prompt,
            "x0": x0_lat.to(torch.bfloat16).cpu(),
            "condition": cond.to(torch.bfloat16).cpu(),
            "prompt_embeds": prompt_cache[prompt],
        }
        torch.save(item, cache_dir / f"{cid}__{var}.pt")
        print(f"[exp39v2 s{args.shard}] cached {cid}__{var}.pt", flush=True)

    if args.shard == 0:
        qcs = sorted(qc_dir.glob("v2_qc_*.png"))
        if qcs:
            e31b._stack_strips(qcs, args.out_dir / "qc39v2_clean_samples.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
