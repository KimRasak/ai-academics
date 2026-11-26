#!/usr/bin/env python3
"""
Exp 39 (Self-Bootstrapped Studio-Grade 2D Anime Grammar Video Dataset & Pre-Encoded `Wan2.2-I2V-A14B` Latent Cache):

Constructs 100% clean open-weight synthetic 49-frame (`512x512`) 2D Non-Physical Anime Grammar training videos
strictly on 32 training characters (`fc_08..fc_39`), holding out 16 COMPLETELY UNSEEN characters (`fc_00..fc_07` + `fc_40..fc_47`)
for zero-shot generalization testing!

Each training character yields 5 authentic, temporally-animated 2D Anime Grammar clips (using our verified pixel-exact
`mouth_l1=0.0` variants from `exp17a_scale48`, combined with gentle living-portrait breathing motion and task-specific
2D animation curves):
  1. `eye_highlight_off`:   Smooth specular highlight extinction ONLY while eyes stay wide open & bright lighting/smile stay intact
  2. `forehead_shadow_on`:  Top-down descending translucent indigo anime gloom band + vertical hatch lines strictly on upper forehead
  3. `dead_eyes_and_shadow`:Sequenced highlight extinction + descending forehead gloom shadow while keeping calm smile & bright hair/clothes
  4. `anger_vein_on`:       Classic 2D red cross-bracket anger vein popping in with elastic overshoot + two rhythmic pulses on temple
  5. `sweat_drop_on`:       Classic 2D fixed-shape translucent blue comic sweat drop popping beside temple & sliding smoothly downward

PLUS Stage `gen_anchors` generates 16 clean base `Wan2.2-I2V-A14B` 3D-kinematics anchor clips (`S1_HeadTurn`, `S2_DynamicWind`)
on training chars `fc_08..fc_23` for zero-forgetting regularization & compositional disentanglement!

Stage `encode_cache` pre-encodes all 176 clips with `Wan2.2-I2V-A14B`'s exact `AutoencoderKLWan` (normalized `x_0` +
`condition = [mask, y_0]`) and `UMT5EncoderModel` (`prompt_embeds`), enabling ultra-fast pure-GPU 14B MoE LoRA training (~5 min!).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp31b_video_dit_anime_vs_real_diagnostic as e31b


PROMPT_TEMPLATES = {
    "eye_highlight_off": [
        "The character stays still in the exact same bright even studio lighting with unchanged bright hair, clothes, and gentle smile, while only the small white specular highlight dots inside both eyes smoothly fade away leaving flat matte dull irises, clean 2D Japanese TV anime.",
        "In bright even lighting, the character keeps their eyes open and holds a calm gentle smile while the bright white highlights inside both eyes vanish leaving flat lifeless matte irises, hair and clothes remain completely bright and unchanged, 2D anime cel shading.",
    ],
    "forehead_shadow_on": [
        "In bright even studio daylight, a stylized flat slate-blue 2D anime gloom shadow band with fine vertical ink lines appears strictly across the character's upper forehead and brow while the character keeps their exact gentle smile, bright hair color, bright cheeks, clothes, and bright white background completely unchanged.",
        "A crisp translucent indigo 2D anime forehead gloom shadow with fine vertical hatching descends cleanly across the character's upper forehead and brow while their bright hair, smiling mouth, clothing, and studio lighting stay completely unchanged, 2D Japanese TV anime.",
    ],
    "dead_eyes_and_shadow": [
        "The character's bright eye highlights suddenly vanish leaving hollow dark lifeless eyes while a sharp ominous dark shadow falls across their upper forehead, yet their mouth stays frozen in a calm eerie smile, 2D Japanese TV anime cel shading.",
        "In bright 2D Japanese TV anime style, the character's white eye highlights fade away leaving hollow matte eyes as a translucent dark gloom shadow with vertical ink lines falls across their upper forehead, while their bright hair, clothes, background, and calm smile remain completely intact.",
    ],
    "anger_vein_on": [
        "In bright flat 2D Japanese anime style, the character holds a polite closed-mouth smile as a classic stylized red cross-shaped anime anger vein symbol pops onto their upper forehead temple and pulses gently, everything else stays bright, clean, and consistent.",
        "The smiling anime character keeps bright even lighting while a classic red 2D Japanese manga anger vein mark appears on their upper temple and throbs twice with comedic tension, clean flat cel shading.",
    ],
    "sweat_drop_on": [
        "In bright flat 2D Japanese anime cel shading, the character gives an awkward nervous smile as a single large stylized translucent blue comic sweat drop appears beside their temple and slides gently downward, clean crisp lineart.",
        "The anime character holds an awkward polite smile in bright lighting as a single fixed-shape glossy blue 2D anime sweat drop pops near their temple and slides smoothly down their cheek, clean 2D TV anime.",
    ],
}

ANCHOR_PROMPTS = [
    (
        "anchor_headturn",
        "The character smoothly turns their head from front view to a three-quarter side profile while gently smiling, fluid natural motion, consistent appearance.",
    ),
    (
        "anchor_wind",
        "A sudden gust of wind blows the character's hair strongly across their face as they raise one hand to brush their bangs back with a surprised expression, dynamic lively motion.",
    ),
]


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _apply_breathing_motion(im_np: np.ndarray, t: int, T: int) -> np.ndarray:
    """Gentle 2D anime living-portrait breathing bob + subtle zoom (identical warp keeps orig & variant aligned!)."""
    H, W, _ = im_np.shape
    phase = 2.0 * math.pi * (t / max(T - 1, 1))
    dy = 2.2 * math.sin(phase)
    zoom = 1.0 + 0.014 * _smoothstep(t / max(T - 1, 1))
    pil = Image.fromarray(im_np.astype(np.uint8))
    # Affine transform mapping output (x, y) -> input (x', y') around image center
    cx, cy = W / 2.0, H / 2.0
    inv_z = 1.0 / zoom
    tx = cx * (1.0 - inv_z)
    ty = cy * (1.0 - inv_z) - dy * inv_z
    warped = pil.transform((W, H), Image.AFFINE, (inv_z, 0.0, tx, 0.0, inv_z, ty), resample=Image.BICUBIC)
    return np.asarray(warped, dtype=np.uint8)


def _synthesize_grammar_clip(orig_im: Image.Image, var_im: Image.Image, variant: str, T: int = 49, size: int = 512, eye_im: Image.Image | None = None) -> list[Image.Image]:
    """Synthesize a crisp, temporally animated 49-frame 2D Anime Grammar clip."""
    o = np.asarray(orig_im.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32)
    v = np.asarray(var_im.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32)
    H, W, _ = o.shape

    diff_mag = np.abs(v - o).max(axis=-1)  # [H, W]
    ys, xs = np.where(diff_mag > 8.0)

    frames = []
    if variant == "eye_highlight_off":
        for t in range(T):
            alpha = _smoothstep((t - 7) / 14.0)
            blended = np.clip((1.0 - alpha) * o + alpha * v, 0, 255)
            frames.append(Image.fromarray(_apply_breathing_motion(blended, t, T)))

    elif variant == "forehead_shadow_on":
        y_min = int(ys.min()) if len(ys) else int(0.15 * H)
        y_max = int(ys.max()) if len(ys) else int(0.48 * H)
        yy = np.arange(H, dtype=np.float32)[:, None]
        for t in range(T):
            prog = _smoothstep((t - 6) / 20.0)
            y_cut = y_min + prog * (y_max - y_min + 24.0)
            # Soft 18px descending curtain
            curtain = np.clip((y_cut - yy) / 18.0, 0.0, 1.0)[..., None] * min(1.0, prog * 1.4)
            blended = np.clip((1.0 - curtain) * o + curtain * v, 0, 255)
            frames.append(Image.fromarray(_apply_breathing_motion(blended, t, T)))

    elif variant == "dead_eyes_and_shadow":
        e = np.asarray(eye_im.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32) if eye_im is not None else o
        shd_diff = np.abs(v - e).max(axis=-1)
        sys_s, _ = np.where(shd_diff > 8.0)
        y_min = int(sys_s.min()) if len(sys_s) else int(0.15 * H)
        y_max = int(sys_s.max()) if len(sys_s) else int(0.48 * H)
        yy = np.arange(H, dtype=np.float32)[:, None]
        for t in range(T):
            # First extinguish eye highlights (t=5..17), then descend forehead shadow (t=13..31)
            a_eye = _smoothstep((t - 5) / 12.0)
            step1 = (1.0 - a_eye) * o + a_eye * e
            prog_s = _smoothstep((t - 13) / 18.0)
            y_cut = y_min + prog_s * (y_max - y_min + 24.0)
            curtain = np.clip((y_cut - yy) / 18.0, 0.0, 1.0)[..., None] * min(1.0, prog_s * 1.4)
            blended = np.clip((1.0 - curtain) * step1 + curtain * v, 0, 255)
            frames.append(Image.fromarray(_apply_breathing_motion(blended, t, T)))

    elif variant == "anger_vein_on":
        # Extract exact sprite bounding box and soft alpha mask
        if len(ys) > 10:
            y1, y2 = max(0, int(ys.min()) - 4), min(H, int(ys.max()) + 5)
            x1, x2 = max(0, int(xs.min()) - 4), min(W, int(xs.max()) + 5)
            cy_s, cx_s = (y1 + y2) / 2.0, (x1 + x2) / 2.0
            sp_rgb = Image.fromarray(v[y1:y2, x1:x2].astype(np.uint8))
            alpha_patch = np.clip((diff_mag[y1:y2, x1:x2] - 4.0) / 16.0, 0.0, 1.0)
            sp_a = Image.fromarray((alpha_patch * 255).astype(np.uint8))
        else:
            sp_rgb = sp_a = None

        for t in range(T):
            base = Image.fromarray(o.astype(np.uint8))
            if sp_rgb is not None and t >= 9:
                # Elastic pop-in (t=9..15) + two crisp anime pulsing beats!
                if t < 15:
                    p = (t - 9) / 6.0
                    sc = 1.22 * math.sin(p * math.pi * 0.68)
                    op = min(1.0, p * 2.0)
                else:
                    p1 = math.exp(-((t - 24.0) / 3.0) ** 2)
                    p2 = math.exp(-((t - 36.0) / 3.0) ** 2)
                    sc = 1.0 + 0.18 * (p1 + p2)
                    op = 1.0
                nw = max(2, int(round(sp_rgb.width * sc)))
                nh = max(2, int(round(sp_rgb.height * sc)))
                r_rgb = sp_rgb.resize((nw, nh), Image.BICUBIC)
                r_a = sp_a.resize((nw, nh), Image.BICUBIC)
                if op < 1.0:
                    r_a = Image.fromarray((np.asarray(r_a, dtype=np.float32) * op).astype(np.uint8))
                px = int(round(cx_s - nw / 2.0))
                py = int(round(cy_s - nh / 2.0))
                base.paste(r_rgb, (px, py), r_a)
            frames.append(Image.fromarray(_apply_breathing_motion(np.asarray(base, dtype=np.uint8), t, T)))

    elif variant == "sweat_drop_on":
        # Extract exact fixed-shape sweat drop sprite & animate clean rigid slide downward!
        if len(ys) > 10:
            y1, y2 = max(0, int(ys.min()) - 4), min(H, int(ys.max()) + 5)
            x1, x2 = max(0, int(xs.min()) - 4), min(W, int(xs.max()) + 5)
            sp_rgb = Image.fromarray(v[y1:y2, x1:x2].astype(np.uint8))
            alpha_patch = np.clip((diff_mag[y1:y2, x1:x2] - 4.0) / 16.0, 0.0, 1.0)
            sp_a = Image.fromarray((alpha_patch * 255).astype(np.uint8))
        else:
            sp_rgb = sp_a = None

        for t in range(T):
            base = Image.fromarray(o.astype(np.uint8))
            if sp_rgb is not None and t >= 8:
                if t < 13:
                    p = (t - 8) / 5.0
                    sc = 0.3 + 0.7 * _smoothstep(p)
                    op = _smoothstep(p)
                    slide_y = -10.0 * (1.0 - _smoothstep(p))
                else:
                    sc = 1.0
                    op = 1.0
                    slide_y = 26.0 * _smoothstep((t - 13) / 30.0)
                nw = max(2, int(round(sp_rgb.width * sc)))
                nh = max(2, int(round(sp_rgb.height * sc)))
                r_rgb = sp_rgb.resize((nw, nh), Image.BICUBIC)
                r_a = sp_a.resize((nw, nh), Image.BICUBIC)
                if op < 1.0:
                    r_a = Image.fromarray((np.asarray(r_a, dtype=np.float32) * op).astype(np.uint8))
                cx_s = (x1 + x2) / 2.0
                cy_s = (y1 + y2) / 2.0 + slide_y
                base.paste(r_rgb, (int(round(cx_s - nw / 2.0)), int(round(cy_s - nh / 2.0))), r_a)
            frames.append(Image.fromarray(_apply_breathing_motion(np.asarray(base, dtype=np.uint8), t, T)))
    else:
        raise ValueError(variant)

    return frames


def _load_train_char_map(manifest_path: Path) -> dict[str, dict[str, Path]]:
    recs = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_char: dict[str, dict[str, Path]] = {}
    for r in recs:
        cid = r["char_id"]
        idx = int(cid.split("_")[1])
        # Strictly training chars fc_08..fc_39 (32 chars); fc_00..fc_07 & fc_40..fc_47 are HELD-OUT UNSEEN TEST!
        if 8 <= idx <= 39:
            by_char.setdefault(cid, {})[r["variant"]] = Path(r["image_path"])
    return by_char


@torch.no_grad()
def stage_gen_anchors(args) -> int:
    """Generate 16 clean base Wan2.2-I2V-A14B 3D-motion anchor clips (fc_08..fc_23) for zero-forgetting regularization."""
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    by_char = _load_train_char_map(args.scale48_manifest)
    cids = sorted(by_char.keys())[:16]  # fc_08..fc_23
    tasks = []
    for i, cid in enumerate(cids):
        aname, aprompt = ANCHOR_PROMPTS[i % len(ANCHOR_PROMPTS)]
        tasks.append((cid, by_char[cid]["orig"], aname, aprompt, 7700 + i * 19))

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    anc_dir = args.out_dir / "anchors_frames"
    anc_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in my_tasks if not (anc_dir / f"{t[0]}__{t[2]}" / "meta.json").exists()]
    if not todo:
        print(f"[anchors s{args.shard}] all complete!", flush=True)
        return 0

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    for cid, img_path, aname, aprompt, seed in todo:
        im = Image.open(img_path).convert("RGB").resize((512, 512), Image.LANCZOS)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        frames = pipe(
            image=im,
            prompt=aprompt,
            negative_prompt="blurry, low quality, distorted, extra limbs, watermark, text",
            height=512,
            width=512,
            num_frames=49,
            num_inference_steps=32,
            guidance_scale=3.5,
            guidance_scale_2=3.5,
            generator=gen,
        ).frames[0]
        pil_frames = e31b._to_pil_frames(frames)
        d = anc_dir / f"{cid}__{aname}"
        d.mkdir(parents=True, exist_ok=True)
        for fi, fr in enumerate(pil_frames):
            fr.save(d / f"f_{fi:03d}.png")
        (d / "meta.json").write_text(json.dumps({"char_id": cid, "variant": aname, "prompt": aprompt, "is_anchor": True}), encoding="utf-8")
        print(f"[anchors s{args.shard}] saved {d.name}", flush=True)
    return 0


@torch.no_grad()
def stage_encode_cache(args) -> int:
    """Synthesize all 160 grammar videos + load 16 motion anchors, pre-encode exact Wan2.2-A14B VAE latents + UMT5 embeds."""
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    by_char = _load_train_char_map(args.scale48_manifest)
    cids = sorted(by_char.keys())
    tasks = []
    for ci, cid in enumerate(cids):
        vmap = by_char[cid]
        for vi, var in enumerate(["eye_highlight_off", "forehead_shadow_on", "dead_eyes_and_shadow", "anger_vein_on", "sweat_drop_on"]):
            if var not in vmap or "orig" not in vmap:
                continue
            prompts = PROMPT_TEMPLATES[var]
            prompt = prompts[(ci + vi) % len(prompts)]
            tasks.append(("grammar", cid, var, vmap["orig"], vmap[var], vmap.get("eye_highlight_off"), prompt))

    # Add generated base-motion anchors
    anc_dir = args.out_dir / "anchors_frames"
    if anc_dir.exists():
        for d in sorted(anc_dir.iterdir()):
            mp = d / "meta.json"
            if mp.exists():
                m = json.loads(mp.read_text(encoding="utf-8"))
                tasks.append(("anchor", m["char_id"], m["variant"], d, None, None, m["prompt"]))

    cache_dir = args.out_dir / "latent_cache"
    qc_dir = args.out_dir / "qc_samples"
    cache_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    my_tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    todo = [t for t in my_tasks if not (cache_dir / f"{t[1]}__{t[2]}.pt").exists()]
    if not todo:
        print(f"[encode s{args.shard}] all complete!", flush=True)
        return 0

    # Load ONLY VAE + text_encoder + tokenizer (no 27B transformers needed -> super fast & tiny VRAM!)
    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    pipe = WanImageToVideoPipeline.from_pretrained(
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        vae=vae,
        transformer=None,
        transformer_2=None,
        torch_dtype=torch.bfloat16,
    )
    pipe.text_encoder.to("cuda").eval()

    latents_mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=torch.float32).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std, device="cuda", dtype=torch.float32).view(1, vae.config.z_dim, 1, 1, 1)

    # Pre-encode unique prompts once
    prompt_cache: dict[str, torch.Tensor] = {}
    for t in todo:
        pr = t[6]
        if pr not in prompt_cache:
            pe, _ = pipe.encode_prompt(
                prompt=pr, do_classifier_free_guidance=False, num_videos_per_prompt=1,
                device=torch.device("cuda"), dtype=torch.bfloat16,
            )
            prompt_cache[pr] = pe.cpu()

    qc_saved = 0
    for kind, cid, var, src_a, src_b, eye_p, prompt in todo:
        if kind == "grammar":
            orig_im = Image.open(src_a).convert("RGB")
            var_im = Image.open(src_b).convert("RGB")
            eye_im = Image.open(eye_p).convert("RGB") if eye_p else None
            pil_frames = _synthesize_grammar_clip(orig_im, var_im, var, T=49, size=512, eye_im=eye_im)
        else:
            pil_frames = [Image.open(src_a / f"f_{fi:03d}.png").convert("RGB").resize((512, 512), Image.LANCZOS) for fi in range(49)]

        # Save a few visual QC strips so we visually verify training data quality!
        if args.shard == 0 and qc_saved < 6:
            e31b._save_strip(pil_frames, qc_dir / f"train_qc_{cid}__{var}.png", f"TRAIN TARGET | {cid} | {var}", n_show=6, thumb_h=170)
            qc_saved += 1

        # Exact Wan2.2-I2V-A14B tensor preprocessing: [-1, 1] float32 [1, 3, T, H, W]
        arr = np.stack([np.asarray(f, dtype=np.float32) / 127.5 - 1.0 for f in pil_frames], axis=0)  # [T, H, W, 3]
        vid = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).to("cuda", dtype=torch.float32)  # [1, 3, 49, 512, 512]
        x0_lat = vae.encode(vid).latent_dist.mode()  # [1, 16, 13, 64, 64]
        x0_lat = (x0_lat - latents_mean) * latents_std

        # Exact condition tensor matching WanImageToVideoPipeline.prepare_latents (expand_timesteps=False)
        img0 = vid[:, :, :1]
        vid_cond = torch.cat([img0, torch.zeros_like(vid[:, :, 1:])], dim=2)
        y0_lat = vae.encode(vid_cond).latent_dist.mode()
        y0_lat = (y0_lat - latents_mean) * latents_std

        T_f, H_lat, W_lat = 49, 64, 64
        mask = torch.ones(1, 1, T_f, H_lat, W_lat, device="cuda", dtype=torch.float32)
        mask[:, :, 1:] = 0.0
        f0_m = torch.repeat_interleave(mask[:, :, 0:1], dim=2, repeats=4)
        mask = torch.cat([f0_m, mask[:, :, 1:]], dim=2)
        mask = mask.view(1, -1, 4, H_lat, W_lat).transpose(1, 2)  # [1, 4, 13, 64, 64]
        cond = torch.cat([mask, y0_lat], dim=1)  # [1, 20, 13, 64, 64]

        item = {
            "char_id": cid,
            "variant": var,
            "kind": kind,
            "prompt": prompt,
            "x0": x0_lat.to(torch.bfloat16).cpu(),
            "condition": cond.to(torch.bfloat16).cpu(),
            "prompt_embeds": prompt_cache[prompt],
        }
        torch.save(item, cache_dir / f"{cid}__{var}.pt")
        print(f"[encode s{args.shard}] cached {cid}__{var}.pt | x0={tuple(item['x0'].shape)} cond={tuple(item['condition'].shape)}", flush=True)

    if args.shard == 0:
        qcs = sorted(qc_dir.glob("train_qc_*.png"))
        if qcs:
            e31b._stack_strips(qcs, args.out_dir / "qc39_train_samples.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen_anchors", "encode_cache"])
    ap.add_argument("--scale48-manifest", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp17a_scale48/manifest_scale48_9var.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp39_2d_grammar_train"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return stage_gen_anchors(args) if args.stage == "gen_anchors" else stage_encode_cache(args)


if __name__ == "__main__":
    raise SystemExit(main())
