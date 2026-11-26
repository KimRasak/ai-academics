#!/usr/bin/env python3
"""
Exp 25B (P2 Pre-Trained Backbone Pilot — 1-Step Animagine-XL-4.0 UNet LoRA Cel-Shadow Delighter!):

Directly tests Exp 12's open scientific question:
  Exp 12 proved a from-scratch 2.1M U-Net gets only +0.09 dB PSNR on pre-lit anime cel-shadow removal.
  Can a lightweight rank-16 LoRA fine-tuned on the PRE-TRAINED `Animagine-XL-4.0` UNet (1-step latent
  residual prediction: `z_albedo_hat = z_prelit - unet_lora(z_prelit, t=0)`) actually erase strong
  baked 2D cel shadows on BOTH held-out synthetic pre-lit characters AND real native pre-lit portraits
  (`pl_00..pl_03` from Exp 3D)?!

Uses our 48 clean full-color TV-anime characters (40 train / 8 held-out test) with our exact
Depth-Guided Normal-Flattened Cel Relighter (Exp 3B) generating paired (Pre-Lit, Flat Albedo)
training data on the fly (40 chars x 4 light directions = 160 pairs, 320 steps on 1 H100 ~90s!).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import DiffusionPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp3b_p2_cel_relight_opt as e3b


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scale = alpha / float(r)
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device, dtype=torch.float32) * (1.0 / math.sqrt(r)))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dx = (x.float() @ self.A.t() @ self.B.t()) * self.scale
        return out + dx.to(out.dtype)


def inject_unet_lora(unet: nn.Module, r: int = 16) -> list[nn.Parameter]:
    for p in unet.parameters():
        p.requires_grad = False
    params = []
    targets = ("to_q", "to_k", "to_v", "to_out.0")
    for name, mod in list(unet.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in targets):
                w = LoRALinear(child, r=r, alpha=float(r * 2))
                setattr(mod, child_name, w)
                params.extend([w.A, w.B])
    return params


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 10.0 * math.log10((255.0 ** 2) / max(mse, 1e-6))


def _l_minus_r(img: Image.Image) -> float:
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    face_l = luma[int(h * 0.25):int(h * 0.72), int(w * 0.25):int(w * 0.50)]
    face_r = luma[int(h * 0.25):int(h * 0.72), int(w * 0.50):int(w * 0.75)]
    return float(face_l.mean() - face_r.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest48", type=Path, required=True)
    ap.add_argument("--prelit-dir", type=Path, default=Path("/gluster_hdd_cv/user/jinzili/research_2026_anime/outputs/exp3d_prelit"))
    ap.add_argument("--sdxl-dir", type=str, default="/gluster_hdd_cv/user/jinzili/weights/animagine-xl-4.0")
    ap.add_argument("--da2-dir", type=str, default="depth-anything/Depth-Anything-V2-Large-hf")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=320)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in args.manifest48.read_text(encoding="utf-8").splitlines() if l.strip()]
    orig48 = [r for r in recs if r["variant"] == "orig"]
    tr_recs, te_recs = orig48[:40], orig48[40:]

    # 1. Precompute Depth maps for all 48 chars using DA2-Large, then free DA2
    print("Step 1: Computing DA2-Large depth maps for 48 anime chars...", flush=True)
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    d_proc = AutoImageProcessor.from_pretrained(args.da2_dir)
    d_mod = AutoModelForDepthEstimation.from_pretrained(args.da2_dir).to("cuda").eval()
    depths = {}
    with torch.no_grad():
        for r in orig48:
            im = Image.open(r["image_path"]).convert("RGB").resize((512, 512), Image.LANCZOS)
            inp = d_proc(images=im, return_tensors="pt").to("cuda")
            pred = d_mod(**inp).predicted_depth
            pred = F.interpolate(pred.unsqueeze(1), size=(512, 512), mode="bicubic", align_corners=False)[0, 0]
            d_np = pred.float().cpu().numpy()
            d_np = (d_np - d_np.min()) / max(d_np.max() - d_np.min(), 1e-6)
            depths[r["char_id"]] = d_np
    del d_mod, d_proc
    torch.cuda.empty_cache()

    # 2. Load pre-trained Animagine-XL-4.0 VAE + UNet + fixed empty prompt embeddings
    print("Step 2: Loading pre-trained Animagine-XL-4.0 VAE + UNet...", flush=True)
    pipe = DiffusionPipeline.from_pretrained(args.sdxl_dir, torch_dtype=torch.bfloat16, use_safetensors=True).to("cuda")
    vae = pipe.vae.eval()
    unet = pipe.unet
    with torch.no_grad():
        prompt_embeds, _, pooled_embeds, _ = pipe.encode_prompt(
            prompt="anime flat even studio lighting, clean albedo, no shadow",
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    add_time_ids = torch.tensor([[512.0, 512.0, 0.0, 0.0, 512.0, 512.0]], device="cuda", dtype=torch.bfloat16)
    added_cond = {"text_embeds": pooled_embeds.to(torch.bfloat16), "time_ids": add_time_ids}
    # Free text encoders
    del pipe.text_encoder, pipe.text_encoder_2
    torch.cuda.empty_cache()

    def img_to_lat(im: Image.Image) -> torch.Tensor:
        t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            return vae.encode(t).latent_dist.mean * vae.config.scaling_factor

    def lat_to_img(z: torch.Tensor) -> Image.Image:
        with torch.no_grad():
            x = vae.decode(z.to(torch.bfloat16) / vae.config.scaling_factor).sample
        arr = ((x[0].float().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    # 3. Pre-encode paired (Pre-Lit, Flat Albedo) latents for 40 train chars x 4 light angles = 160 pairs + 8 test chars
    light_dirs = [
        np.array([-0.85, 0.25, 0.45], dtype=np.float32),
        np.array([0.85, 0.25, 0.45], dtype=np.float32),
        np.array([-0.70, 0.55, 0.45], dtype=np.float32),
        np.array([0.70, 0.55, 0.45], dtype=np.float32),
    ]
    tr_pairs, te_pairs = [], []
    for split, dst in [(tr_recs, tr_pairs), (te_recs, te_pairs)]:
        for r in split:
            alb_im = Image.open(r["image_path"]).convert("RGB").resize((512, 512), Image.LANCZOS)
            z_alb = img_to_lat(alb_im)
            d_np = depths[r["char_id"]]
            for ld in light_dirs[: (4 if dst is tr_pairs else 2)]:
                ld_n = ld / np.linalg.norm(ld)
                pre_im = e3b.depth_guided_cel_relight(alb_im, d_np, light_dir=tuple(ld_n.tolist()), flatten_alpha=0.82)
                z_pre = img_to_lat(pre_im)
                dst.append((z_pre, z_alb, pre_im, alb_im, r["char_id"]))
    print(f"Encoded {len(tr_pairs)} train pairs and {len(te_pairs)} held-out test pairs!", flush=True)

    # 4. Inject Rank-16 LoRA into pre-trained Animagine-XL UNet & train 1-step residual delighter!
    lora_params = inject_unet_lora(unet, r=16)
    print(f"Injected pre-trained SDXL UNet rank-16 LoRA: {sum(p.numel() for p in lora_params):,} trainable params", flush=True)
    unet.train()
    opt = torch.optim.AdamW(lora_params, lr=2e-4, weight_decay=1e-4)
    t_zero = torch.tensor([1], device="cuda", dtype=torch.long)
    rng = random.Random(42)

    for step in range(1, args.steps + 1):
        z_pre, z_alb, _, _, _ = rng.choice(tr_pairs)
        pred_delta = unet(z_pre, t_zero, encoder_hidden_states=prompt_embeds.to(torch.bfloat16), added_cond_kwargs=added_cond).sample
        z_hat = z_pre - pred_delta
        loss = F.l1_loss(z_hat.float(), z_alb.float()) + 0.5 * F.mse_loss(z_hat.float(), z_alb.float())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        if step % 80 == 0:
            print(f"  step {step}/{args.steps} loss={float(loss.item()):.4f}", flush=True)

    # 5. Evaluate on held-out 8 characters (16 pairs) AND real native pre-lit portraits (pl_00..pl_03)!
    unet.eval()
    in_psnrs, out_psnrs, vae_ceils = [], [], []
    sheets = []
    with torch.no_grad():
        for idx, (z_pre, z_alb, pre_im, alb_im, cid) in enumerate(te_pairs):
            pred_delta = unet(z_pre, t_zero, encoder_hidden_states=prompt_embeds.to(torch.bfloat16), added_cond_kwargs=added_cond).sample
            delit_im = lat_to_img(z_pre - pred_delta)
            vae_im = lat_to_img(z_alb)
            alb_a = np.asarray(alb_im)
            in_psnrs.append(_psnr(np.asarray(pre_im), alb_a))
            out_psnrs.append(_psnr(np.asarray(delit_im), alb_a))
            vae_ceils.append(_psnr(np.asarray(vae_im), alb_a))
            if idx < 4:
                row = Image.new("RGB", (512 * 3, 512))
                row.paste(pre_im, (0, 0)); row.paste(delit_im, (512, 0)); row.paste(alb_im, (1024, 0))
                row.save(args.out_dir / f"test_delit_{idx}_{cid}.png")

        # Real native pre-lit portraits from Exp 3D!
        real_pl = sorted(args.prelit_dir.glob("pl_*.png"))
        real_in_lr, real_out_lr = [], []
        for idx, rp in enumerate(real_pl):
            im = Image.open(rp).convert("RGB").resize((512, 512), Image.LANCZOS)
            z = img_to_lat(im)
            pred_delta = unet(z, t_zero, encoder_hidden_states=prompt_embeds.to(torch.bfloat16), added_cond_kwargs=added_cond).sample
            delit = lat_to_img(z - pred_delta)
            real_in_lr.append(_l_minus_r(im))
            real_out_lr.append(_l_minus_r(delit))
            row = Image.new("RGB", (1024, 512))
            row.paste(im, (0, 0)); row.paste(delit, (512, 0))
            row.save(args.out_dir / f"real_prelit_delit_{rp.stem}.png")

    res = {
        "train_pairs": len(tr_pairs),
        "test_pairs": len(te_pairs),
        "steps": args.steps,
        "heldout_input_psnr": round(float(np.mean(in_psnrs)), 3),
        "heldout_delit_psnr": round(float(np.mean(out_psnrs)), 3),
        "heldout_psnr_gain": round(float(np.mean(out_psnrs) - np.mean(in_psnrs)), 3),
        "vae_roundtrip_ceiling_psnr": round(float(np.mean(vae_ceils)), 3),
        "real_prelit_n": len(real_pl),
        "real_prelit_orig_L_minus_R": round(float(np.mean(real_in_lr)), 3) if real_in_lr else None,
        "real_prelit_delit_L_minus_R": round(float(np.mean(real_out_lr)), 3) if real_out_lr else None,
    }
    (args.out_dir / "exp25b_p2_sdxl_delighter.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n=== EXP 25B PRE-TRAINED ANIMAGINE-XL 1-STEP LORA DELIGHTER ===", flush=True)
    print(json.dumps(res, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
