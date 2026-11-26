#!/usr/bin/env python3
"""M3: Emotion-faithful LoRA SFT for EchoMimic on CREMA-D.
Usage: CUDA_VISIBLE_DEVICES=5 python train_emolora.py --steps 8000 --out /ssd2/jinzili/avdata/lora_vA
Mirrors pipeline_echo_mimic conditioning: ref unet (writer) -> denoising unet (reader),
audio_cond_fea = per-frame whisper chunks, face_musk_fea = face_locator(mask)."""
import argparse, json, math, os, random, sys, subprocess
from pathlib import Path

sys.path.insert(0, '/ssd2/jinzili/echomimic')
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.whisper.audio2feature import load_audio_model
from src.models.unet_2d_condition import UNet2DConditionModel  # EchoMimic fork (no attn2)
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.face_locator import FaceLocator
import cv2

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--frames', type=int, default=12)
    ap.add_argument('--H', type=int, default=256)
    ap.add_argument('--W', type=int, default=256)
    ap.add_argument('--out', type=str, default='/ssd2/jinzili/avdata/lora_vA')
    ap.add_argument('--r', type=int, default=32)
    ap.add_argument('--save_every', type=int, default=2000)
    ap.add_argument('--val_actors', type=str, default='1031,1041')
    ap.add_argument('--variant', type=str, default='VA', choices=['VA','VB','VC'])
    ap.add_argument('--alpha', type=int, default=64)
    ap.add_argument('--init_lora', type=str, default=None, help='load existing lora dir to continue training')
    ap.add_argument('--gen_pairs', type=str, default=None, help='json {winner_video: audio_wav} for RAFT training')
    ap.add_argument('--ref_img', type=str, default='/ssd2/jinzili/echomimic/assets/test_imgs/a.png')
    return ap.parse_args()

CFG = OmegaConf.load('/ssd2/jinzili/echomimic/configs/inference/inference_v2.yaml')
UNetKW = OmegaConf.load('/ssd2/jinzili/echomimic/configs/inference/inference_v2.yaml').unet_additional_kwargs

def build_models(device, dtype):
    vae = AutoencoderKL.from_pretrained('/ssd2/jinzili/av_weights/echomimic/sd-vae-ft-mse').to(device, dtype=dtype)
    reference_unet = UNet2DConditionModel.from_pretrained(
        '/ssd2/jinzili/av_weights/echomimic/sd-image-variations-diffusers', subfolder='unet').to(device, dtype=dtype)
    reference_unet.load_state_dict(torch.load('/ssd2/jinzili/av_weights/echomimic/reference_unet.pth', map_location='cpu'))
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        '/ssd2/jinzili/av_weights/echomimic/sd-image-variations-diffusers',
        '/ssd2/jinzili/av_weights/echomimic/motion_module.pth',
        subfolder='unet', unet_additional_kwargs=UNetKW).to(device, dtype=dtype)
    denoising_unet.load_state_dict(
        torch.load('/ssd2/jinzili/av_weights/echomimic/denoising_unet.pth', map_location='cpu'),
        strict=False)
    face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(device, dtype=dtype)
    face_locator.load_state_dict(torch.load('/ssd2/jinzili/av_weights/echomimic/face_locator.pth', map_location='cpu'))
    audio_guider = load_audio_model('/ssd2/jinzili/av_weights/echomimic/audio_processor/whisper_tiny.pt', device)
    scheduler = DDIMScheduler.from_pretrained('/ssd2/jinzili/av_weights/echomimic/sd-image-variations-diffusers', subfolder='scheduler')
    for m in (vae, face_locator):
        m.requires_grad_(False)
    if hasattr(audio_guider, 'requires_grad_'):
        audio_guider.requires_grad_(False)
    return vae, reference_unet, denoising_unet, face_locator, audio_guider, scheduler

def add_lora(unet, r=32, alpha=64):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05,
                     target_modules=['to_q', 'to_k', 'to_v'])
    return get_peft_model(unet, cfg)

def rect_mask(H, W, device, dtype):
    m = np.zeros((H, W), np.uint8)
    m[int(0.12*H):int(0.88*H), int(0.22*W):int(0.78*W)] = 255
    t = torch.from_numpy(m).to(device, dtype=dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0
    return t  # (1,1,1,H,W)

def load_clip(video_path, n_frames, H, W, fps=24):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < n_frames + 2:
        cap.release(); return None, None
    start = random.randint(0, total - n_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, ref = cap.read()
    if not ok:
        cap.release(); return None, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for i in range(n_frames):
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (W, H))
        frames.append(f)
    cap.release()
    if len(frames) < n_frames:
        return None, None
    ref = cv2.resize(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), (W, H))
    vid = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 127.5 - 1.0  # (C,T,H,W)
    ref = torch.from_numpy(ref).permute(2, 0, 1).float() / 127.5 - 1.0  # (C,H,W)
    return vid, ref

def load_clip_gen(video_path, ref_img_path, n_frames, H, W, fps=24):
    ref = cv2.cvtColor(cv2.resize(cv2.imread(ref_img_path), (W, H)), cv2.COLOR_BGR2RGB)
    ref = torch.from_numpy(ref).permute(2, 0, 1).float() / 127.5 - 1.0
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < n_frames:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (W, H)))
    cap.release()
    if len(frames) < n_frames:
        return None, None
    vid = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    return vid, ref

def audio_chunks(audio_guider, audio_path, n_frames, fps=24, device='cuda'):
    feat = audio_guider.audio2feat(str(audio_path))
    chunks = audio_guider.feature2chunks(feature_array=feat, fps=fps)
    if len(chunks) < n_frames:
        chunks = np.concatenate([chunks] * (n_frames // max(1, len(chunks)) + 1), 0)[:n_frames]
    else:
        s = random.randint(0, len(chunks) - n_frames)
        chunks = chunks[s:s + n_frames]
    return torch.Tensor(chunks).to(device)  # (T, D, d)

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parse()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda'
    dtype = torch.float32
    gen_pairs = None
    if args.gen_pairs:
        gen_pairs = json.load(open(args.gen_pairs))
        print(f'RAFT mode: {len(gen_pairs)} winner pairs')
    else:
        manifest = json.load(open('/ssd2/jinzili/avdata/m3_manifest.json'))['rows']
        val_actors = set(int(a) for a in args.val_actors.split(','))
        train_rows = [r for r in manifest if r['actor'] not in val_actors]
        print(f'train clips: {len(train_rows)} (val actors {val_actors} held out)')

    vae, reference_unet, denoising_unet, face_locator, audio_guider, scheduler = build_models(device, dtype)
    vae.eval()
    denoising_unet = add_lora(denoising_unet, r=args.r, alpha=args.alpha)
    reference_unet = add_lora(reference_unet, r=args.r, alpha=args.alpha)
    if args.init_lora:
        for nm, m in (('denoising', denoising_unet), ('reference', reference_unet)):
            sd = torch.load(os.path.join(args.init_lora, f'{nm}_lora.pt'), map_location='cpu')
            m.load_state_dict(sd, strict=False)
        print(f'initialized LoRA from {args.init_lora}')
    train_params = [p for m in (denoising_unet, reference_unet)
                    for p in m.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in train_params)
    print(f'LoRA trainable params: {n_trainable/1e6:.1f}M')
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-2)
    scheduler.set_timesteps(1000, device=device)
    timesteps_all = scheduler.timesteps

    ref_ctrl_w = ReferenceAttentionControl(reference_unet, do_classifier_free_guidance=False,
                                           mode='write', fusion_blocks='midup')
    ref_ctrl_r = ReferenceAttentionControl(denoising_unet, do_classifier_free_guidance=False,
                                           mode='read', fusion_blocks='midup')
    emo_embed = None
    if args.variant in ('VB', 'VC'):
        emo_embed = torch.nn.Embedding(12, 768).to(device)
        torch.nn.init.normal_(emo_embed.weight, std=0.02)
    EMO_ID = {'neutral':0,'calm':1,'happy':2,'sad':3,'angry':4,'fearful':5,'fear':5,'disgust':6,'surprised':7,
              'anxious':8}
    AROUSAL_W = {'neutral':0.5,'calm':0.5,'happy':1.5,'sad':1.2,'angry':1.5,'fearful':1.5,'fear':1.5,
                 'disgust':1.2,'surprised':1.5,'anxious':1.5}
    mask = rect_mask(args.H, args.W, device, dtype)
    face_fea = face_locator(mask)  # (1,320,1,H,W)
    audio_cache = {}

    step = 0
    if args.init_lora:
        try:
            step = int(Path(args.init_lora).name.replace('step', ''))
            print(f'resuming step counter from {step}', flush=True)
        except ValueError:
            step = 0
    accum_loss, report = 0.0, [0.0] * min(step, 2000)
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        for k in range(args.accum):
            try:
                if gen_pairs is not None:
                    vfile = random.choice(list(gen_pairs.keys()))
                    row = {'video': vfile, 'audio': gen_pairs[vfile], 'emotion': Path(vfile).stem.split('_')[2]}
                    vid, ref = load_clip_gen(row['video'], args.ref_img, args.frames, args.H, args.W)
                else:
                    row = random.choice(train_rows)
                    vid, ref = load_clip(row['video'], args.frames, args.H, args.W)
                if vid is None:
                    continue
                vid = vid.to(device)
                ref = ref.to(device)
                key = row['audio']
                if key not in audio_cache:
                    try:
                        audio_cache[key] = audio_guider.audio2feat(key)
                    except Exception:
                        audio_cache[key] = None
                feat = audio_cache[key]
                if feat is None:
                    continue
                chunks = audio_guider.feature2chunks(feature_array=feat, fps=24)
                if len(chunks) < args.frames:
                    continue
                si = random.randint(0, len(chunks) - args.frames)
                audio_cond = torch.Tensor(chunks[si:si+args.frames]).to(device)
                if random.random() < 0.1:  # CFG dropout
                    audio_cond = torch.zeros_like(audio_cond)

                with torch.no_grad():
                    ref_lat = vae.encode(ref.unsqueeze(0)).latent_dist.mean * 0.18215  # (1,4,h,w)
                    fr = vid.permute(1, 0, 2, 3)  # (T,C,H,W)
                    lat_list = []
                    for i in range(0, args.frames, 4):
                        lat_list.append(vae.encode(fr[i:i+4]).latent_dist.mean * 0.18215)
                    lat = torch.cat(lat_list, 0).unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1,4,T,h,w)

                noise = torch.randn_like(lat)
                t = timesteps_all[torch.randint(0, len(timesteps_all), (1,))].to(device)

                ref_ctrl_r.clear(); ref_ctrl_w.clear()
                ref_out = reference_unet(ref_lat, torch.zeros_like(t),
                                         encoder_hidden_states=None, return_dict=False)
                ref_ctrl_r.update(ref_ctrl_w, do_classifier_free_guidance=False, dtype=lat.dtype)
                enc_hs = None
                if emo_embed is not None:
                    eid = EMO_ID.get(row['emotion'], 0)
                    enc_hs = emo_embed(torch.tensor([eid], device=device)).unsqueeze(1)  # (1,1,768)
                noisy_lat = scheduler.add_noise(lat, noise, t)
                pred = denoising_unet(noisy_lat, t, encoder_hidden_states=enc_hs,
                                      audio_cond_fea=audio_cond.unsqueeze(0),
                                      face_musk_fea=face_fea, return_dict=False)[0]
                ref_ctrl_r.clear(); ref_ctrl_w.clear()
                if args.variant == 'VC':
                    w = AROUSAL_W.get(row['emotion'], 1.0)
                    loss = F.mse_loss(pred.float(), noise.float(), reduction='mean') * w / args.accum
                else:
                    loss = F.mse_loss(pred.float(), noise.float(), reduction='mean') / args.accum
                if not torch.isfinite(loss):
                    print('non-finite loss, batch skipped', flush=True)
                    ref_ctrl_r.clear(); ref_ctrl_w.clear()
                    continue
                loss.backward()
                accum_loss += loss.item()
            except RuntimeError as e:
                print('batch skipped:', str(e)[:90], flush=True)
                torch.cuda.empty_cache()
                continue
        grads_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in train_params)
        if not grads_finite:
            print('non-finite grads, step skipped', flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
        step += 1
        report.append(accum_loss)
        if step % 50 == 0:
            print(f'step {step} loss {accum_loss:.4f} | {row["emotion"]} | {args.variant}', flush=True)
        if step % args.save_every == 0 or step == args.steps:
            ddir = Path(args.out) / f'step{step}'
            ddir.mkdir(parents=True, exist_ok=True)
            if emo_embed is not None:
                torch.save(emo_embed.state_dict(), ddir / 'emo_embed.pt')
            for name, m in (('denoising', denoising_unet), ('reference', reference_unet)):
                sd = {k: v for k, v in m.state_dict().items() if 'lora' in k}
                torch.save(sd, ddir / f'{name}_lora.pt')
            json.dump({'loss_curve': report}, open(ddir / 'loss.json', 'w'))
            print(f'saved {ddir}', flush=True)
        accum_loss = 0.0

if __name__ == '__main__':
    main()
