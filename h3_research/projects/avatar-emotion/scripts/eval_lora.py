#!/usr/bin/env python3
"""M3 eval: generate videos with LoRA-merged EchoMimic on eval audios.
Usage: CUDA_VISIBLE_DEVICES=4 python eval_lora.py --lora_dir /ssd2/jinzili/avdata/lora_vA/step8000 \
  --tag vA_step8000 --audios /ssd2/jinzili/avdata/eval_audio_list.txt"""
import argparse, os, sys, subprocess
from pathlib import Path
sys.path.insert(0, '/ssd2/jinzili/echomimic')
import numpy as np
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionModel  # noqa: F401
from src.models.unet_2d_condition import UNet2DConditionModel as ForkUNet2D
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.face_locator import FaceLocator
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from facenet_pytorch import MTCNN
import cv2

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora_dir', type=str, required=True)
    ap.add_argument('--tag', type=str, required=True)
    ap.add_argument('--audios', type=str, required=True)  # txt: one wav path per line
    ap.add_argument('--ref', type=str, default='/ssd2/jinzili/echomimic/assets/test_imgs/a.png')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--seed', type=int, default=420)
    ap.add_argument('--r', type=int, default=32)
    ap.add_argument('--alpha', type=int, default=64)
    return ap.parse_args()

def main():
    args = parse()
    device = 'cuda'
    dtype = torch.float16
    cfg = OmegaConf.load('/ssd2/jinzili/echomimic/configs/prompts/census_smoke.yaml')
    infer_cfg = OmegaConf.load('/ssd2/jinzili/echomimic/configs/inference/inference_v2.yaml')

    vae = AutoencoderKL.from_pretrained('/ssd2/jinzili/av_weights/echomimic/sd-vae-ft-mse').to(device, dtype=dtype)
    reference_unet = ForkUNet2D.from_pretrained('/ssd2/jinzili/av_weights/echomimic/sd-image-variations-diffusers', subfolder='unet').to(dtype=dtype, device=device)
    reference_unet.load_state_dict(torch.load('/ssd2/jinzili/av_weights/echomimic/reference_unet.pth', map_location='cpu'), strict=False)
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        '/ssd2/jinzili/av_weights/echomimic/sd-image-variations-diffusers',
        '/ssd2/jinzili/av_weights/echomimic/motion_module.pth',
        subfolder='unet', unet_additional_kwargs=infer_cfg.unet_additional_kwargs).to(dtype=dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load('/ssd2/jinzili/av_weights/echomimic/denoising_unet.pth', map_location='cpu'),
        strict=False)
    face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(device, dtype=dtype)
    face_locator.load_state_dict(torch.load('/ssd2/jinzili/av_weights/echomimic/face_locator.pth', map_location='cpu'))
    audio_guider = load_audio_model('/ssd2/jinzili/av_weights/echomimic/audio_processor/whisper_tiny.pt', device)
    scheduler = DDIMScheduler.from_pretrained('/ssd2/jinzili/av_weights/echomimic/sd-image-variations-diffusers', subfolder='scheduler')

    # ---- merge LoRA ----
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, target_modules=['to_q','to_k','to_v'])
    for name, net in (('denoising', denoising_unet), ('reference', reference_unet)):
        lp = os.path.join(args.lora_dir, f'{name}_lora.pt')
        if os.path.exists(lp):
            sd = torch.load(lp, map_location='cpu')
            netp = get_peft_model(net, lora_cfg)
            netp.load_state_dict(sd, strict=False)
            netp.merge_and_unload()
            print(f'merged {name} LoRA ({len(sd)} tensors)', flush=True)
        else:
            print(f'no LoRA for {name} (baseline mode)', flush=True)
    # ---- end merge ----

    pipe = Audio2VideoPipeline(vae=vae, reference_unet=reference_unet, denoising_unet=denoising_unet,
                               audio_guider=audio_guider, face_locator=face_locator, scheduler=scheduler).to(device)

    def select_face(det_bboxes, probs):
        if det_bboxes is None or probs is None: return None
        bboxes = det_bboxes[probs > 0.8]
        if len(bboxes) == 0: return None
        areas = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
        return int(np.argmax(areas))

    def get_face_mask(ref, W, H, ratio=0.4):
        f = face_detector.detect(ref, landmarks=True)
        bboxes, probs, points = f[0], f[1], f[2]
        fi = select_face(bboxes, probs)
        mask = np.zeros((H, W), np.uint8)
        if fi is None:
            mask[int(0.12*H):int(0.88*H), int(0.22*W):int(0.78*W)] = 255
        else:
            x0,y0,x1,y1 = bboxes[fi]
            rw, rh = int((x1-x0)*0.52/2), int((y1-y0)*0.4/2)
            cx, cy = int((x0+x1)/2), int((y0+y1)/2 + (y1-y0)*0.1)
            mask[max(0,cy-rh):min(H,cy+rh), max(0,cx-rw):min(W,cx+rw)] = 255
        return torch.Tensor(cv2.resize(mask,(W,H))).to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)/255.0

    face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6,0.7,0.7], factor=0.709, post_process=True, device=device)
    mtcnn = face_detector
    ref_img = cv2.imread(args.ref)
    H = W = 512
    face_mask = np.zeros((ref_img.shape[0], ref_img.shape[1])).astype('uint8')
    f = face_detector.detect(ref_img, landmarks=False)
    det_bboxes, probs = f[0], f[1]
    select_bbox = None
    if det_bboxes is not None and len(det_bboxes) > 0:
        areas = (det_bboxes[:,2]-det_bboxes[:,0])*(det_bboxes[:,3]-det_bboxes[:,1])
        areas = np.asarray(areas, dtype=np.float64) * np.asarray(probs, dtype=np.float64)
        if len(areas) > 0 and np.isfinite(areas).all() and areas.max() > 0:
            select_bbox = det_bboxes[int(np.argmax(areas))]
    if select_bbox is None:
        face_mask[:, :] = 255
        face_img = ref_img
    else:
        xyxy = np.round(select_bbox[:4]).astype('int')
        rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
        r_pad = int((re - rb) * 0.15)   # facemusk_dilation_ratio
        c_pad = int((ce - cb) * 0.15)
        face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255
        r_pad_crop = int((re - rb) * 0.25)  # facecrop_dilation_ratio
        c_pad_crop = int((ce - cb) * 0.25)
        crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, ref_img.shape[1]), min(re + r_pad_crop, ref_img.shape[0])]
        x0, y0, x1, y1 = crop_rect
        face_img = ref_img[y0:y1, x0:x1]
        face_mask = face_mask[y0:y1, x0:x1]
        face_img = cv2.resize(face_img, (W, H))
        face_mask = cv2.resize(face_mask, (W, H))
    face_mask = torch.Tensor(face_mask).to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0
    ref_pil = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    for apath in audios:
        name = Path(apath).parent.name
        outf = outdir / f'{name}_seed{args.seed}.mp4'
        if outf.exists():
            continue
        g = torch.Generator(device=device); g.manual_seed(args.seed)
        video = pipe(ref_pil, apath, face_mask, W, H, args.frames, args.steps,
                     2.5, generator=g, audio_sample_rate=16000,
                     context_frames=12, fps=24, context_overlap=3).videos
        frames = (video[0].permute(1,2,3,0).clamp(-1,1).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
        imageio.mimwrite(str(outf), frames, fps=24, quality=8)
        print('saved', outf, flush=True)
    print('EVAL_GEN_DONE')

if __name__ == '__main__':
    main()
