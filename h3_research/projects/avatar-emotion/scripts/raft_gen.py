#!/usr/bin/env python3
"""RAFT generation: K samples per audio with a given LoRA checkpoint.
Usage: CUDA_VISIBLE_DEVICES=6 python raft_gen.py --lora_dir .../lora_vD/step2000 --k 3 --shard 0 --nshards 2"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, '/ssd2/jinzili/echomimic')
import numpy as np
import torch, cv2, imageio
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from src.models.unet_2d_condition import UNet2DConditionModel as ForkUNet2D
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.face_locator import FaceLocator
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from facenet_pytorch import MTCNN

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora_dir', required=True)
    ap.add_argument('--audios', default='/ssd2/jinzili/avdata/raft_audios.json')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=2)
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--r', type=int, default=8)
    ap.add_argument('--alpha', type=int, default=16)
    ap.add_argument('--out', default='/ssd2/jinzili/avdata/gen/raft_K')
    return ap.parse_args()

def main():
    args = parse()
    device = 'cuda'
    dtype = torch.float16
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
    face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16,32,96,256)).to(device, dtype=dtype)
    face_locator.load_state_dict(torch.load('/ssd2/jinzili/av_weights/echomimic/face_locator.pth', map_location='cpu'))
    audio_guider = load_audio_model('/ssd2/jinzili/av_weights/echomimic/audio_processor/whisper_tiny.pt', device)
    scheduler = DDIMScheduler.from_pretrained('/ssd2/jinzili/av_weights/echomimic/sd-image-variations-diffusers', subfolder='scheduler')
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, target_modules=['to_q','to_k','to_v'])
    for name, net in (('denoising', denoising_unet), ('reference', reference_unet)):
        sd_path = os.path.join(args.lora_dir, f'{name}_lora.pt')
        if os.path.exists(sd_path):
            sd = torch.load(sd_path, map_location='cpu')
            netp = get_peft_model(net, lora_cfg)
            netp.load_state_dict(sd, strict=False)
            netp.merge_and_unload()
            print(f'merged {name} ({len(sd)} tensors)', flush=True)
    pipe = Audio2VideoPipeline(vae=vae, reference_unet=reference_unet, denoising_unet=denoising_unet,
                               audio_guider=audio_guider, face_locator=face_locator, scheduler=scheduler).to(device)
    mtcnn = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6,0.7,0.7], factor=0.709, post_process=True, device=device)
    ref_path = '/ssd2/jinzili/echomimic/assets/test_imgs/a.png'
    ref_img = cv2.imread(ref_path)
    # face-crop reference like infer_audio2vid (crop_and_pad around detected face)
    _det = mtcnn.detect(ref_img, landmarks=True)
    if _det[0] is not None and len(_det[0]) > 0:
        x0, y0, x1, y1 = _det[0][0]
        w_pad, h_pad = int((x1-x0)*0.3), int((y1-y0)*0.3)
        x0, y0 = max(0, int(x0-w_pad)), max(0, int(y0-h_pad))
        x1, y1 = min(ref_img.shape[1], int(x1+w_pad)), min(ref_img.shape[0], int(y1+h_pad))
        ref_img = ref_img[y0:y1, x0:x1]
        print('face-cropped ref to', ref_img.shape, flush=True)
    ref_img = cv2.resize(ref_img, (512, 512))
    H = W = 512
    mask = np.zeros((H, W), np.uint8)
    f = mtcnn.detect(ref_img, landmarks=True)
    bboxes, probs = f[0], f[1]
    if bboxes is not None and len(bboxes) > 0 and probs[0] > 0.6:
        x0,y0,x1,y1 = bboxes[0]
        rw, rh = int((x1-x0)*0.52/2), int((y1-y0)*0.4/2)
        cx, cy = int((x0+x1)/2), int((y0+y1)/2 + (y1-y0)*0.1)
        mask[max(0,cy-rh):min(H,cy+rh), max(0,cx-rw):min(W,cx+rw)] = 255
    else:
        mask[int(0.12*H):int(0.88*H), int(0.22*W):int(0.78*W)] = 255
    face_mask = torch.Tensor(cv2.resize(mask,(W,H))).to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)/255.0
    ref_pil = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))

    audios = json.load(open(args.audios))[args.shard::args.nshards]
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    for apath in audios:
        stem = Path(apath).stem
        for k in range(args.k):
            outf = outdir / f'{stem}_k{k}.mp4'
            if outf.exists():
                continue
            g = torch.Generator(device=device); g.manual_seed(1000+k)
            video = pipe(ref_pil, apath, face_mask, W, H, args.frames, args.steps,
                         2.5, generator=g, audio_sample_rate=16000,
                         context_frames=12, fps=24, context_overlap=3).videos
            frames = (video[0].permute(1,2,3,0).clamp(-1,1).cpu().numpy()*127.5+127.5).astype(np.uint8)
            imageio.mimwrite(str(outf), frames, fps=24, quality=7)
            print('saved', outf.name, flush=True)
    print('RAFT_GEN_DONE')

if __name__ == '__main__':
    main()
