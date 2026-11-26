#!/usr/bin/env python3
"""V-B/V-C eval: generate with LoRA + emotion-embedding conditioning.
At inference, SER(audio) picks the emotion id (zero-annotation).
Usage: CUDA_VISIBLE_DEVICES=5 python eval_lora_vb.py --lora_dir .../lora_vB/step8000 --tag vB_s8000"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, '/ssd2/jinzili/echomimic')
import numpy as np
import torch
import cv2, imageio
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from src.models.unet_2d_condition import UNet2DConditionModel as ForkUNet2D
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.face_locator import FaceLocator
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from facenet_pytorch import MTCNN

EMO_ID = {'neutral':0,'calm':1,'happy':2,'sad':3,'angry':4,'fearful':5,'fear':5,'disgust':6,'surprised':7,'anxious':8}

class EncHSInjector:
    """wraps denoising unet, forces encoder_hidden_states (CFG-batched)"""
    def __init__(self, unet, hs):
        self.unet = unet; self.hs = hs
    def __call__(self, *a, **kw):
        kw['encoder_hidden_states'] = self.hs
        return self.unet(*a, **kw)
    def __getattr__(self, n):
        return getattr(self.unet, n)

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora_dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--audios', required=True)
    ap.add_argument('--emo_map', required=True)  # json {audio_path: emotion_str}
    ap.add_argument('--ref', default='/ssd2/jinzili/echomimic/assets/test_imgs/a.png')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--seed', type=int, default=420)
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
    lora_cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=['to_q','to_k','to_v'])
    for name, net in (('denoising', denoising_unet), ('reference', reference_unet)):
        sd = torch.load(os.path.join(args.lora_dir, f'{name}_lora.pt'), map_location='cpu')
        netp = get_peft_model(net, lora_cfg)
        netp.load_state_dict(sd, strict=False)
        netp.merge_and_unload()
        print(f'merged {name} LoRA', flush=True)

    emo = torch.nn.Embedding(12, 768).to(device, dtype=dtype)
    emo.load_state_dict(torch.load(os.path.join(args.lora_dir, 'emo_embed.pt'), map_location='cpu'))
    emo_map = json.load(open(args.emo_map))

    pipe = Audio2VideoPipeline(vae=vae, reference_unet=reference_unet, denoising_unet=denoising_unet,
                               audio_guider=audio_guider, face_locator=face_locator, scheduler=scheduler).to(device)
    mtcnn = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6,0.7,0.7], factor=0.709, post_process=True, device=device)
    ref_img = cv2.imread(args.ref)
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

    outdir = Path(f'/ssd2/jinzili/avdata/gen/eval_{args.tag}')
    outdir.mkdir(parents=True, exist_ok=True)
    emo_log = {}
    for apath in [l.strip() for l in open(args.audios) if l.strip()]:
        name = Path(apath).parent.name
        if 'chunk_' not in name:
            name = Path(apath).stem
        outf = outdir / f'{name}_seed{args.seed}.mp4'
        if outf.exists():
            continue
        estr = emo_map.get(apath, 'neutral')
        eid = EMO_ID.get(estr, 0)
        e = emo(torch.tensor([eid], device=device)).unsqueeze(1)  # (1,1,768)
        e = e.repeat(2, 1, 1)  # CFG batch
        denoising_unet_forward = denoising_unet.forward
        pipe.denoising_unet = EncHSInjector(denoising_unet, e)
        g = torch.Generator(device=device); g.manual_seed(args.seed)
        video = pipe(ref_pil, apath, face_mask, W, H, args.frames, args.steps,
                     2.5, generator=g, audio_sample_rate=16000,
                     context_frames=12, fps=24, context_overlap=3).videos
        pipe.denoising_unet = denoising_unet
        frames = (video[0].permute(1,2,3,0).clamp(-1,1).cpu().numpy()*127.5+127.5).astype(np.uint8)
        imageio.mimwrite(str(outf), frames, fps=24, quality=8)
        emo_log[outf.name] = estr
        print('saved', outf.name, '| emo:', estr, flush=True)
    json.dump(emo_log, open(outdir/'emo_log.json','w'))
    print('EVAL_GEN_DONE')

if __name__ == '__main__':
    main()
