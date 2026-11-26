#!/usr/bin/env python3
"""Merge LoRA into a copy of denoising_unet.pth so the ORIGINAL infer_audio2vid
pipeline can serve LoRA variants with zero pipeline divergence.
Usage: python merge_lora.py --lora_dir /ssd2/jinzili/avdata/lora_vA2/step8000 --out /ssd2/jinzili/av_weights/merged/vA2_denoising_unet.pth [--reference_out ...]
"""
import argparse, os, sys
sys.path.insert(0, '/ssd2/jinzili/echomimic')
import torch
from omegaconf import OmegaConf
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.unet_2d_condition import UNet2DConditionModel as ForkUNet2D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference_out', default=None)
    ap.add_argument('--r', type=int, default=32)
    ap.add_argument('--alpha', type=int, default=64)
    args = ap.parse_args()
    infer_cfg = OmegaConf.load('/ssd2/jinzili/echomimic/configs/inference/inference_v2.yaml')
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        '/ssd2/jinzili/av_weights/echomimic/sd-image-variations-diffusers',
        '/ssd2/jinzili/av_weights/echomimic/motion_module.pth',
        subfolder='unet', unet_additional_kwargs=infer_cfg.unet_additional_kwargs)
    denoising_unet.load_state_dict(
        torch.load('/ssd2/jinzili/av_weights/echomimic/denoising_unet.pth', map_location='cpu'), strict=False)
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, target_modules=['to_q','to_k','to_v'])
    for name, net in (('denoising', denoising_unet),):
        lp = os.path.join(args.lora_dir, f'{name}_lora.pt')
        sd = torch.load(lp, map_location='cpu')
        netp = get_peft_model(net, lora_cfg)
        netp.load_state_dict(sd, strict=False)
        netp.merge_and_unload()
        print(f'merged {name} ({len(sd)} tensors)', flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(denoising_unet.state_dict(), args.out)
    print('saved merged denoising unet ->', args.out, flush=True)
    if args.reference_out:
        reference_unet = ForkUNet2D.from_pretrained(
            '/ssd2/jinzili/av_weights/echomimic/sd-image-variations-diffusers', subfolder='unet')
        reference_unet.load_state_dict(
            torch.load('/ssd2/jinzili/av_weights/echomimic/reference_unet.pth', map_location='cpu'), strict=False)
        sd = torch.load(os.path.join(args.lora_dir, 'reference_lora.pt'), map_location='cpu')
        netp = get_peft_model(reference_unet, lora_cfg)
        netp.load_state_dict(sd, strict=False)
        netp.merge_and_unload()
        torch.save(reference_unet.state_dict(), args.reference_out)
        print('saved merged reference unet ->', args.reference_out, flush=True)

if __name__ == '__main__':
    main()
