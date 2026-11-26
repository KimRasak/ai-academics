#!/bin/bash
# Reference-anchoring control: expressive ref (b.png) x emotional chunks, co-located on GPU 7
set -u
ENV_PY=/ssd2/jinzili/envs/echomimic/bin/python
export PATH=/ssd2/jinzili/ffmpeg-7.0.2-amd64-static:$PATH
export FFMPEG_PATH=/ssd2/jinzili/ffmpeg-7.0.2-amd64-static
cd /ssd2/jinzili/echomimic
mkdir -p /ssd2/jinzili/avdata/gen/anchor
for pair in "chunk_0000 angry" "chunk_0007 happy"; do
  set -- $pair
  cc=$1; tag=$2
  yml=/tmp/census/anchor_${cc}.yaml
  cat > $yml <<YEOF
audio_model_path: "/ssd2/jinzili/av_weights/echomimic/audio_processor/whisper_tiny.pt"
denoising_unet_path: "/ssd2/jinzili/av_weights/echomimic/denoising_unet.pth"
reference_unet_path: "/ssd2/jinzili/av_weights/echomimic/reference_unet.pth"
face_locator_path: "/ssd2/jinzili/av_weights/echomimic/face_locator.pth"
motion_module_path: "/ssd2/jinzili/av_weights/echomimic/motion_module.pth"
pretrained_base_model_path: "/ssd2/jinzili/av_weights/echomimic/sd-image-variations-diffusers"
pretrained_vae_path: "/ssd2/jinzili/av_weights/echomimic/sd-vae-ft-mse"
inference_config: "./configs/inference/inference_v2.yaml"
weight_dtype: "fp16"
test_cases:
  "./assets/test_imgs/b.png":
    - "/ssd2/jinzili/avdata/$cc/audio.wav"
YEOF
  echo "[anchor] $cc ($tag) $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=7 $ENV_PY infer_audio2vid.py --config $yml -W 512 -H 512 -L 200 --seed 420 > /tmp/census/anchor_${cc}.log 2>&1
  newest=$(ls -t output/*/*/*.mp4 2>/dev/null | grep -v withaudio | head -1)
  [ -n "$newest" ] && cp "$newest" "/ssd2/jinzili/avdata/gen/anchor/${cc}_${tag}_seed420.mp4"
done
echo "ANCHOR_DONE $(date +%H:%M:%S)"
