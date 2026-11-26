#!/bin/bash
# M4 baseline probes via original infer pipeline (proven arm1 path)
set -u
ENV_PY=/ssd2/jinzili/envs/echomimic/bin/python
export PATH=/ssd2/jinzili/ffmpeg-7.0.2-amd64-static:$PATH
cd /ssd2/jinzili/echomimic
mkdir -p /ssd2/jinzili/avdata/gen/m4e_base /tmp/census
run_one () {
  local audio=$1 out=$2 gpu=$3
  local tag
  tag=$(basename "$audio" .wav)
  local yml=/tmp/census/${tag}_probe.yaml
  cat > "$yml" <<EOFCFG
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
  "./assets/test_imgs/a.png":
    - "$audio"
EOFCFG
  CUDA_VISIBLE_DEVICES=$gpu $ENV_PY infer_audio2vid.py --config "$yml" -W 512 -H 512 -L 200 --seed 420 > /tmp/census/${tag}_probe.log 2>&1
  local newest
  newest=$(ls -t output/*/*.mp4 2>/dev/null | grep -v withaudio | head -1)
  [ -n "$newest" ] && cp "$newest" "$out"
}
for i in 00 01 02 03 04 05 06 07 08 09 10 11; do
  out=/ssd2/jinzili/avdata/gen/m4e_base/probe2_${i}_k0.mp4
  [ -f "$out" ] && continue
  echo "[probe] probe2_$i $(date +%H:%M:%S)"
  run_one /ssd2/jinzili/avdata/probes48/probe2_${i}.wav "$out" 6
done
echo PROBE_BATCH_DONE
