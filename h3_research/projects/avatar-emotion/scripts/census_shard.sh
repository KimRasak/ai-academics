#!/bin/bash
# Census shard: run on one GPU. Usage: bash census_shard.sh <SHARD 0-3>
set -u
SHARD=$1
ENV_PY=/ssd2/jinzili/envs/echomimic/bin/python
export PATH=/ssd2/jinzili/ffmpeg-7.0.2-amd64-static:$PATH
export FFMPEG_PATH=/ssd2/jinzili/ffmpeg-7.0.2-amd64-static
cd /ssd2/jinzili/echomimic
GPU=$((SHARD+4))
SEEDS=(420 421 422 423)

run_one () {
  local audio=$1 outname=$2 seed=$3 frames=$4
  local yml=/tmp/census/${outname}_seed${seed}.yaml
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
  "./assets/test_imgs/a.png":
    - "$audio"
YEOF
  CUDA_VISIBLE_DEVICES=$GPU $ENV_PY infer_audio2vid.py --config $yml -W 512 -H 512 -L $frames --seed $seed > /tmp/census/${outname}_seed${seed}.log 2>&1
  local newest
  newest=$(ls -t output/*/*/*.mp4 2>/dev/null | grep -v withaudio | head -1)
  if [ -n "$newest" ]; then
    cp "$newest" "/ssd2/jinzili/avdata/gen/arm1/${outname}_seed${seed}.mp4"
  else
    echo "[shard$SHARD] NO_OUTPUT $outname seed$seed"
  fi
}

for c in $(seq 0 39); do
  if [ $((c % 4)) -ne $SHARD ]; then continue; fi
  cc=$(printf "chunk_%04d" $c)
  for s in "${SEEDS[@]}"; do
    out=/ssd2/jinzili/avdata/gen/arm1/${cc}_seed${s}.mp4
    if [ ! -f "$out" ]; then
      echo "[shard$SHARD] $cc seed$s $(date +%H:%M:%S)"
      run_one /ssd2/jinzili/avdata/$cc/audio.wav $cc $s 200
    fi
  done
done

if [ $SHARD -eq 0 ]; then
  for s in "${SEEDS[@]}"; do
    out=/ssd2/jinzili/avdata/gen/arm1/session40_seed${s}.mp4
    if [ ! -f "$out" ]; then
      echo "[shard0-ARM2] session40 seed$s $(date +%H:%M:%S)"
      run_one /ssd2/jinzili/avdata/arm2/session40.wav session40 $s 4000
    fi
  done
fi
echo "SHARD${SHARD}_DONE $(date +%H:%M:%S)"
