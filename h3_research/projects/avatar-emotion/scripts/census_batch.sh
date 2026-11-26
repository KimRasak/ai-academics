#!/bin/bash
# Census batch driver on remote (run inside /ssd2/jinzili/echomimic)
# Usage: nohup bash census_batch.sh > census_batch.log 2>&1 &
set -u
ENV_PY=/ssd2/jinzili/envs/echomimic/bin/python
export PATH=/ssd2/jinzili/ffmpeg-7.0.2-amd64-static:$PATH
export FFMPEG_PATH=/ssd2/jinzili/ffmpeg-7.0.2-amd64-static
cd /ssd2/jinzili/echomimic
mkdir -p /ssd2/jinzili/avdata/gen/arm1 /ssd2/jinzili/avdata/gen/arm2 /tmp/census

GPU_LIST=(4 5 6 7)
SUFFIX=0

run_one () {  # audio_path tag seed gpu
  local audio=$1 tag=$2 seed=$3 gpu=$4
  local yml=/tmp/census/${tag}_seed${seed}.yaml
  cat > $yml <<EOF
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
EOF
  CUDA_VISIBLE_DEVICES=$gpu $ENV_PY infer_audio2vid.py --config $yml -W 512 -H 512 -L ${5:-200} --seed $seed > /tmp/census/${tag}_seed${seed}.log 2>&1
  # move newest mp4
  local newest
  newest=$(ls -t output/*/*/*.mp4 2>/dev/null | grep -v withaudio | head -1)
  if [ -n "$newest" ]; then
    cp "$newest" /ssd2/jinzili/avdata/gen/${tag}/${tag}_seed${seed}.mp4
  fi
}

# ---------- ARM 1: chunks 0..39 x seeds 420,421,422,423 ----------
SEEDS=(420 421 422 423)
JOBS=()
for c in $(seq 0 39); do
  cc=$(printf "chunk_%04d" $c)
  for s in "${SEEDS[@]}"; do
    out=/ssd2/jinzili/avdata/gen/arm1/${cc}_seed${s}.mp4
    [ -f "$out" ] || JOBS+=("$c $s")
  done
done
echo "arm1 jobs: ${#JOBS[@]}"
i=0
for j in "${JOBS[@]}"; do
  set -- $j
  c=$1; s=$2
  cc=$(printf "chunk_%04d" $c)
  gpu=${GPU_LIST[$(( i % 4 ))]}
  echo "[arm1] chunk $cc seed $s gpu $gpu $(date +%H:%M:%S)"
  run_one /ssd2/jinzili/avdata/$cc/audio.wav arm1/${cc} $s $gpu 200
  i=$((i+1))
done
echo "ARM1_DONE $(date +%H:%M:%S)"

# ---------- ARM 2: long 145s audio x 4 seeds (serial, long jobs) ----------
for s in "${SEEDS[@]}"; do
  out=/ssd2/jinzili/avdata/gen/arm2/session40_seed${s}.mp4
  if [ ! -f "$out" ]; then
    echo "[arm2] session40 seed $s gpu 4 $(date +%H:%M:%S)"
    run_one /ssd2/jinzili/avdata/arm2/session40.wav arm2/session40 $s 4 4000
  fi
done
echo "ARM2_DONE $(date +%H:%M:%S)"
echo "CENSUS_ALL_DONE"
