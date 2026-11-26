#!/bin/bash
# M4/M3 eval for LoRA variants via ORIGINAL infer pipeline (zero divergence):
# 1) merge LoRA into denoising_unet.pth copy  2) run infer_audio2vid with merged weights
# Usage: nohup bash eval_variant_batch.sh > eval_variant.log 2>&1 &
set -u
ENV_PY=/ssd2/jinzili/envs/echomimic/bin/python
export PATH=/ssd2/jinzili/ffmpeg-7.0.2-amd64-static:$PATH
cd /ssd2/jinzili/echomimic

LORA_DIR=${LORA_DIR:?must set LORA_DIR}
TAG=${TAG:?must set TAG}
AUDIOS=${AUDIOS:-/ssd2/jinzili/avdata/probes48.json}
R=${R:-32}
ALPHA=${ALPHA:-64}
OUTDIR=${OUTDIR:-/ssd2/jinzili/avdata/gen/eval_$TAG}
GPU=${GPU:-6}

mkdir -p "$OUTDIR" /tmp/census

# 1) merge
MERGED=/tmp/census/${TAG}_denoising_unet.pth
if [ ! -f "$MERGED" ]; then
  CUDA_VISIBLE_DEVICES=$GPU $ENV_PY /ssd2/jinzili/avdata/merge_lora.py \
    --lora_dir "$LORA_DIR" --out "$MERGED" --r "$R" --alpha "$ALPHA" || exit 1
fi

# 2) generate per audio with merged weights
run_one () {
  local audio=$1 out=$2
  local tag
  tag=$(basename "$audio" .wav)
  local yml=/tmp/census/${TAG}_${tag}.yaml
  cat > "$yml" <<EOFCFG
audio_model_path: "/ssd2/jinzili/av_weights/echomimic/audio_processor/whisper_tiny.pt"
denoising_unet_path: "$MERGED"
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
  CUDA_VISIBLE_DEVICES=$GPU $ENV_PY infer_audio2vid.py --config "$yml" -W 512 -H 512 -L 200 --seed 420 > /tmp/census/${TAG}_${tag}.log 2>&1
  local newest
  newest=$(ls -t output/*/*/*.mp4 2>/dev/null | grep -v withaudio | head -1)
  [ -n "$newest" ] && cp "$newest" "$out" && echo "saved $out"
}

for audio in $(python3 -c "import json; print(' '.join(json.load(open('$AUDIOS'))))"); do
  idx=$(basename "$audio" .wav | grep -oE '[0-9]+$')
  out="$OUTDIR/probe2_${idx}_k0.mp4"
  [ -f "$out" ] && continue
  echo "[eval-$TAG] $audio $(date +%H:%M:%S)"
  run_one "$audio" "$out"
done
echo EVAL_VARIANT_DONE
