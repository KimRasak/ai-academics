#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=/gluster_hdd_cv/user/jinzili/code/research-work-reproduction/VADER
WEIGHT_ROOT=/gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/vader
ENV_ROOT=/gluster_hdd_cv/user/jinzili/env/vader-videocrafter
OUTPUT_ROOT=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/vader/outputs

sha256sum \
  "$WEIGHT_ROOT/videocrafter2-model.ckpt" \
  "$WEIGHT_ROOT/vader_videocrafter_pickscore.pt" \
  "$WEIGHT_ROOT/vader_videocrafter_hps_aesthetic.pt"

cd "$CODE_ROOT/VADER-VideoCrafter"

# Run only after protocol.json is frozen and the target GPU is audited.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set one audited GPU}" \
"$ENV_ROOT/bin/accelerate" launch --num_processes 1 \
  scripts/main/train_t2v_lora.py \
  --seed 200 \
  --mode base \
  --ckpt_path "$WEIGHT_ROOT/videocrafter2-model.ckpt" \
  --config configs/inference_t2v_512_v2.0.yaml \
  --height 320 \
  --width 512 \
  --unconditional_guidance_scale 12.0 \
  --ddim_steps 25 \
  --ddim_eta 1.0 \
  --frames 24 \
  --prompt_fn chatgpt_inference \
  --val_batch_size 1 \
  --num_val_runs 1 \
  --lora_rank 16 \
  --inference_only true \
  --project_dir "$OUTPUT_ROOT/pickscore-smoke" \
  --lora_ckpt_path "$WEIGHT_ROOT/vader_videocrafter_pickscore.pt" \
  --is_sample_preview true \
  --use_wandb false

