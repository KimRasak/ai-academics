#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=/gluster_hdd_cv/user/jinzili/code/research-work-reproduction/t2v-turbo
RELEASE_CODE_ROOT=/gluster_hdd_cv/user/jinzili/code/research-work-reproduction/t2v-turbo-vc2-release
ENV_ROOT=/gluster_hdd_cv/user/jinzili/env/t2v-turbo-vc2
WEIGHT_ROOT=/gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/t2v-turbo
BASE_CHECKPOINT=/gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/vader/videocrafter2-model.ckpt
OUTPUT_ROOT=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/t2v-turbo/outputs/current-code-release-protocol-smoke-v2
VBENCH_ROOT=/gluster_hdd_cv/user/jinzili/code/research-work-reproduction/VBench-paper-20240529
VBENCH_OUTPUT=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/t2v-turbo/outputs/vbench-standard-independent-seeds-v1
VBENCH_INPUT=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/t2v-turbo/evaluation-input/vbench-standard-independent-seeds-v1/videos
VBENCH_EVAL_OUTPUT=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/t2v-turbo/evaluation/vbench-paper-20240529-independent-seeds-v1
VBENCH_ENV=/gluster_hdd_cv/user/jinzili/env/vbench-paper-20240529
VBENCH_GRIT_ENV=/gluster_hdd_cv/user/jinzili/env/vbench-grit-paper-20240529
RUNNER=/gluster_hdd_cv/user/jinzili/code/research-work-reproduction/t2v-turbo-vc2-batch-generate.py

git -C "$CODE_ROOT" rev-parse HEAD
git -C "$RELEASE_CODE_ROOT" rev-parse HEAD
sha256sum "$WEIGHT_ROOT/unet_lora.pt" "$BASE_CHECKPOINT"

CUDA_VISIBLE_DEVICES=7 "$ENV_ROOT/bin/python" batch_generate.py \
  --code-root "$CODE_ROOT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --lora-checkpoint "$WEIGHT_ROOT/unet_lora.pt" \
  --output-dir "$OUTPUT_ROOT" \
  --prompt-id smoke_0000 \
  --prompt "An astronaut riding a horse." \
  --seed 0 \
  --steps 4 \
  --origin-steps 50 \
  --fps 16 \
  --dtype float16

# Audit the four-way plan without loading the model.
for shard in 0 1 2 3; do
  "$ENV_ROOT/bin/python" "$RUNNER" \
    --code-root "$CODE_ROOT" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --lora-checkpoint "$WEIGHT_ROOT/unet_lora.pt" \
    --output-dir "$VBENCH_OUTPUT" \
    --prompt-file "$VBENCH_ROOT/prompts/all_dimension.txt" \
    --samples-per-prompt 5 \
    --num-shards 4 \
    --shard-index "$shard" \
    --plan-only
done

# Generate one shard per GPU. Author seeds are unavailable; seeds 0..4719 are
# assigned deterministically and recorded per video.
pids=()
for shard in 0 1 2 3; do
  gpu=$((4 + shard))
  CUDA_VISIBLE_DEVICES="$gpu" "$ENV_ROOT/bin/python" "$RUNNER" \
    --code-root "$CODE_ROOT" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --lora-checkpoint "$WEIGHT_ROOT/unet_lora.pt" \
    --output-dir "$VBENCH_OUTPUT" \
    --prompt-file "$VBENCH_ROOT/prompts/all_dimension.txt" \
    --samples-per-prompt 5 \
    --seed-base 0 \
    --num-shards 4 \
    --shard-index "$shard" \
    --skip-existing \
    > "$VBENCH_OUTPUT/run.shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$ENV_ROOT/bin/python" \
  /gluster_hdd_cv/user/jinzili/code/research-work-reproduction/t2v-turbo-vc2-validate-generation.py \
  --output-dir "$VBENCH_OUTPUT" \
  --prompt-file "$VBENCH_ROOT/prompts/all_dimension.txt" \
  --samples-per-prompt 5 \
  --num-shards 4 \
  --report "$VBENCH_OUTPUT/validation_report.json"

# Paper-day VBench uses a hard-coded /root/.cache/vbench path. Link the
# separately audited checkpoints there before evaluation. Do not point the
# evaluator at the mixed generation directory: it infers the media suffix from
# the first directory entry and can silently produce zero-coverage NaNs.
mkdir -p /root/.cache/vbench/{caption_model,umt_model,ViCLIP,grit_model}
ln -sfn /gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/vbench/caption_model/tag2text_swin_14m.pth \
  /root/.cache/vbench/caption_model/tag2text_swin_14m.pth
ln -sfn /gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/vbench/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth \
  /root/.cache/vbench/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth
ln -sfn /gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth \
  /root/.cache/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth
ln -sfn /gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/vbench/ViCLIP/bpe_simple_vocab_16e6.txt.gz \
  /root/.cache/vbench/ViCLIP/bpe_simple_vocab_16e6.txt.gz
ln -sfn /gluster_hdd_cv/user/jinzili/weights/research-work-reproduction/vbench/grit_model/grit_b_densecap_objectdet.pth \
  /root/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth

standard_dimensions=(
  subject_consistency background_consistency temporal_flickering
  motion_smoothness aesthetic_quality dynamic_degree imaging_quality
  human_action scene appearance_style temporal_style overall_consistency
)
for dimension in "${standard_dimensions[@]}"; do
  "$VBENCH_ENV/bin/python" "$VBENCH_ROOT/evaluate.py" \
    --videos_path "$VBENCH_INPUT" \
    --dimension "$dimension" \
    --output_path "$VBENCH_EVAL_OUTPUT/${dimension}-videos-only-replay" \
    --mode vbench_standard \
    --load_ckpt_from_local True
done

grit_dimensions=(object_class multiple_objects color spatial_relationship)
for dimension in "${grit_dimensions[@]}"; do
  PYTHONPATH="$VBENCH_ROOT" "$VBENCH_GRIT_ENV/bin/python" \
    "$VBENCH_ROOT/evaluate.py" \
    --videos_path "$VBENCH_INPUT" \
    --dimension "$dimension" \
    --output_path "$VBENCH_EVAL_OUTPUT/${dimension}-videos-only-replay" \
    --mode vbench_standard \
    --load_ckpt_from_local True
done
