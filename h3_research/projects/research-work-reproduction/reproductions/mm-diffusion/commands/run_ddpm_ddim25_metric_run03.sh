#!/usr/bin/env bash
set -euo pipefail

# Preregistered MM-Diffusion Landscape checkpoint metric repeat, run 3 of 6.
# GPU binding occurs before Python/PyTorch import; the sampler's native
# unseeded behavior is preserved.
MMROOT=/gluster_hdd_cv/user/jinzili/data/research-work-reproduction/mm-diffusion
GPU_LIST=(4 5 6 7)
RUN_ID=ddpm_ddim25_metric_run03_officialrng_gpu4567_20260920
RANK=${PMI_RANK:?PMI_RANK is required}
GPU=${GPU_LIST[$RANK]:?rank outside GPU list}

export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="$MMROOT/conda-env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$MMROOT/conda-env/bin:$PATH"
export PYTHONPATH="$MMROOT/code"
export HOME="$MMROOT/runtime-home"

exec "$MMROOT/env/bin/python" "$MMROOT/code/py_scripts/multimodal_sample_sr.py" \
  --cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8 \
  --cross_attention_shift True --video_attention_resolutions 2,4,8 \
  --audio_attention_resolutions -1 --video_size 16,3,64,64 --audio_size 1,25600 \
  --learn_sigma False --num_channels 128 --num_head_channels 64 --num_res_blocks 2 \
  --resblock_updown True --use_fp16 True --use_scale_shift_norm True \
  --sr_attention_resolutions 8,16,32 --large_size 256 --small_size 64 \
  --sr_learn_sigma True --sr_num_channels 192 --sr_num_heads 4 --sr_num_res_blocks 2 \
  --sr_resblock_updown True --sr_use_scale_shift_norm True --diffusion_steps 1000 \
  --noise_schedule linear --all_save_num 2048 --save_type mp4 --devices 0,1,2,4 \
  --batch_size 4 --is_strict True --sample_fn ddpm --sr_diffusion_steps 1000 \
  --sr_sample_fn ddim --sr_timestep_respacing ddim25 \
  --ref_path "$MMROOT/raw/extracted/landscape/train" \
  --output_dir "$MMROOT/outputs/$RUN_ID" \
  --multimodal_model_path "$MMROOT/weights/landscape.pt" \
  --sr_model_path "$MMROOT/weights/landscape_SR.pt"
