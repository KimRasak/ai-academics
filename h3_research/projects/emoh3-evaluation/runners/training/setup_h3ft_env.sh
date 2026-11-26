#!/bin/bash
# Set up the H3 fine-tuning env (Python 3.11 + torch>=2.8 + pinned diffusers/transformers)
set -euo pipefail
export PATH=/opt/conda/bin:$PATH
ENV=/gluster_hdd_cv/user/jinzili/env/h3ft
LOG=/gluster_hdd_cv/user/jinzili/tmp/pilot/h3ft_env_setup.log
{
echo "=== [$(date +%H:%M:%S)] create conda env ==="
mamba create -p $ENV python=3.11 -y 2>&1 | tail -3
echo "=== [$(date +%H:%M:%S)] install torch cu128 ==="
$ENV/bin/pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.8" torchvision torchaudio 2>&1 | tail -4
echo "=== [$(date +%H:%M:%S)] install remaining deps ==="
$ENV/bin/pip install "accelerate>=1.10" "deepspeed>=0.18" "peft>=0.18" "av>=12" "safetensors>=0.4" "transformers>=4.56" 2>&1 | tail -3
echo "=== [$(date +%H:%M:%S)] run install_env.sh (pinned diffusers+transformers) ==="
cd /gluster_hdd_cv/user/jinzili/tmp/h3ft
H3_ENV_PREFIX=$ENV bash install_env.sh 2>&1 | tail -8
echo "=== [$(date +%H:%M:%S)] verify ==="
$ENV/bin/python -c "import torch; from diffusers import MiniMaxH3Transformer3DModel; print('ENV_OK torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -3
echo "=== [$(date +%H:%M:%S)] DONE ==="
} > $LOG 2>&1
