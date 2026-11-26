#!/bin/bash
set -euo pipefail
root=/ssd2/jinzili/scivbench_vt_weights
mkdir -p "$root/dino_model" "$root/clip_model" "$root/aesthetic_model/emb_reader" "$root/amt_model" "$root/pyiqa_model"
curl -fL --retry 3 https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth -o "$root/dino_model/dino_vitbase16_pretrain.pth"
curl -fL --retry 3 https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -o "$root/clip_model/ViT-B-32.pt"
curl -fL --retry 3 https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt -o "$root/clip_model/ViT-L-14.pt"
curl -fL --retry 3 https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth -o "$root/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
curl -fL --retry 3 https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth -o "$root/amt_model/amt-s.pth"
curl -fL --retry 3 https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth -o "$root/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
