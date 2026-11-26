#!/bin/bash
# Prep: extract base-8s x01 frames (8fr) + build a prefixed symlink tree of ALL frame dirs.
set -e
cd /gluster_hdd_cv/user/jinzili/tmp/pilot
PY=/gluster_hdd_cv/user/jinzili/env/sglang310/bin/python

# 1) base 8-step x01 frames (the pilot 8-step base clip), 8 frames, named like base cases
rm -rf frames_base8s
$PY extract8_frames.py outputs frames_base8s x01_neutral_to_joyful
if [ -d frames_base8s/x01_neutral_to_joyful ]; then
  mv frames_base8s/x01_neutral_to_joyful frames_base8s/x01_neutral_to_joyful_base
fi
echo "base8s: $(ls frames_base8s)"

# 2) prefixed symlink tree of every frame dir (disambiguate duplicate case names)
rm -rf frames_all
mkdir -p frames_all
for d in frames_base16 frames_base16b frames_base8s frames_lora frames_lora2 \
         frames_lora2_8 frames_lora3 frames_lora3s frames_s999 frames_lora4 \
         frames_lora4s frames_x02ctrl frames_x02sweep_base frames_x02sweep_lora \
         frames_refswap_base_s42 frames_refswap_base_s1234 frames_refswap_lora_s42 \
         frames_refswap_lora_s1234; do
  [ -d "$d" ] || { echo "SKIP missing $d"; continue; }
  for c in "$d"/*; do
    [ -d "$c" ] || continue
    ln -s "../$d/$(basename "$c")" "frames_all/${d}__$(basename "$c")"
  done
done
echo "frames_all cases: $(ls frames_all | wc -l)"
echo PREP_DONE
