#!/bin/bash

files=(
# "ghost_enhanced_rgba.png"
# "catstatue_enhanced_rgba.png"
# "rabbit_chinese_enhanced_rgba.png"
# "anya_enhanced_rgba.png"
# "csm_luigi_enhanced_rgba.png"
"frog_sweater_enhanced_rgba.png"
# "zelda_enhanced_rgba.png"
# "astronaut_enhanced_rgba.png"
)

for f in "${files[@]}"; do
    input="data/$f"
    save_path="${f%_enhanced_rgba.png}"

    CUDA_VISIBLE_DEVICES=0 python stage2.py --config configs/Contrastive.yaml input="$input" save_path="$save_path"
done
