#!/bin/bash

files=(
# "ghost_rgba.png"
# "catstatue_rgba.png"
# "rabbit_chinese_rgba.png"
# "anya_rgba.png"
# "csm_luigi_rgba.png"
"frog_sweater_rgba.png"
# "zelda_rgba.png"
# "astronaut_rgba.png"
)

for f in "${files[@]}"; do
    input="data/$f"
    save_path="${f%_rgba.png}"

    CUDA_VISIBLE_DEVICES=0 python stage1.py --config configs/Contrastive.yaml input="$input" save_path="$save_path"
done
