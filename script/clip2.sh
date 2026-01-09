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
    input_img="data/$f"
    input_obj="${f%_enhanced_rgba.png}.obj"

    CUDA_VISIBLE_DEVICES=0 python kiuikit-main/kiui/cli/clip_sim.py "$input_img" "logs/$input_obj"

done
