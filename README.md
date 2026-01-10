# GenerativeGaussian.github.io

This repository contains the source code for the [GenerativeGaussian website](https://github.com/GenGaussian/GenerativeGaussian.github.io).

## Install

```bash
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit

# depth-anything-v2
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth_Anything_V2
cd Depth-Anything-V2
pip install -r requirements.txt

# real-esrgan
cd ../..
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install basicsr
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop

# syncdreamer
cd ..
git clone https://github.com/liuyuan-pal/SyncDreamer.git
cd SynDreamer
pip install -r requirements.txt 
```

Tested on:

- Linux with torch 2.1 & CUDA 12.1 on a 4090.

Other versions should also work.

## Usage
```bash
### preprocess
# background removal and recentering, save rgba at 256x256
python process.py data/name.jpg

# save at a larger resolution
python process.py data/name.jpg --size 512

# process all jpg images under a dir
python process.py data

### training gaussian stage
# train 500 iters (~1min) and export ckpt & coarse_mesh to logs
python main.py --config configs/gen.yaml input=data/name_rgba.png save_path=name

### training mesh stage
# auto load coarse_mesh and refine 50 iters (~1min), export fine_mesh to logs
python main2.py --config configs/gen.yaml input=data/name_rgba.png save_path=name

# specify coarse mesh path explicity
python main2.py --config configs/gen.yaml input=data/name_rgba.png save_path=name mesh=logs/name_mesh.obj

### visualization
# gui for visualizing mesh
# `kire` is short for `python -m kiui.render`
kire logs/name.obj

# save 360 degree video of mesh (can run without gui)
kire logs/name.obj --save_video name.mp4 --wogui

# save 8 view images of mesh (can run without gui)
kire logs/name.obj --save images/name/ --wogui

### evaluation of CLIP-similarity
python -m kiui.cli.clip_sim data/name_rgba.png logs/name.obj
```

Please check `./configs/gen.yaml` for more options.


Helper scripts:

```bash
# run the stage 1 for images (modified in scripts)
source ./script/run1.sh

# run the stage 2 for images (modified in scripts)
source ./script/run2.sh

# test the clip-similarity for stage 1 mesh (modified in scripts)
source ./script/clip1.sh

# test the clip-similarity for stage 2 mesh (modified in scripts)
source ./script/clip2.sh
```

The LPIPS and PSNR of training will be recorded in stage1record.txt and stage2record.txt automatically when you run stage1.py and stage2.py.

# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
  <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" />
</a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
Creative Commons Attribution-ShareAlike 4.0 International License
</a>.


