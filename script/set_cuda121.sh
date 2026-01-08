#!/bin/bash
# 设置CUDA 12.1环境变量
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# 优先使用conda环境的库，然后是CUDA库
if [ ! -z "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
fi

echo "CUDA环境已设置为12.1版本"
echo "CUDA_HOME: $CUDA_HOME"
nvcc --version

