#!/bin/bash
set -e

echo "==================================="
echo "单独编译 rotary_embedding kernel"
echo "==================================="

# 获取 PyTorch 路径
TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
echo "PyTorch 路径: $TORCH_PATH"

# 获取 Python include 路径
PYTHON_INCLUDE=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
echo "Python include 路径: $PYTHON_INCLUDE"

# 编译参数
CUDA_ARCH="90"  # 根据您的 GPU 修改

echo ""
echo "开始编译..."
nvcc -std=c++17 -O3 --shared -Xcompiler -fPIC \
    -DTORCH_EXTENSION_NAME=standalone_rotary \
    -I/usr/local/cuda/include \
    -I${TORCH_PATH}/include \
    -I${TORCH_PATH}/include/torch/csrc/api/include \
    -I${PYTHON_INCLUDE} \
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \
    standalone_rotary.cu \
    -L${TORCH_PATH}/lib \
    -ltorch -ltorch_python -lc10 -lc10_cuda \
    -o standalone_rotary.so

if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "编译成功!"
    echo "输出文件: standalone_rotary.so"
    echo "==================================="
    echo ""
    echo "现在可以运行测试:"
    echo "  python test_standalone_rotary.py"
else
    echo "编译失败"
    exit 1
fi

