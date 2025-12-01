#!/bin/bash
# Compile the rotary_embedding kernel (vLLM-style, separate cos/sin)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
TORCH_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
TORCH_INCLUDE="${TORCH_DIR}/../../include"
TORCH_INCLUDE_CSRC="${TORCH_DIR}/../../include/torch/csrc/api/include"
TORCH_LIB="${TORCH_DIR}/../../lib"

GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
if [ -z "$GPU_ARCH" ]; then
    GPU_ARCH="80"  # Default to sm_80
fi

echo "=============================================="
echo "Compiling rotary_embedding kernel ..."
echo "Python include: $PYTHON_INCLUDE"
echo "Torch include: $TORCH_INCLUDE"
echo "GPU architecture: sm_${GPU_ARCH}"
echo "=============================================="

# Compile
nvcc -std=c++17 -O3 --shared -Xcompiler -fPIC \
    -DTORCH_EXTENSION_NAME=rotary_embedding \
    -I"$PYTHON_INCLUDE" \
    -I"$TORCH_INCLUDE" \
    -I"$TORCH_INCLUDE_CSRC" \
    -I/usr/local/cuda/include \
    -gencode arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH} \
    rotary_embedding.cu \
    -L"$TORCH_LIB" \
    -ltorch -ltorch_python -lc10 -lc10_cuda \
    -o rotary_embedding.so

echo "=============================================="
echo "Compilation successful!"
echo "Output: rotary_embedding.so"
echo "=============================================="

# Test import
python -c "
import torch
import sys
sys.path.insert(0, '.')
import rotary_embedding
print('Available functions:', dir(rotary_embedding))
print('Import test passed!')
"