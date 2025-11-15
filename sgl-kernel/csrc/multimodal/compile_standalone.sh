#!/bin/bash
set -e

echo "==================================="
echo "单独编译 rotary_embedding kernel"
echo "==================================="

# 项目根目录（相对本脚本）
ROOT_DIR="$(cd "$(dirname "$0")/../.."; pwd)"

# 获取 PyTorch 路径 / Python include
TORCH_PATH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
PYTHON_INCLUDE=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
echo "PyTorch 路径: $TORCH_PATH"
echo "Python include 路径: $PYTHON_INCLUDE"

# 自动探测本机 CUDA 架构
CUDA_CC=$(python3 - <<'PY'
import torch
cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (9,0)
print(f"{cc[0]}{cc[1]}")
PY
)
echo "CUDA Compute Capability: ${CUDA_CC}"

echo ""
echo "开始编译..."
nvcc -std=c++17 -O3 --shared -Xcompiler -fPIC \
    -DTORCH_EXTENSION_NAME=sgl_rotary \
    -I/usr/local/cuda/include \
    -I${TORCH_PATH}/include \
    -I${TORCH_PATH}/include/torch/csrc/api/include \
    -I${PYTHON_INCLUDE} \
    -I${ROOT_DIR}/include \
    -I${ROOT_DIR}/csrc \
    -gencode arch=compute_${CUDA_CC},code=sm_${CUDA_CC} \
    ./sgl_rotary_bind.cu \
    -L${TORCH_PATH}/lib \
    -ltorch -ltorch_python -lc10 -lc10_cuda \
    -o sgl_rotary.so

if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "编译成功!"
    echo "输出文件: sgl_rotary.so"
    echo "==================================="
    echo ""
    echo "现在可以运行测试:"
    echo "  python test_standalone_rotary.py"
else
    echo "编译失败"
    exit 1
fi

