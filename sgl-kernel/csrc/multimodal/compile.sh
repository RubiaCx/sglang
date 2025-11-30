#!/bin/bash
# Compile the RoPE kernels

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get Python and PyTorch paths
PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
TORCH_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
TORCH_INCLUDE="${TORCH_DIR}/../../include"
TORCH_INCLUDE_CSRC="${TORCH_DIR}/../../include/torch/csrc/api/include"
TORCH_LIB="${TORCH_DIR}/../../lib"

# Get flashinfer include path - try multiple locations
FLASHINFER_INCLUDE=""

# Method 1: From pip installed flashinfer (data/include)
if [ -z "$FLASHINFER_INCLUDE" ] || [ ! -f "$FLASHINFER_INCLUDE/flashinfer/vec_dtypes.cuh" ]; then
    FLASHINFER_INCLUDE=$(python -c "import flashinfer; import os; print(os.path.join(os.path.dirname(flashinfer.__file__), 'data', 'include'))" 2>/dev/null || echo "")
fi

# Method 2: From pip installed flashinfer (include)
if [ -z "$FLASHINFER_INCLUDE" ] || [ ! -f "$FLASHINFER_INCLUDE/flashinfer/vec_dtypes.cuh" ]; then
    FLASHINFER_INCLUDE=$(python -c "import flashinfer; import os; print(os.path.join(os.path.dirname(flashinfer.__file__), 'include'))" 2>/dev/null || echo "")
fi

# Method 3: From site-packages directly
if [ -z "$FLASHINFER_INCLUDE" ] || [ ! -f "$FLASHINFER_INCLUDE/flashinfer/vec_dtypes.cuh" ]; then
    FLASHINFER_INCLUDE=$(python -c "import site; import os; dirs = site.getsitepackages(); print(next((os.path.join(d, 'flashinfer', 'data', 'include') for d in dirs if os.path.exists(os.path.join(d, 'flashinfer', 'data', 'include'))), ''))" 2>/dev/null || echo "")
fi

# Method 4: Search in common locations
if [ -z "$FLASHINFER_INCLUDE" ] || [ ! -f "$FLASHINFER_INCLUDE/flashinfer/vec_dtypes.cuh" ]; then
    for path in \
        "/usr/local/lib/python3.12/dist-packages/flashinfer/data/include" \
        "/usr/local/lib/python3.10/dist-packages/flashinfer/data/include" \
        "/opt/conda/lib/python3.10/site-packages/flashinfer/data/include" \
        "/usr/local/include" \
        "../../3rdparty/flashinfer/include"; do
        if [ -f "$path/flashinfer/vec_dtypes.cuh" ]; then
            FLASHINFER_INCLUDE="$path"
            break
        fi
    done
fi

# Method 5: Use sgl-kernel's own copy if it exists
if [ -z "$FLASHINFER_INCLUDE" ] || [ ! -f "$FLASHINFER_INCLUDE/flashinfer/vec_dtypes.cuh" ]; then
    SGL_KERNEL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    if [ -f "$SGL_KERNEL_ROOT/3rdparty/flashinfer/include/flashinfer/vec_dtypes.cuh" ]; then
        FLASHINFER_INCLUDE="$SGL_KERNEL_ROOT/3rdparty/flashinfer/include"
    fi
fi

# Method 6: Check CMake build directory (FetchContent location)
if [ -z "$FLASHINFER_INCLUDE" ] || [ ! -f "$FLASHINFER_INCLUDE/flashinfer/vec_dtypes.cuh" ]; then
    SGL_KERNEL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    # Find CMake FetchContent directory
    for build_dir in "$SGL_KERNEL_ROOT/build" "$SGL_KERNEL_ROOT/_skbuild/"*"/cmake-build"; do
        if [ -d "$build_dir" ]; then
            FETCH_DIR=$(find "$build_dir" -name "repo-flashinfer-src" -type d 2>/dev/null | head -1)
            if [ -n "$FETCH_DIR" ] && [ -f "$FETCH_DIR/include/flashinfer/vec_dtypes.cuh" ]; then
                FLASHINFER_INCLUDE="$FETCH_DIR/include"
                break
            fi
        fi
    done
fi

# Method 7: Clone flashinfer if not found
if [ -z "$FLASHINFER_INCLUDE" ] || [ ! -f "$FLASHINFER_INCLUDE/flashinfer/vec_dtypes.cuh" ]; then
    FLASHINFER_CLONE_DIR="$SCRIPT_DIR/.flashinfer"
    if [ ! -d "$FLASHINFER_CLONE_DIR" ]; then
        echo "FlashInfer not found, cloning..."
        git clone --depth 1 https://github.com/flashinfer-ai/flashinfer.git "$FLASHINFER_CLONE_DIR"
    fi
    if [ -f "$FLASHINFER_CLONE_DIR/include/flashinfer/vec_dtypes.cuh" ]; then
        FLASHINFER_INCLUDE="$FLASHINFER_CLONE_DIR/include"
    fi
fi

if [ -z "$FLASHINFER_INCLUDE" ] || [ ! -f "$FLASHINFER_INCLUDE/flashinfer/vec_dtypes.cuh" ]; then
    echo "ERROR: Could not find flashinfer/vec_dtypes.cuh"
    echo "Please set FLASHINFER_INCLUDE environment variable manually"
    echo "Example: export FLASHINFER_INCLUDE=/path/to/flashinfer/include"
    exit 1
fi

# Detect GPU architecture
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
if [ -z "$GPU_ARCH" ]; then
    GPU_ARCH="80"  # Default to sm_80
fi

echo "=============================================="
echo "Compiling RoPE kernels..."
echo "Python include: $PYTHON_INCLUDE"
echo "Torch include: $TORCH_INCLUDE"
echo "FlashInfer include: $FLASHINFER_INCLUDE"
echo "GPU architecture: sm_${GPU_ARCH}"
echo "=============================================="

# Compile
# TORCH_EXTENSION_NAME must match the output .so name for PyInit_<name> to work
nvcc -std=c++20 -O3 --shared -Xcompiler -fPIC \
    -DTORCH_EXTENSION_NAME=rope \
    -I"$PYTHON_INCLUDE" \
    -I"$TORCH_INCLUDE" \
    -I"$TORCH_INCLUDE_CSRC" \
    -I"$FLASHINFER_INCLUDE" \
    -I/usr/local/cuda/include \
    -gencode arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH} \
    rope.cu \
    -L"$TORCH_LIB" \
    -ltorch -ltorch_python -lc10 -lc10_cuda \
    -o rope.so

echo "=============================================="
echo "Compilation successful!"
echo "Output: rope.so"
echo "=============================================="

python -c "
import torch
import sys
sys.path.insert(0, '.')
import rope
print('Available functions:', dir(rope))
print('Import test passed!')
"

