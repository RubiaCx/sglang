"""
Benchmark RoPE for FlashInfer, vLLM, and SGLang optimized kernels.

Usage:
$ pip install vllm flashinfer
$ bash compile.sh  # 编译 rope.so
$ python bench_rope.py
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import triton
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding as vLLMRotaryEmbedding

from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
from flashinfer.testing.utils import bench_gpu_time

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

try:
    import rope as rope_optimized
    HAS_ROPE_OPTIMIZED = True
    print(f"✓ Loaded optimized rope module from {_here}")
except ImportError as e:
    rope_optimized = None
    HAS_ROPE_OPTIMIZED = False
    print(f"✗ Could not load optimized rope module: {e}")
    print("  Run 'bash compile.sh' first to build rope.so")

try:
    from sgl_kernel import rotary_embedding as sgl_rotary_old
    HAS_SGL_KERNEL_OLD = True
except Exception:
    sgl_rotary_old = None
    HAS_SGL_KERNEL_OLD = False


class FlashInferRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=self.is_neox_style,
        )
        return query, key


class SGLOptimizedRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        if not HAS_ROPE_OPTIMIZED:
            raise RuntimeError("rope module is not available; run 'bash compile.sh' first")
        
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style  # False = NEOX style (interleave=False)
        self.dtype = dtype
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache (fp32)."""
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 格式: [cos_0, cos_1, ..., cos_{d/2-1}, sin_0, sin_1, ..., sin_{d/2-1}]
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: [num_tokens], int32/int64
            query: [num_tokens, num_q_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        num_tokens = positions.shape[0]
        
        # Reshape to 3D: (num_tokens, num_heads, head_size)
        q = query.view(num_tokens, self.num_q_heads, self.head_size)
        k = key.view(num_tokens, self.num_kv_heads, self.head_size)
        
        # 输出 tensors
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        
        # 调用优化后的 kernel
        # interleave=False 对应 NEOX style (is_neox_style=True)
        rope_optimized.apply_rope_pos_ids_cos_sin_cache(
            q, k, q_out, k_out,
            self.cos_sin_cache,
            positions.to(torch.int32),
            interleave=not self.is_neox_style,  # NEOX style = interleave False
        )
        
        # 展平成与 vLLM / FlashInfer 一致的形状
        query_out = q_out.view(num_tokens, self.num_q_heads * self.head_size)
        key_out = k_out.view(num_tokens, self.num_kv_heads * self.head_size)
        return query_out, key_out


class SGLOptimizedRotaryEmbeddingInplace(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        if not HAS_ROPE_OPTIMIZED:
            raise RuntimeError("rope module is not available; run 'bash compile.sh' first")
        
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.shape[0]
        
        # Reshape to 3D: (num_tokens, num_heads, head_size)
        q = query.view(num_tokens, self.num_q_heads, self.head_size)
        k = key.view(num_tokens, self.num_kv_heads, self.head_size)
        
        # In-place: 输出和输入是同一个 tensor
        rope_optimized.apply_rope_pos_ids_cos_sin_cache(
            q, k, q, k,  # in-place
            self.cos_sin_cache,
            positions.to(torch.int32),
            interleave=not self.is_neox_style,
        )
        
        return query, key


class SGLOptimizedRotaryEmbeddingOnTheFly(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        if not HAS_ROPE_OPTIMIZED:
            raise RuntimeError("rope module is not available; run 'bash compile.sh' first")
        
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.shape[0]
        
        q = query.view(num_tokens, self.num_q_heads, self.head_size)
        k = key.view(num_tokens, self.num_kv_heads, self.head_size)
        
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        
        # 运行时计算 cos/sin
        rope_optimized.apply_rope_pos_ids(
            q, k, q_out, k_out,
            positions.to(torch.int32),
            self.rotary_dim,
            interleave=not self.is_neox_style,
            rope_scale=1.0,
            rope_theta=float(self.base),
        )
        
        query_out = q_out.view(num_tokens, self.num_q_heads * self.head_size)
        key_out = k_out.view(num_tokens, self.num_kv_heads * self.head_size)
        return query_out, key_out


def get_available_providers():
    """获取可用的 provider 列表。"""
    providers = ["flashinfer", "vllm_native", "vllm"]
    names = ["FlashInfer", "vLLM_Native", "vLLM"]
    
    if HAS_ROPE_OPTIMIZED:
        providers.extend(["sgl_opt", "sgl_opt_inplace", "sgl_opt_onthefly"])
        names.extend(["SGL_Optimized", "SGL_Opt_Inplace", "SGL_Opt_OnTheFly"])
    
    if HAS_SGL_KERNEL_OLD:
        providers.append("sgl_old")
        names.append("SGL_Old")
    
    return providers, names


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[
            2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
            2048, 4096, 8192, 16384, 32768, 65536,
        ],
        line_arg="provider",
        line_vals=get_available_providers()[0],
        line_names=get_available_providers()[1],
        styles=[
            ("blue", "-"), ("red", "-"), ("green", "-"),
            ("orange", "-"), ("purple", "-"), ("brown", "-"),
            ("black", "--"),
        ][:len(get_available_providers()[0])],
        ylabel="Latency (ms)",
        plot_name="rope-latency",
        args={
            "head_size": 128,
            "rotary_dim": 128,
            "max_position_embeddings": 65536,
            "base": 500000,
            "is_neox_style": True,
            "dtype": torch.bfloat16,
            "device": "cuda",
            "batch_size": 2,
            "num_q_heads": 32,
            "num_kv_heads": 8,
        },
    )
)
def benchmark(
    provider,
    head_size,
    rotary_dim,
    max_position_embeddings,
    base,
    is_neox_style,
    dtype,
    device,
    batch_size,
    seq_len,
    num_q_heads,
    num_kv_heads,
):
    print(
        f"provider: {provider}, seq_len: {seq_len}, "
        f"heads: {num_q_heads}/{num_kv_heads}, head_size: {head_size}"
    )

    rope_forward = None

    if provider == "vllm":
        rope = vLLMRotaryEmbedding(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        ).to(device)
        rope_forward = rope.forward_cuda
        
    elif provider == "flashinfer":
        rope = FlashInferRotaryEmbedding(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        ).to(device)
        rope_forward = rope.forward_cuda
        
    elif provider == "vllm_native":
        rope = vLLMRotaryEmbedding(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        ).to(device)
        rope_forward = rope.forward_native
        
    elif provider == "sgl_opt":
        rope = SGLOptimizedRotaryEmbedding(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype,
            num_q_heads, num_kv_heads,
        ).to(device)
        rope_forward = rope.forward_cuda
        
    elif provider == "sgl_opt_inplace":
        rope = SGLOptimizedRotaryEmbeddingInplace(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype,
            num_q_heads, num_kv_heads,
        ).to(device)
        rope_forward = rope.forward_cuda
        
    elif provider == "sgl_opt_onthefly":
        rope = SGLOptimizedRotaryEmbeddingOnTheFly(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype,
            num_q_heads, num_kv_heads,
        ).to(device)
        rope_forward = rope.forward_cuda
        
    elif provider == "sgl_old" and HAS_SGL_KERNEL_OLD:
        # 旧的 sgl_kernel 实现
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
        rope = RotaryEmbedding(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        ).to(device)
        rope_forward = rope.forward_cuda
    else:
        print(f"Unknown provider: {provider}")
        return 0, 0, 0

    # 创建输入
    pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
    query = torch.randn(
        batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device
    )

    # 测量性能
    measurements = bench_gpu_time(lambda: rope_forward(pos_ids, query, key))
    ms = np.median(measurements)
    min_ms = np.percentile(measurements, 20)
    max_ms = np.percentile(measurements, 80)

    return ms, min_ms, max_ms


def correctness_test():
    """正确性测试：对比各实现的输出。"""
    print("\n" + "=" * 60)
    print("Correctness Test")
    print("=" * 60)
    
    if not HAS_ROPE_OPTIMIZED:
        print("✗ Skipping correctness test: rope module not available")
        return
    
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 2
    seq_len = 128
    num_q_heads = 32
    num_kv_heads = 8
    head_size = 128
    rotary_dim = 128
    max_position_embeddings = 8192
    base = 10000
    is_neox_style = True
    
    # 创建输入
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(batch_size)
    query = torch.randn(batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device)
    
    # FlashInfer 参考实现
    fi_rope = FlashInferRotaryEmbedding(
        head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
    ).to(device)
    q_ref = query.clone()
    k_ref = key.clone()
    fi_rope.forward_cuda(pos_ids, q_ref, k_ref)
    
    # SGLang 优化实现
    sgl_rope = SGLOptimizedRotaryEmbedding(
        head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype,
        num_q_heads, num_kv_heads,
    ).to(device)
    q_sgl, k_sgl = sgl_rope.forward_cuda(pos_ids, query.clone(), key.clone())
    
    # 对比
    q_diff = (q_sgl - q_ref).abs().max().item()
    k_diff = (k_sgl - k_ref).abs().max().item()
    
    print(f"Q max diff: {q_diff:.6e}")
    print(f"K max diff: {k_diff:.6e}")
    
    tol = 1e-2
    if q_diff < tol and k_diff < tol:
        print("✓ Correctness test PASSED!")
    else:
        print("✗ Correctness test FAILED!")


if __name__ == "__main__":
    print("=" * 60)
    print("RoPE Benchmark")
    print("=" * 60)
    
    providers, names = get_available_providers()
    print(f"Available providers: {providers}")
    print(f"FlashInfer: ✓")
    print(f"vLLM: ✓")
    print(f"SGL Optimized: {'✓' if HAS_ROPE_OPTIMIZED else '✗'}")
    print(f"SGL Old: {'✓' if HAS_SGL_KERNEL_OLD else '✗'}")
    
    # 运行正确性测试
    correctness_test()
    
    # 运行性能测试
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    benchmark.run(print_data=True)
