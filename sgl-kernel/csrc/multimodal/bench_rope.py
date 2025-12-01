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
except ImportError as e:
    rope_optimized = None
    HAS_ROPE_OPTIMIZED = False

try:
    from sgl_kernel import rotary_embedding as sgl_rotary_old
    HAS_SGL_KERNEL_OLD = True
except Exception:
    sgl_rotary_old = None
    HAS_SGL_KERNEL_OLD = False

HAS_SGL_KERNEL_EXISTING = False
try:
    import sgl_kernel
    if hasattr(torch.ops, 'sgl_kernel') and hasattr(torch.ops.sgl_kernel, 'apply_rope_pos_ids_cos_sin_cache'):
        HAS_SGL_KERNEL_EXISTING = True
except Exception:
    pass


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
        num_tokens = positions.shape[0]
        
        q = query.view(num_tokens, self.num_q_heads, self.head_size)
        k = key.view(num_tokens, self.num_kv_heads, self.head_size)
        
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        
        rope_optimized.apply_rope_pos_ids_cos_sin_cache(
            q, k, q_out, k_out,
            self.cos_sin_cache,
            positions.to(torch.int32),
            interleave=not self.is_neox_style,  # NEOX style = interleave False
        )
        
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
        
        q = query.view(num_tokens, self.num_q_heads, self.head_size)
        k = key.view(num_tokens, self.num_kv_heads, self.head_size)
        
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


class SGLKernelExistingRotaryEmbedding(nn.Module):
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
        if not HAS_SGL_KERNEL_EXISTING:
            raise RuntimeError("sgl_kernel not available")
        
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
        
        q = query.view(num_tokens, self.num_q_heads, self.head_size)
        k = key.view(num_tokens, self.num_kv_heads, self.head_size)
        
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        
        # Call sgl_kernel's existing implementation
        # requires int64 positions
        pos_i64 = positions.to(torch.int64)
        
        torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache(
            q, k, q_out, k_out, self.cos_sin_cache, pos_i64,
            not self.is_neox_style,  # interleave
            False,  # enable_pdl
            0,      # cuda_stream
            None,   # v
            None,   # k_buffer
            None,   # v_buffer
            None,   # kv_cache_loc
        )
        
        query_out = q_out.view(num_tokens, self.num_q_heads * self.head_size)
        key_out = k_out.view(num_tokens, self.num_kv_heads * self.head_size)
        return query_out, key_out


def get_available_providers():
    providers = ["flashinfer", "vllm_native", "vllm"]
    names = ["FlashInfer", "vLLM_Native", "vLLM"]
    
    if HAS_SGL_KERNEL_EXISTING:
        providers.append("sgl_existing")
        names.append("SGL_Existing")
    
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
        x_vals=[i for i in range(1, 1024) if i % 2 == 0],
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
        
    elif provider == "sgl_existing":
        rope = SGLKernelExistingRotaryEmbedding(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype,
            num_q_heads, num_kv_heads,
        ).to(device)
        rope_forward = rope.forward_cuda
        
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

    measurements = bench_gpu_time(lambda: rope_forward(pos_ids, query, key))
    ms = np.median(measurements)
    min_ms = np.percentile(measurements, 20)
    max_ms = np.percentile(measurements, 80)

    return ms, min_ms, max_ms


def correctness_test():
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
    
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(batch_size)
    query = torch.randn(batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device)
    
    fi_rope = FlashInferRotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype).to(device)
    q_ref = query.clone()
    k_ref = key.clone()
    fi_rope.forward_cuda(pos_ids, q_ref, k_ref)
    
    # SGLang 优化实现
    sgl_rope = SGLOptimizedRotaryEmbedding(
        head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype,
        num_q_heads, num_kv_heads,
    ).to(device)
    q_sgl, k_sgl = sgl_rope.forward_cuda(pos_ids, query.clone(), key.clone())
    
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
    
    providers, names = get_available_providers()
    print(f"Available providers: {providers}")
    print(f"FlashInfer: ✓")
    print(f"vLLM: ✓")
    print(f"SGL Optimized: {'✓' if HAS_ROPE_OPTIMIZED else '✗'}")
    print(f"SGL Old: {'✓' if HAS_SGL_KERNEL_OLD else '✗'}")
    
    correctness_test()
    
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    benchmark.run(print_data=True)
