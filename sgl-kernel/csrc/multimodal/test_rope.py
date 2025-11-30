"""
Test and benchmark RoPE kernels.
"""

import os
import sys
import torch
import numpy as np
from typing import Tuple

# Add current directory to path for importing the compiled module
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

try:
    import rope
    HAS_ROPE = True
except ImportError as e:
    print(f"Warning: Could not import rope: {e}")
    print("Please run compile.sh first")
    HAS_ROPE = False

from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace


def compute_cos_sin_cache(max_seq_len: int, rotary_dim: int, base: float = 10000.0) -> torch.Tensor:
    """Compute cos/sin cache for RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)  # (max_seq_len, rotary_dim)
    return cache


def reference_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation using FlashInfer."""
    q_out = q.clone()
    k_out = k.clone()
    head_size = q.size(-1)
    apply_rope_with_cos_sin_cache_inplace(
        positions=pos_ids,
        query=q_out,
        key=k_out,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )
    return q_out, k_out


def test_apply_rope_pos_ids_cos_sin_cache():
    """Test apply_rope_pos_ids_cos_sin_cache kernel."""
    print("\n" + "=" * 60)
    print("Testing apply_rope_pos_ids_cos_sin_cache...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    rotary_dim = head_dim
    max_seq_len = 8192
    dtype = torch.bfloat16
    device = "cuda"
    
    # Create inputs
    nnz = batch_size * seq_len
    q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(batch_size)
    
    cos_sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim).to(device)
    
    # Reference
    q_ref, k_ref = reference_rope(q.clone(), k.clone(), pos_ids, cos_sin_cache, is_neox=True)
    
    # Our optimized kernel
    q_test = q.clone()
    k_test = k.clone()
    q_out = torch.empty_like(q_test)
    k_out = torch.empty_like(k_test)
    
    rope.apply_rope_pos_ids_cos_sin_cache(
        q_test, k_test, q_out, k_out, cos_sin_cache, pos_ids, interleave=False
    )
    
    # Compare
    q_diff = (q_out - q_ref).abs().max().item()
    k_diff = (k_out - k_ref).abs().max().item()
    
    print(f"Q max diff: {q_diff:.6e}")
    print(f"K max diff: {k_diff:.6e}")
    
    # Check correctness
    tol = 1e-2
    assert q_diff < tol, f"Q diff {q_diff} exceeds tolerance {tol}"
    assert k_diff < tol, f"K diff {k_diff} exceeds tolerance {tol}"
    print("✓ Test passed!")


def test_apply_rope_pos_ids():
    """Test apply_rope_pos_ids kernel (compute cos/sin on-the-fly)."""
    print("\n" + "=" * 60)
    print("Testing apply_rope_pos_ids...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    rotary_dim = head_dim
    rope_theta = 10000.0
    max_seq_len = 8192
    dtype = torch.bfloat16
    device = "cuda"
    
    # Create inputs
    nnz = batch_size * seq_len
    q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(batch_size)
    
    cos_sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim, rope_theta).to(device)
    
    # Reference
    q_ref, k_ref = reference_rope(q.clone(), k.clone(), pos_ids, cos_sin_cache, is_neox=True)
    
    # Our optimized kernel
    q_test = q.clone()
    k_test = k.clone()
    q_out = torch.empty_like(q_test)
    k_out = torch.empty_like(k_test)
    
    rope.apply_rope_pos_ids(
        q_test, k_test, q_out, k_out, pos_ids, rotary_dim,
        interleave=False, rope_scale=1.0, rope_theta=rope_theta
    )
    
    # Compare
    q_diff = (q_out - q_ref).abs().max().item()
    k_diff = (k_out - k_ref).abs().max().item()
    
    print(f"Q max diff: {q_diff:.6e}")
    print(f"K max diff: {k_diff:.6e}")
    
    # Check correctness (slightly higher tolerance for on-the-fly computation)
    tol = 5e-2
    assert q_diff < tol, f"Q diff {q_diff} exceeds tolerance {tol}"
    assert k_diff < tol, f"K diff {k_diff} exceeds tolerance {tol}"
    print("✓ Test passed!")


def test_apply_rope_indptr():
    """Test apply_rope kernel with indptr and offsets."""
    print("\n" + "=" * 60)
    print("Testing apply_rope (indptr + offset)...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 4
    seq_lens = [32, 64, 128, 48]  # Variable sequence lengths
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    rotary_dim = head_dim
    rope_theta = 10000.0
    max_seq_len = 8192
    dtype = torch.bfloat16
    device = "cuda"
    
    # Create inputs
    nnz = sum(seq_lens)
    q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    
    # Create indptr and offsets
    indptr = torch.tensor([0] + list(np.cumsum(seq_lens)), dtype=torch.int32, device=device)
    offsets = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    # Create pos_ids for reference
    pos_ids = []
    for i, seq_len in enumerate(seq_lens):
        pos_ids.extend(range(seq_len))
    pos_ids = torch.tensor(pos_ids, dtype=torch.int32, device=device)
    
    cos_sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim, rope_theta).to(device)
    
    # Reference
    q_ref, k_ref = reference_rope(q.clone(), k.clone(), pos_ids, cos_sin_cache, is_neox=True)
    
    # Our optimized kernel
    q_test = q.clone()
    k_test = k.clone()
    q_out = torch.empty_like(q_test)
    k_out = torch.empty_like(k_test)
    
    rope.apply_rope(
        q_test, k_test, q_out, k_out, indptr, offsets, rotary_dim,
        interleave=False, rope_scale=1.0, rope_theta=rope_theta
    )
    
    # Compare
    q_diff = (q_out - q_ref).abs().max().item()
    k_diff = (k_out - k_ref).abs().max().item()
    
    print(f"Q max diff: {q_diff:.6e}")
    print(f"K max diff: {k_diff:.6e}")
    
    # Check correctness
    tol = 5e-2
    assert q_diff < tol, f"Q diff {q_diff} exceeds tolerance {tol}"
    assert k_diff < tol, f"K diff {k_diff} exceeds tolerance {tol}"
    print("✓ Test passed!")


def benchmark_kernels():
    """Benchmark different RoPE implementations."""
    print("\n" + "=" * 60)
    print("Benchmarking RoPE kernels...")
    print("=" * 60)
    
    from flashinfer.testing.utils import bench_gpu_time
    
    # Test parameters
    batch_size = 2
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    rotary_dim = head_dim
    max_seq_len = 65536
    dtype = torch.bfloat16
    device = "cuda"
    
    seq_lens = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    
    cos_sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim).to(device)
    
    results = []
    
    for seq_len in seq_lens:
        nnz = batch_size * seq_len
        q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
        pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(batch_size)
        
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        
        # FlashInfer
        q_fi = q.clone()
        k_fi = k.clone()
        fi_time = np.median(bench_gpu_time(
            lambda: apply_rope_with_cos_sin_cache_inplace(
                positions=pos_ids, query=q_fi, key=k_fi,
                head_size=head_dim, cos_sin_cache=cos_sin_cache, is_neox=True
            )
        ))
        
        # Our Optimized Kernel
        our_time = np.median(bench_gpu_time(
            lambda: rope.apply_rope_pos_ids_cos_sin_cache(
                q, k, q_out, k_out, cos_sin_cache, pos_ids, interleave=False
            )
        ))
        
        results.append({
            "seq_len": seq_len,
            "FlashInfer": fi_time,
            "Optimized": our_time,
            "Speedup": fi_time / our_time if our_time > 0 else 0,
        })
        
        print(f"seq_len={seq_len:5d}: FlashInfer={fi_time:.4f}ms, Optimized={our_time:.4f}ms, Speedup={fi_time/our_time:.2f}x")
    
    return results


if __name__ == "__main__":
    if not HAS_ROPE:
        print("Please compile the module first: bash compile.sh")
        sys.exit(1)
    
    # Run tests
    test_apply_rope_pos_ids_cos_sin_cache()
    test_apply_rope_pos_ids()
    test_apply_rope_indptr()
    
    # Run benchmark
    print("\n" + "=" * 60)
    print("All tests passed! Running benchmark...")
    print("=" * 60)
    benchmark_kernels()

