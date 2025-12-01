import os
import sys
import torch
import numpy as np
from typing import Tuple, List
import time

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

# Try to import sgl_kernel's existing implementation
HAS_SGL_KERNEL = False
sgl_apply_rope = None
try:
    from sgl_kernel.elementwise import apply_rope_with_cos_sin_cache_inplace as sgl_apply_rope
    HAS_SGL_KERNEL = True
except ImportError:
    try:
        import sgl_kernel
        from sgl_kernel.elementwise import apply_rope_with_cos_sin_cache_inplace as sgl_apply_rope
        HAS_SGL_KERNEL = True
    except ImportError:
        print("sgl_kernel not available")

from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
from flashinfer.testing.utils import bench_gpu_time


def compute_cos_sin_cache(max_seq_len: int, rotary_dim: int, base: float = 10000.0) -> torch.Tensor:
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


class TestResult:
    def __init__(self, name: str, passed: bool, q_diff: float = 0, k_diff: float = 0, details: str = ""):
        self.name = name
        self.passed = passed
        self.q_diff = q_diff
        self.k_diff = k_diff
        self.details = details


def test_apply_rope_pos_ids_cos_sin_cache(
    batch_size: int = 2,
    seq_len: int = 128,
    num_qo_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    tol: float = 1e-2,
) -> TestResult:
    device = "cuda"
    rotary_dim = head_dim
    max_seq_len = 8192
    
    nnz = batch_size * seq_len
    q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(batch_size)
    
    cos_sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim).to(device)
    
    # Reference
    q_ref, k_ref = reference_rope(q.clone(), k.clone(), pos_ids, cos_sin_cache, is_neox=True)
    
    # Our kernel
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    rope.apply_rope_pos_ids_cos_sin_cache(
        q, k, q_out, k_out, cos_sin_cache, pos_ids, interleave=False
    )
    
    q_diff = (q_out - q_ref).abs().max().item()
    k_diff = (k_out - k_ref).abs().max().item()
    
    passed = q_diff < tol and k_diff < tol
    return TestResult(
        f"cos_sin_cache [bs={batch_size}, seq={seq_len}, heads={num_qo_heads}/{num_kv_heads}, dim={head_dim}, {dtype}]",
        passed, q_diff, k_diff
    )


def test_apply_rope_pos_ids(
    batch_size: int = 2,
    seq_len: int = 128,
    num_qo_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    rope_theta: float = 10000.0,
    tol: float = 5e-2,
) -> TestResult:
    """Test apply_rope_pos_ids kernel (on-the-fly cos/sin)."""
    device = "cuda"
    rotary_dim = head_dim
    max_seq_len = 8192
    
    nnz = batch_size * seq_len
    q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(batch_size)
    
    cos_sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim, rope_theta).to(device)
    
    # Reference
    q_ref, k_ref = reference_rope(q.clone(), k.clone(), pos_ids, cos_sin_cache, is_neox=True)
    
    # Our kernel
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    rope.apply_rope_pos_ids(
        q, k, q_out, k_out, pos_ids, rotary_dim,
        interleave=False, rope_scale=1.0, rope_theta=rope_theta
    )
    
    q_diff = (q_out - q_ref).abs().max().item()
    k_diff = (k_out - k_ref).abs().max().item()
    
    passed = q_diff < tol and k_diff < tol
    return TestResult(
        f"pos_ids [bs={batch_size}, seq={seq_len}, heads={num_qo_heads}/{num_kv_heads}, dim={head_dim}, theta={rope_theta}]",
        passed, q_diff, k_diff
    )


def test_apply_rope_indptr(
    seq_lens: List[int] = [32, 64, 128, 48],
    num_qo_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    rope_theta: float = 10000.0,
    tol: float = 5e-2,
) -> TestResult:
    """Test apply_rope with indptr and offsets."""
    device = "cuda"
    rotary_dim = head_dim
    max_seq_len = 8192
    batch_size = len(seq_lens)
    
    nnz = sum(seq_lens)
    q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    
    indptr = torch.tensor([0] + list(np.cumsum(seq_lens)), dtype=torch.int32, device=device)
    offsets = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    pos_ids = []
    for seq_len in seq_lens:
        pos_ids.extend(range(seq_len))
    pos_ids = torch.tensor(pos_ids, dtype=torch.int32, device=device)
    
    cos_sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim, rope_theta).to(device)
    
    q_ref, k_ref = reference_rope(q.clone(), k.clone(), pos_ids, cos_sin_cache, is_neox=True)
    
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    rope.apply_rope(
        q, k, q_out, k_out, indptr, offsets, rotary_dim,
        interleave=False, rope_scale=1.0, rope_theta=rope_theta
    )
    
    q_diff = (q_out - q_ref).abs().max().item()
    k_diff = (k_out - k_ref).abs().max().item()
    
    passed = q_diff < tol and k_diff < tol
    return TestResult(
        f"indptr [seq_lens={seq_lens}, heads={num_qo_heads}/{num_kv_heads}, dim={head_dim}]",
        passed, q_diff, k_diff
    )


def test_inplace_operation() -> TestResult:
    """Test in-place RoPE operation (q_out = q, k_out = k)."""
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 2
    seq_len = 64
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    rotary_dim = head_dim
    max_seq_len = 8192
    tol = 1e-2
    
    nnz = batch_size * seq_len
    q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(batch_size)
    
    cos_sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim).to(device)
    
    # Reference (not in-place)
    q_ref, k_ref = reference_rope(q.clone(), k.clone(), pos_ids, cos_sin_cache, is_neox=True)
    
    # In-place operation
    q_inplace = q.clone()
    k_inplace = k.clone()
    rope.apply_rope_pos_ids_cos_sin_cache(
        q_inplace, k_inplace, q_inplace, k_inplace,  # in-place
        cos_sin_cache, pos_ids, interleave=False
    )
    
    q_diff = (q_inplace - q_ref).abs().max().item()
    k_diff = (k_inplace - k_ref).abs().max().item()
    
    passed = q_diff < tol and k_diff < tol
    return TestResult("in-place operation", passed, q_diff, k_diff)


def test_sgl_kernel_existing(
    batch_size: int = 2,
    seq_len: int = 128,
    num_qo_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    tol: float = 1e-2,
) -> TestResult:
    if not HAS_SGL_KERNEL:
        return TestResult("sgl_kernel (not available)", False, 0, 0, "sgl_kernel not installed")
    
    device = "cuda"
    rotary_dim = head_dim
    max_seq_len = 8192
    
    nnz = batch_size * seq_len
    q = torch.randn(nnz, num_qo_heads * head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, num_kv_heads * head_dim, dtype=dtype, device=device)
    pos_ids = torch.arange(seq_len, device=device, dtype=torch.int64).repeat(batch_size)
    
    cos_sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim).to(device)
    
    q_3d = q.view(nnz, num_qo_heads, head_dim)
    k_3d = k.view(nnz, num_kv_heads, head_dim)
    q_ref, k_ref = reference_rope(q_3d.clone(), k_3d.clone(), pos_ids.to(torch.int32), cos_sin_cache, is_neox=True)
    q_ref = q_ref.view(nnz, num_qo_heads * head_dim)
    k_ref = k_ref.view(nnz, num_kv_heads * head_dim)
    
    # sgl_kernel implementation (in-place)
    q_sgl = q.clone()
    k_sgl = k.clone()
    
    sgl_apply_rope(
        positions=pos_ids,
        query=q_sgl,
        key=k_sgl,
        head_size=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
    )
    
    q_diff = (q_sgl - q_ref).abs().max().item()
    k_diff = (k_sgl - k_ref).abs().max().item()
    
    passed = q_diff < tol and k_diff < tol
    return TestResult(
        f"sgl_kernel [bs={batch_size}, seq={seq_len}, heads={num_qo_heads}/{num_kv_heads}, dim={head_dim}]",
        passed, q_diff, k_diff
    )


def test_different_dtypes() -> List[TestResult]:
    results = []
    for dtype in [torch.float16, torch.bfloat16]:
        result = test_apply_rope_pos_ids_cos_sin_cache(dtype=dtype)
        results.append(result)
    return results


def test_different_head_dims() -> List[TestResult]:
    results = []
    for head_dim in [64, 128, 256]:
        result = test_apply_rope_pos_ids_cos_sin_cache(head_dim=head_dim)
        results.append(result)
    return results


def test_different_seq_lens() -> List[TestResult]:
    """Test with different sequence lengths."""
    results = []
    for seq_len in [1, 2, 16, 128, 1024, 4096]:
        result = test_apply_rope_pos_ids_cos_sin_cache(seq_len=seq_len)
        results.append(result)
    return results


def test_different_head_counts() -> List[TestResult]:
    results = []
    configs = [
        (8, 8),    # MHA
        (32, 8),   # GQA
        (32, 1),   # MQA
        (64, 8),   # Large GQA
    ]
    for num_qo_heads, num_kv_heads in configs:
        result = test_apply_rope_pos_ids_cos_sin_cache(
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads
        )
        results.append(result)
    return results


def test_different_rope_theta() -> List[TestResult]:
    """Test with different rope_theta values."""
    results = []
    for theta in [10000.0, 500000.0, 1000000.0]:
        result = test_apply_rope_pos_ids(rope_theta=theta)
        results.append(result)
    return results


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ROPE KERNEL TESTS")
    print("=" * 80)
    
    all_results = []
    
    # Basic functionality tests
    print("\n[1] Basic Functionality Tests")
    print("-" * 40)
    
    result = test_apply_rope_pos_ids_cos_sin_cache()
    all_results.append(result)
    print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    result = test_apply_rope_pos_ids()
    all_results.append(result)
    print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    result = test_apply_rope_indptr()
    all_results.append(result)
    print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    result = test_inplace_operation()
    all_results.append(result)
    print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    # Data type tests
    print("\n[2] Data Type Tests")
    print("-" * 40)
    for result in test_different_dtypes():
        all_results.append(result)
        print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    # Head dimension tests
    print("\n[3] Head Dimension Tests")
    print("-" * 40)
    for result in test_different_head_dims():
        all_results.append(result)
        print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    # Sequence length tests
    print("\n[4] Sequence Length Tests")
    print("-" * 40)
    for result in test_different_seq_lens():
        all_results.append(result)
        print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    # Head count tests
    print("\n[5] Head Configuration Tests (MHA/GQA/MQA)")
    print("-" * 40)
    for result in test_different_head_counts():
        all_results.append(result)
        print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    # Rope theta tests
    print("\n[6] Rope Theta Tests")
    print("-" * 40)
    for result in test_different_rope_theta():
        all_results.append(result)
        print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    # Variable length batch tests
    print("\n[7] Variable Length Batch Tests")
    print("-" * 40)
    
    test_cases = [
        [1, 1, 1, 1],           # All length 1
        [128, 128, 128, 128],   # All same length
        [1, 64, 256, 1024],     # Mixed lengths
        [7, 13, 29, 41],        # Prime numbers
    ]
    for seq_lens in test_cases:
        result = test_apply_rope_indptr(seq_lens=seq_lens)
        all_results.append(result)
        print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    # sgl_kernel existing implementation tests
    print("\n[8] sgl_kernel Existing Implementation Tests")
    print("-" * 40)
    
    if HAS_SGL_KERNEL:
        result = test_sgl_kernel_existing()
        all_results.append(result)
        print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
        
        # Test with different configurations
        for head_dim in [64, 128, 256]:
            result = test_sgl_kernel_existing(head_dim=head_dim)
            all_results.append(result)
            print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
        
        for seq_len in [1, 16, 128, 1024]:
            result = test_sgl_kernel_existing(seq_len=seq_len)
            all_results.append(result)
            print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    else:
        print("  ⚠ sgl_kernel not available, skipping tests")
    
    # Summary
    print("\n" + "=" * 80)
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    print(f"SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED:")
        for r in all_results:
            if not r.passed:
                print(f"  - {r.name}")
    
    print("=" * 80)
    return all_results


def benchmark_comparison():
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK: Optimized RoPE vs FlashInfer vs sgl_kernel")
    print("=" * 80)
    
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 2
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    rotary_dim = head_dim
    max_seq_len = 65536
    
    seq_lens = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    
    cos_sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim).to(device)
    
    print(f"\nConfig: batch_size={batch_size}, heads={num_qo_heads}/{num_kv_heads}, head_dim={head_dim}, dtype={dtype}")
    print("-" * 100)
    
    header = f"{'seq_len':>8} | {'FlashInfer':>12} | {'sgl_kernel':>12} | {'Optimized':>12} | {'Opt_Inplace':>12} | {'vs_FI':>8} | {'vs_SGL':>8}"
    print(header)
    print("-" * 100)
    
    results = []
    
    for seq_len in seq_lens:
        nnz = batch_size * seq_len
        q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
        pos_ids = torch.arange(seq_len, device=device, dtype=torch.int32).repeat(batch_size)
        pos_ids_i64 = pos_ids.to(torch.int64)
        
        q_2d = torch.randn(nnz, num_qo_heads * head_dim, dtype=dtype, device=device)
        k_2d = torch.randn(nnz, num_kv_heads * head_dim, dtype=dtype, device=device)
        
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
        
        # sgl_kernel existing implementation
        sgl_time = float('nan')
        if HAS_SGL_KERNEL:
            q_sgl = q_2d.clone()
            k_sgl = k_2d.clone()
            sgl_time = np.median(bench_gpu_time(
                lambda: sgl_apply_rope(
                    positions=pos_ids_i64, query=q_sgl, key=k_sgl,
                    head_size=head_dim, cos_sin_cache=cos_sin_cache, is_neox=True
                )
            ))
        
        # Optimized (out-of-place)
        opt_time = np.median(bench_gpu_time(
            lambda: rope.apply_rope_pos_ids_cos_sin_cache(
                q, k, q_out, k_out, cos_sin_cache, pos_ids, interleave=False
            )
        ))
        
        # Optimized (in-place)
        q_ip = q.clone()
        k_ip = k.clone()
        opt_ip_time = np.median(bench_gpu_time(
            lambda: rope.apply_rope_pos_ids_cos_sin_cache(
                q_ip, k_ip, q_ip, k_ip, cos_sin_cache, pos_ids, interleave=False
            )
        ))
        
        speedup_fi = fi_time / opt_ip_time if opt_ip_time > 0 else 0
        speedup_sgl = sgl_time / opt_ip_time if opt_ip_time > 0 and not np.isnan(sgl_time) else float('nan')
        
        sgl_time_str = f"{sgl_time:>10.4f}ms" if not np.isnan(sgl_time) else "     N/A   "
        speedup_sgl_str = f"{speedup_sgl:>7.2f}x" if not np.isnan(speedup_sgl) else "    N/A"
        
        print(f"{seq_len:>8} | {fi_time:>10.4f}ms | {sgl_time_str} | {opt_time:>10.4f}ms | {opt_ip_time:>10.4f}ms | {speedup_fi:>7.2f}x | {speedup_sgl_str}")
        
        results.append({
            "seq_len": seq_len,
            "FlashInfer": fi_time,
            "sgl_kernel": sgl_time,
            "Optimized": opt_time,
            "Opt_Inplace": opt_ip_time,
            "Speedup_FI": speedup_fi,
            "Speedup_SGL": speedup_sgl,
        })
    
    print("-" * 100)
    
    # Summary statistics
    avg_speedup_fi = np.mean([r["Speedup_FI"] for r in results])
    print(f"\nAverage speedup vs FlashInfer (in-place): {avg_speedup_fi:.2f}x")
    
    if HAS_SGL_KERNEL:
        valid_sgl = [r["Speedup_SGL"] for r in results if not np.isnan(r["Speedup_SGL"])]
        if valid_sgl:
            avg_speedup_sgl = np.mean(valid_sgl)
            print(f"Average speedup vs sgl_kernel (in-place): {avg_speedup_sgl:.2f}x")
    
    # Best/worst cases vs FlashInfer
    best = max(results, key=lambda x: x["Speedup_FI"])
    worst = min(results, key=lambda x: x["Speedup_FI"])
    print(f"Best speedup vs FI: {best['Speedup_FI']:.2f}x at seq_len={best['seq_len']}")
    print(f"Worst speedup vs FI: {worst['Speedup_FI']:.2f}x at seq_len={worst['seq_len']}")
    
    return results


if __name__ == "__main__":
    if not HAS_ROPE:
        print("Please compile the module first: bash compile.sh")
        sys.exit(1)
    
    run_comprehensive_tests()
    benchmark_comparison()
    
