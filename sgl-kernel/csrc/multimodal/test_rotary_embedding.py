import os
import sys
from typing import Tuple, Optional

import torch
import numpy as np

# Add current directory to path
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# Import compiled module
try:
    import rotary_embedding as rotary_emb_module
    HAS_ROTARY_EMBEDDING = True
except ImportError as e:
    print("  Run 'bash compile_rotary_embedding.sh' first")
    HAS_ROTARY_EMBEDDING = False

# Try to import vLLM for comparison
try:
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding as vLLMRotaryEmbedding
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("vLLM not available")

# Try to import sgl_kernel for comparison
try:
    from sgl_kernel.rotary_embedding import rotary_embedding as sgl_rotary_embedding
    HAS_SGL_KERNEL = True
    print("✓ sgl_kernel.rotary_embedding available")
except ImportError:
    HAS_SGL_KERNEL = False
    print("⚠ sgl_kernel.rotary_embedding not available")


def compute_cos_sin_cache(
    max_seq_len: int,
    rotary_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute separate cos and sin caches."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin  # (max_seq_len, rotary_dim/2)


def reference_rotary_neox(
    query: torch.Tensor,  # (num_tokens, num_heads * head_size)
    key: torch.Tensor,    # (num_tokens, num_kv_heads * head_size)
    cos: torch.Tensor,    # (num_tokens, rotary_dim/2)
    sin: torch.Tensor,    # (num_tokens, rotary_dim/2)
    head_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of NeoX-style rotary embedding."""
    num_tokens = query.size(0)
    num_heads = query.size(1) // head_size
    num_kv_heads = key.size(1) // head_size
    rotary_dim = cos.size(1) * 2
    
    q = query.view(num_tokens, num_heads, head_size).float()
    k = key.view(num_tokens, num_kv_heads, head_size).float()
    
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]
    
    q_rot = q_rot.view(num_tokens, num_heads, rotary_dim // 2, 2)
    k_rot = k_rot.view(num_tokens, num_kv_heads, rotary_dim // 2, 2)
    
    cos_expanded = cos.float().unsqueeze(1)
    sin_expanded = sin.float().unsqueeze(1)
    
    # Apply rotary: x' = x*cos - y*sin, y' = y*cos + x*sin
    q_x = q_rot[..., 0] 
    q_y = q_rot[..., 1]
    q_rot_out = torch.stack([q_x * cos_expanded - q_y * sin_expanded, q_y * cos_expanded + q_x * sin_expanded], dim=-1)
    
    k_x = k_rot[..., 0]
    k_y = k_rot[..., 1]
    k_rot_out = torch.stack([k_x * cos_expanded - k_y * sin_expanded, k_y * cos_expanded + k_x * sin_expanded], dim=-1)
    
    q_rot_out = q_rot_out.view(num_tokens, num_heads, rotary_dim)
    k_rot_out = k_rot_out.view(num_tokens, num_kv_heads, rotary_dim)
    
    q_out = torch.cat([q_rot_out, q_pass], dim=-1)
    k_out = torch.cat([k_rot_out, k_pass], dim=-1)
    
    q_out = q_out.view(num_tokens, num_heads * head_size).to(query.dtype)
    k_out = k_out.view(num_tokens, num_kv_heads * head_size).to(key.dtype)
    
    return q_out, k_out


class TestResult:
    def __init__(self, name: str, passed: bool, q_diff: float = 0, k_diff: float = 0, details: str = ""):
        self.name = name
        self.passed = passed
        self.q_diff = q_diff
        self.k_diff = k_diff
        self.details = details

def test_basic_correctness(
    batch_size: int = 2,
    seq_len: int = 128,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    tol: float = 1e-2,
) -> TestResult:
    device = "cuda"
    rotary_dim = head_size
    max_seq_len = 8192
    
    num_tokens = batch_size * seq_len
    
    query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)
    
    cos_cache, sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim, dtype=dtype)
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)
    
    positions = torch.arange(seq_len, device=device).repeat(batch_size)
    cos = cos_cache[positions] 
    sin = sin_cache[positions]
    
    q_ref, k_ref = reference_rotary_neox(query.clone(), key.clone(), cos, sin, head_size)
    
    q_out = query.clone()
    k_out = key.clone()
    rotary_emb_module.rotary_embedding(cos, sin, q_out, k_out, head_size, True)  # is_neox=True
    
    q_diff = (q_out - q_ref).abs().max().item()
    k_diff = (k_out - k_ref).abs().max().item()
    
    passed = q_diff < tol and k_diff < tol
    return TestResult(
        f"basic [bs={batch_size}, seq={seq_len}, heads={num_heads}/{num_kv_heads}, dim={head_size}, {dtype}]",
        passed, q_diff, k_diff
    )


def test_vs_sgl_kernel(
    batch_size: int = 2,
    seq_len: int = 128,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    tol: float = 1e-2,
) -> TestResult:
    if not HAS_SGL_KERNEL:
        return TestResult("vs_sgl_kernel (skipped)", True, 0, 0, "sgl_kernel not available")
    
    device = "cuda"
    rotary_dim = head_size
    max_seq_len = 8192
    
    num_tokens = batch_size * seq_len
    
    query_orig = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device=device)
    key_orig = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)
    
    cos_cache, sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim, dtype=dtype)
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)
    
    positions = torch.arange(seq_len, device=device).repeat(batch_size)
    cos = cos_cache[positions]
    sin = sin_cache[positions]
    
    q_sgl = query_orig.clone()
    k_sgl = key_orig.clone()
    sgl_rotary_embedding(cos, sin, q_sgl, k_sgl, head_size, True)  # is_neox=True
    
    q_out = query_orig.clone()
    k_out = key_orig.clone()
    rotary_emb_module.rotary_embedding(cos, sin, q_out, k_out, head_size, True)
    
    q_diff = (q_out - q_sgl).abs().max().item()
    k_diff = (k_out - k_sgl).abs().max().item()
    
    passed = q_diff < tol and k_diff < tol
    return TestResult(
        f"vs_sgl_kernel [bs={batch_size}, seq={seq_len}, heads={num_heads}/{num_kv_heads}]",
        passed, q_diff, k_diff
    )


def test_different_configs():
    results = []
    
    for head_size in [64, 128, 256]:
        result = test_basic_correctness(head_size=head_size)
        results.append(result)
    
    for num_heads, num_kv_heads in [(8, 8), (32, 8), (32, 1), (64, 8)]:
        result = test_basic_correctness(num_heads=num_heads, num_kv_heads=num_kv_heads)
        results.append(result)
    
    for seq_len in [1, 16, 128, 1024]:
        result = test_basic_correctness(seq_len=seq_len)
        results.append(result)
    
    for dtype in [torch.float16, torch.bfloat16]:
        result = test_basic_correctness(dtype=dtype)
        results.append(result)
    
    return results


def benchmark_comparison():
    import time
    
    print("\n" + "=" * 100)
    print("PERFORMANCE BENCHMARK: rotary_embedding")
    print("=" * 100)
    
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 1
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    rotary_dim = head_size
    max_seq_len = 65536
    
    seq_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    cos_cache, sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim, dtype=dtype)
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)
    
    vllm_rope = None
    if HAS_VLLM:
        vllm_rope = vLLMRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_seq_len,
            base=10000,
            is_neox_style=True,
            dtype=dtype,
        ).cuda()
    
    print(f"\nConfig: batch_size={batch_size}, heads={num_heads}/{num_kv_heads}, head_size={head_size}, dtype={dtype}")
    print("-" * 100)
    
    header = f"{'seq_len':>8}"
    if HAS_ROTARY_EMBEDDING:
        header += f" | {'Ours':>12}"
    if HAS_VLLM:
        header += f" | {'vLLM':>12}"
    if HAS_SGL_KERNEL:
        header += f" | {'sgl_kernel':>12}"
    if HAS_ROTARY_EMBEDDING and HAS_VLLM:
        header += f" | {'vs_vLLM':>8}"
    print(header)
    print("-" * 100)
    
    results = []
    
    for seq_len in seq_lens:
        num_tokens = batch_size * seq_len
        query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device=device)
        key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)
        
        positions = torch.arange(seq_len, device=device, dtype=torch.int64).repeat(batch_size)
        cos = cos_cache[positions]
        sin = sin_cache[positions]
        
        row_str = f"{seq_len:>8}"
        ours_time = None
        vllm_time = None
        sgl_time = None
        
        if HAS_ROTARY_EMBEDDING:
            for _ in range(10):
                q = query.clone()
                k = key.clone()
                rotary_emb_module.rotary_embedding(cos, sin, q, k, head_size, True)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                q = query.clone()
                k = key.clone()
                rotary_emb_module.rotary_embedding(cos, sin, q, k, head_size, True)
            torch.cuda.synchronize()
            ours_time = (time.perf_counter() - start) / 100 * 1000  # ms
            row_str += f" | {ours_time:>10.4f}ms"
        
        if HAS_VLLM and vllm_rope is not None:
            for _ in range(10):
                q = query.clone()
                k = key.clone()
                vllm_rope.forward_cuda(positions, q, k)
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(100):
                q = query.clone()
                k = key.clone()
                vllm_rope.forward_cuda(positions, q, k)
            torch.cuda.synchronize()
            vllm_time = (time.perf_counter() - start) / 100 * 1000  # ms
            row_str += f" | {vllm_time:>10.4f}ms"
        
        if HAS_SGL_KERNEL:
            for _ in range(10):
                q = query.clone()
                k = key.clone()
                sgl_rotary_embedding(cos, sin, q, k, head_size, True)
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(100):
                q = query.clone()
                k = key.clone()
                sgl_rotary_embedding(cos, sin, q, k, head_size, True)
            torch.cuda.synchronize()
            sgl_time = (time.perf_counter() - start) / 100 * 1000  # ms
            row_str += f" | {sgl_time:>10.4f}ms"
        
        if ours_time and vllm_time:
            speedup = vllm_time / ours_time
            row_str += f" | {speedup:>7.2f}x"
        
        print(row_str)
        results.append({
            "seq_len": seq_len,
            "ours": ours_time,
            "vllm": vllm_time,
            "sgl_kernel": sgl_time,
        })
    
    print("-" * 100)
    
    print("\nSUMMARY:")
    if HAS_ROTARY_EMBEDDING:
        avg_ours = np.mean([r["ours"] for r in results if r["ours"]])
        print(f"  Ours avg: {avg_ours:.4f}ms")
    if HAS_VLLM:
        avg_vllm = np.mean([r["vllm"] for r in results if r["vllm"]])
        print(f"  vLLM avg: {avg_vllm:.4f}ms")
    if HAS_SGL_KERNEL:
        avg_sgl = np.mean([r["sgl_kernel"] for r in results if r["sgl_kernel"]])
        print(f"  sgl_kernel avg: {avg_sgl:.4f}ms")
    
    if HAS_ROTARY_EMBEDDING and HAS_VLLM:
        speedups = [r["vllm"] / r["ours"] for r in results if r["ours"] and r["vllm"]]
        if speedups:
            print(f"\n  Average speedup vs vLLM: {np.mean(speedups):.2f}x")


def run_all_tests():
    """Run all tests."""
    if not HAS_ROTARY_EMBEDDING:
        print("Please compile the module first: bash compile_rotary_embedding.sh")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("ROTARY EMBEDDING TESTS (vLLM-style, separate cos/sin)")
    print("=" * 80)
    
    all_results = []
    
    print("\n[1] Basic Correctness Tests")
    print("-" * 40)
    result = test_basic_correctness()
    all_results.append(result)
    print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    print("\n[2] vs sgl_kernel Test")
    print("-" * 40)
    result = test_vs_sgl_kernel()
    all_results.append(result)
    print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    print("\n[3] Configuration Tests")
    print("-" * 40)
    for result in test_different_configs():
        all_results.append(result)
        print(f"  {'✓' if result.passed else '✗'} {result.name}: Q={result.q_diff:.2e}, K={result.k_diff:.2e}")
    
    print("\n" + "=" * 80)
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    run_all_tests()
    benchmark_comparison()

