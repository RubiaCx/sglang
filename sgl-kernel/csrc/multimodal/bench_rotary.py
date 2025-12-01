"""
Benchmark: Compare RotaryEmbedding implementations between vLLM and SGLang.

Compares:
1. vLLM RotaryEmbedding
2. sgl_kernel RotaryEmbedding (new, rope.py)
3. sgl_kernel rotary_embedding function (old, rotary_embedding.py)
4. FlashInfer (reference)

Usage:
    python -m sgl_kernel.bench_rotary
    # or
    python sgl_kernel/bench_rotary.py
"""

import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding as vLLMRotaryEmbedding
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace as fi_apply_rope
    from flashinfer.testing.utils import bench_gpu_time
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    print("⚠ FlashInfer not available (pip install flashinfer)")

def simple_bench_gpu_time(fn, warmup=10, repeat=100):
    import time
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    return times

# sgl_kernel new RotaryEmbedding
try:
    from sgl_kernel.rope import RotaryEmbedding as SGLRotaryEmbedding
    HAS_SGL_NEW = True
except ImportError:
    HAS_SGL_NEW = False
    print("⚠ sgl_kernel.rope not available")

# sgl_kernel old rotary_embedding function
try:
    from sgl_kernel.rotary_embedding import rotary_embedding as sgl_rotary_old
    HAS_SGL_OLD = True
except ImportError:
    HAS_SGL_OLD = False


class FlashInferRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int = 10000,
        is_neox_style: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        
        cache = self._compute_cos_sin_cache()
        self.register_buffer("cos_sin_cache", cache, persistent=False)
    
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return torch.cat((cos, sin), dim=-1)
    
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fi_apply_rope(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=self.is_neox_style,
        )
        return query, key


class SGLvLLMStyleRotaryEmbedding(nn.Module):
    """Wrapper for sgl_kernel's rotary_embedding function (vLLM-compatible API).
    
    This uses the SAME cos/sin format as vLLM: separate cos and sin tensors.
    sgl_kernel.rotary_embedding is adapted from vLLM's implementation.
    """
    
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int = 10000,
        is_neox_style: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        
        cos, sin = self._compute_cos_sin()
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)
    
    def _compute_cos_sin(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin cache - same format as vLLM."""
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin
    
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gather cos/sin for current positions - same as vLLM
        cos = self.cos_cache[positions]
        sin = self.sin_cache[positions]
        
        # sgl_kernel.rotary_embedding has the SAME signature as vLLM
        sgl_rotary_old(
            cos=cos,
            sin=sin,
            query=query,
            key=key,
            head_size=self.head_size,
            is_neox=self.is_neox_style,
        )
        return query, key



def benchmark_single(
    name: str,
    rope_module: nn.Module,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    warmup: int = 10,
    repeat: int = 100,
) -> float:
    """Benchmark a single RoPE implementation."""
    # Use FlashInfer's bench_gpu_time if available, otherwise use fallback
    bench_fn = bench_gpu_time if HAS_FLASHINFER else simple_bench_gpu_time
    
    # bench_gpu_time takes just the lambda
    times = bench_fn(
        lambda: rope_module.forward_cuda(positions, query.clone(), key.clone())
    )
    return np.median(times)


def run_benchmark(
    batch_size: int = 1,
    seq_lens: list = None,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    base: int = 10000,
    is_neox_style: bool = True,
    warmup: int = 10,
    repeat: int = 100,
):
    """Run benchmark comparison."""
    if seq_lens is None:
        seq_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    device = "cuda"
    max_seq_len = max(seq_lens) * 2
    rotary_dim = head_size
    
    print("\n" + "=" * 100)
    print("ROTARY EMBEDDING BENCHMARK: vLLM vs SGLang")
    print("=" * 100)
    print(f"\nConfig: batch_size={batch_size}, heads={num_q_heads}/{num_kv_heads}, "
          f"head_size={head_size}, dtype={dtype}, base={base}")
    print("-" * 100)
    
    # Available implementations
    implementations = {}
    
    if HAS_VLLM:
        vllm_rope = vLLMRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_seq_len,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        ).cuda()
        implementations["vLLM"] = vllm_rope
    
    if HAS_FLASHINFER:
        fi_rope = FlashInferRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_seq_len,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        ).cuda()
        implementations["FlashInfer"] = fi_rope
    
    if HAS_SGL_NEW:
        sgl_new_rope = SGLRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_seq_len,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        ).cuda()
        implementations["SGL_New"] = sgl_new_rope
    
    if HAS_SGL_OLD:
        sgl_old_rope = SGLOldRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_seq_len,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        ).cuda()
        implementations["SGL_Old"] = sgl_old_rope
    
    if not implementations:
        print("No implementations available!")
        return
    
    # Print header
    header = f"{'seq_len':>8}"
    for name in implementations:
        header += f" | {name:>12}"
    if "vLLM" in implementations and "SGL_New" in implementations:
        header += f" | {'vs_vLLM':>8}"
    print(header)
    print("-" * 100)
    
    results = []
    
    for seq_len in seq_lens:
        num_tokens = batch_size * seq_len
        
        # Create inputs (2D format: num_tokens, num_heads * head_size)
        query = torch.randn(num_tokens, num_q_heads * head_size, dtype=dtype, device=device)
        key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)
        positions = torch.arange(seq_len, device=device, dtype=torch.int64).repeat(batch_size)
        
        row_data = {"seq_len": seq_len}
        row_str = f"{seq_len:>8}"
        
        for name, rope_module in implementations.items():
            try:
                time_ms = benchmark_single(
                    name, rope_module, positions, query, key,
                    warmup=warmup, repeat=repeat
                )
                row_data[name] = time_ms
                row_str += f" | {time_ms:>10.4f}ms"
            except Exception as e:
                row_data[name] = float('nan')
                row_str += f" | {'ERROR':>12}"
                print(f"  Warning: {name} failed: {e}")
        
        # Speedup vs vLLM
        if "vLLM" in row_data and "SGL_New" in row_data:
            if not np.isnan(row_data["vLLM"]) and not np.isnan(row_data["SGL_New"]):
                speedup = row_data["vLLM"] / row_data["SGL_New"]
                row_str += f" | {speedup:>7.2f}x"
            else:
                row_str += f" | {'N/A':>8}"
        
        print(row_str)
        results.append(row_data)
    
    print("-" * 100)
    
    # Summary
    print("\nSUMMARY:")
    for name in implementations:
        valid_times = [r[name] for r in results if not np.isnan(r.get(name, float('nan')))]
        if valid_times:
            avg_time = np.mean(valid_times)
            print(f"  {name}: avg={avg_time:.4f}ms")
    
    if "vLLM" in implementations and "SGL_New" in implementations:
        speedups = []
        for r in results:
            if not np.isnan(r.get("vLLM", float('nan'))) and not np.isnan(r.get("SGL_New", float('nan'))):
                speedups.append(r["vLLM"] / r["SGL_New"])
        if speedups:
            print(f"\n  Average speedup (SGL_New vs vLLM): {np.mean(speedups):.2f}x")
    
    return results


def test_correctness():
    """Test correctness between implementations.
    
    Note: vLLM and FlashInfer may have slight differences due to:
    1. Different cos/sin cache formats (separate vs combined)
    2. Different precision in intermediate computations
    
    We use a relaxed tolerance for cross-implementation comparison.
    """
    print("\n" + "=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)
    
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 2
    seq_len = 128
    num_q_heads = 32
    num_kv_heads = 8
    head_size = 128
    max_seq_len = 8192
    base = 10000
    is_neox_style = True
    # Relaxed tolerance for cross-implementation comparison
    # vLLM and FlashInfer may have slight numerical differences
    tol = 5e-2
    
    num_tokens = batch_size * seq_len
    
    # Create inputs
    query_orig = torch.randn(num_tokens, num_q_heads * head_size, dtype=dtype, device=device)
    key_orig = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)
    positions = torch.arange(seq_len, device=device, dtype=torch.int64).repeat(batch_size)
    
    results = {}
    
    # FlashInfer (reference)
    if HAS_FLASHINFER:
        fi_rope = FlashInferRotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            max_position_embeddings=max_seq_len,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        ).cuda()
        
        q_fi = query_orig.clone()
        k_fi = key_orig.clone()
        fi_rope.forward_cuda(positions, q_fi, k_fi)
        results["FlashInfer"] = (q_fi, k_fi)
        print("✓ FlashInfer computed")
    
    # vLLM
    if HAS_VLLM:
        vllm_rope = vLLMRotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            max_position_embeddings=max_seq_len,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        ).cuda()
        
        q_vllm = query_orig.clone()
        k_vllm = key_orig.clone()
        vllm_rope.forward_cuda(positions, q_vllm, k_vllm)
        results["vLLM"] = (q_vllm, k_vllm)
        print("✓ vLLM computed")
    
    # sgl_kernel new
    if HAS_SGL_NEW:
        sgl_new_rope = SGLRotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            max_position_embeddings=max_seq_len,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        ).cuda()
        
        q_sgl_new = query_orig.clone()
        k_sgl_new = key_orig.clone()
        sgl_new_rope.forward_cuda(positions, q_sgl_new, k_sgl_new)
        results["SGL_New"] = (q_sgl_new, k_sgl_new)
        print("✓ SGL_New computed")
    
    if HAS_SGL_OLD:
        sgl_old_rope = SGLOldRotaryEmbedding(
            head_size=head_size,
            rotary_dim=head_size,
            max_position_embeddings=max_seq_len,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        ).cuda()
        
        q_sgl_old = query_orig.clone()
        k_sgl_old = key_orig.clone()
        sgl_old_rope.forward_cuda(positions, q_sgl_old, k_sgl_old)
        results["SGL_Old"] = (q_sgl_old, k_sgl_old)
        print("✓ SGL_Old computed")
    
    ref_name = "FlashInfer" if "FlashInfer" in results else ("vLLM" if "vLLM" in results else None)
    
    if ref_name:
        q_ref, k_ref = results[ref_name]
        print(f"\nComparing against {ref_name}:")
        print("-" * 60)
        
        for name, (q, k) in results.items():
            if name == ref_name:
                continue
            
            q_diff = (q - q_ref).abs().max().item()
            k_diff = (k - k_ref).abs().max().item()
            passed = q_diff < tol and k_diff < tol
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name:12}: Q_diff={q_diff:.2e}, K_diff={k_diff:.2e} {status}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RotaryEmbedding implementations")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-q-heads", type=int, default=32, help="Number of Q heads")
    parser.add_argument("--num-kv-heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head-size", type=int, default=128, help="Head size")
    parser.add_argument("--base", type=int, default=10000, help="RoPE base")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Repeat iterations")
    parser.add_argument("--correctness", action="store_true", help="Run correctness test only")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Available implementations:")
    print(f"  vLLM:       {'✓' if HAS_VLLM else '✗'}")
    print(f"  FlashInfer: {'✓' if HAS_FLASHINFER else '✗'}")
    print(f"  SGL_New:    {'✓' if HAS_SGL_NEW else '✗'}")
    print(f"  SGL_Old:    {'✓' if HAS_SGL_OLD else '✗'}")
    print("=" * 80)
    
    test_correctness()
    
    if not args.correctness:
        run_benchmark(
            batch_size=args.batch_size,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            base=args.base,
            warmup=args.warmup,
            repeat=args.repeat,
        )


if __name__ == "__main__":
    main()

