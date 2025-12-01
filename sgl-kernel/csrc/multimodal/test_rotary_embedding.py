#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch

try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

try:
    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

    HAS_FLASH_ATTN = True
except ImportError:
    FlashRotaryEmbedding = None  # type: ignore
    HAS_FLASH_ATTN = False
    print("flash_attn not available")

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

try:
    import rotary_embedding as rotary_emb_module

    HAS_ROTARY_EMBEDDING = True
except ImportError:
    rotary_emb_module = None  # type: ignore
    HAS_ROTARY_EMBEDDING = False
    print("  Run 'bash compile_rotary_embedding.sh' first")

try:
    from vllm.model_executor.layers.rotary_embedding import (
        RotaryEmbedding as vLLMRotaryEmbedding,
    )

    HAS_VLLM = True
except ImportError:
    vLLMRotaryEmbedding = None  # type: ignore
    HAS_VLLM = False
    print("vLLM not available")

try:
    from sgl_kernel.rotary_embedding import rotary_embedding as sgl_rotary_embedding

    HAS_SGL_KERNEL = True
except ImportError:
    sgl_rotary_embedding = None  # type: ignore
    HAS_SGL_KERNEL = False
    print("sgl_kernel.rotary_embedding not available")


@dataclass
class TestResult:
    name: str
    passed: bool
    q_diff: float = 0.0
    k_diff: float = 0.0
    details: str = ""


def compute_cos_sin_cache(
    max_seq_len: int,
    rotary_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute separate cos and sin caches (vLLM-style).

    Returns:
        cos, sin: shape (max_seq_len, rotary_dim / 2)
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def reference_rotary_neox(
    query: torch.Tensor,  # (num_tokens, num_heads * head_size)
    key: torch.Tensor,  # (num_tokens, num_kv_heads * head_size)
    cos: torch.Tensor,  # (num_tokens, rotary_dim / 2)
    sin: torch.Tensor,  # (num_tokens, rotary_dim / 2)
    head_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of NeoX-style rotary embedding, using explicit
    cos/sin caches and Q/K in flattened (num_tokens, heads * dim) layout.
    """
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

    cos_expanded = cos.float().unsqueeze(1)  # (tokens, 1, rotary_dim/2)
    sin_expanded = sin.float().unsqueeze(1)

    # Apply rotary: x' = x*cos - y*sin, y' = y*cos + x*sin
    q_x = q_rot[..., 0]
    q_y = q_rot[..., 1]
    q_rot_out = torch.stack(
        [
            q_x * cos_expanded - q_y * sin_expanded,
            q_y * cos_expanded + q_x * sin_expanded,
        ],
        dim=-1,
    )

    k_x = k_rot[..., 0]
    k_y = k_rot[..., 1]
    k_rot_out = torch.stack(
        [
            k_x * cos_expanded - k_y * sin_expanded,
            k_y * cos_expanded + k_x * sin_expanded,
        ],
        dim=-1,
    )

    q_rot_out = q_rot_out.view(num_tokens, num_heads, rotary_dim)
    k_rot_out = k_rot_out.view(num_tokens, num_kv_heads, rotary_dim)

    q_out = torch.cat([q_rot_out, q_pass], dim=-1)
    k_out = torch.cat([k_rot_out, k_pass], dim=-1)

    q_out = q_out.view(num_tokens, num_heads * head_size).to(query.dtype)
    k_out = k_out.view(num_tokens, num_kv_heads * head_size).to(key.dtype)

    return q_out, k_out


def test_basic_correctness(
    batch_size: int = 2,
    seq_len: int = 128,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    tol: float = 1e-2,
    device: str = "cuda",
) -> TestResult:
    if not HAS_ROTARY_EMBEDDING:
        return TestResult(
            "basic (skipped: rotary_embedding not built)",
            True,
            details="rotary_embedding extension not found",
        )

    rotary_dim = head_size
    max_seq_len = 8192
    num_tokens = batch_size * seq_len

    query = torch.randn(
        num_tokens, num_heads * head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        num_tokens, num_kv_heads * head_size, dtype=dtype, device=device
    )

    cos_cache, sin_cache = compute_cos_sin_cache(
        max_seq_len, rotary_dim, dtype=dtype
    )
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)

    positions = torch.arange(seq_len, device=device).repeat(batch_size)
    cos = cos_cache[positions]
    sin = sin_cache[positions]

    q_ref, k_ref = reference_rotary_neox(
        query.clone(), key.clone(), cos, sin, head_size
    )

    q_out = query.clone()
    k_out = key.clone()
    rotary_emb_module.rotary_embedding(
        cos, sin, q_out, k_out, head_size, True  # is_neox=True
    )

    q_diff = (q_out - q_ref).abs().max().item()
    k_diff = (k_out - k_ref).abs().max().item()
    passed = q_diff < tol and k_diff < tol

    name = (
        f"basic [bs={batch_size}, seq={seq_len}, "
        f"heads={num_heads}/{num_kv_heads}, dim={head_size}, {dtype}]"
    )
    return TestResult(name, passed, q_diff, k_diff)


def test_vs_sgl_kernel(
    batch_size: int = 2,
    seq_len: int = 128,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    tol: float = 1e-2,
    device: str = "cuda",
) -> TestResult:
    if not HAS_SGL_KERNEL:
        return TestResult(
            "vs_sgl_kernel (skipped: sgl_kernel not available)",
            True,
            details="sgl_kernel.rotary_embedding not available",
        )
    if not HAS_ROTARY_EMBEDDING:
        return TestResult(
            "vs_sgl_kernel (skipped: rotary_embedding not built)",
            True,
            details="rotary_embedding extension not found",
        )

    rotary_dim = head_size
    max_seq_len = 8192
    num_tokens = batch_size * seq_len

    query_orig = torch.randn(
        num_tokens, num_heads * head_size, dtype=dtype, device=device
    )
    key_orig = torch.randn(
        num_tokens, num_kv_heads * head_size, dtype=dtype, device=device
    )

    cos_cache, sin_cache = compute_cos_sin_cache(
        max_seq_len, rotary_dim, dtype=dtype
    )
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)

    positions = torch.arange(seq_len, device=device).repeat(batch_size)
    cos = cos_cache[positions]
    sin = sin_cache[positions]

    # sgl_kernel
    q_sgl = query_orig.clone()
    k_sgl = key_orig.clone()
    sgl_rotary_embedding(
        cos, sin, q_sgl, k_sgl, head_size, True  # is_neox=True
    )

    # ours
    q_out = query_orig.clone()
    k_out = key_orig.clone()
    rotary_emb_module.rotary_embedding(
        cos, sin, q_out, k_out, head_size, True
    )

    q_diff = (q_out - q_sgl).abs().max().item()
    k_diff = (k_out - k_sgl).abs().max().item()
    passed = q_diff < tol and k_diff < tol

    name = (
        f"vs_sgl_kernel [bs={batch_size}, seq={seq_len}, "
        f"heads={num_heads}/{num_kv_heads}, dim={head_size}, {dtype}]"
    )
    return TestResult(name, passed, q_diff, k_diff)


def test_different_configs() -> List[TestResult]:
    """Sweep a few head sizes / head counts / seq_lens / dtypes."""
    results: List[TestResult] = []

    # Vary head_size
    for head_size in [64, 128, 256]:
        results.append(test_basic_correctness(head_size=head_size))

    # Vary num_heads / num_kv_heads
    for num_heads, num_kv_heads in [(8, 8), (32, 8), (32, 1), (64, 8)]:
        results.append(
            test_basic_correctness(
                num_heads=num_heads, num_kv_heads=num_kv_heads
            )
        )

    # Vary seq_len
    for seq_len in [1, 16, 128, 1024]:
        results.append(test_basic_correctness(seq_len=seq_len))

    # Vary dtype
    for dtype in [torch.float16, torch.bfloat16]:
        results.append(test_basic_correctness(dtype=dtype))

    return results


def _print_results_table(results: List[TestResult], title: str) -> None:
    if not results:
        return

    header = f"{'#':>3}  {'Test name':<72} {'Q_max_err':>11} {'K_max_err':>11}"
    width = len(header)

    print(f"\n{title}")
    print("-" * width)
    print(header)
    print("-" * width)

    for idx, r in enumerate(results, 1):
        status = "✓" if r.passed else "✗"
        name = r.name
        if len(name) > 70:
            name = name[:67] + "..."

        print(
            f"{idx:3d}  {status} {name:<70} "
            f"{r.q_diff:11.2e} {r.k_diff:11.2e}"
        )

    print()


def run_all_tests() -> List[TestResult]:
    """Run numerical tests for the custom rotary_embedding kernel."""
    if not HAS_ROTARY_EMBEDDING:
        print(
            "\nrotary_embedding extension not found. "
            "Please compile it first: bash compile_rotary_embedding.sh"
        )
        return []

    all_results: List[TestResult] = []

    basic = [test_basic_correctness()]
    all_results.extend(basic)
    _print_results_table(basic, "[1] Basic correctness (single config)")

    vs_sgl = [test_vs_sgl_kernel()]
    all_results.extend(vs_sgl)
    _print_results_table(vs_sgl, "[2] Numerical parity vs sgl_kernel")

    sweeps = test_different_configs()
    all_results.extend(sweeps)
    _print_results_table(sweeps, "[3] Shape / dtype sweeps")

    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    print(f"SUMMARY: {passed}/{total} tests passed")
    if passed != total:
        print("Failing tests:")
        for r in all_results:
            if not r.passed:
                print(
                    f"  - {r.name} "
                    f"(Q={r.q_diff:.2e}, K={r.k_diff:.2e})"
                )
    return all_results


def benchmark_rotary() -> None:
    import time

    if not HAS_ROTARY_EMBEDDING:
        print("\nSkip benchmark: rotary_embedding extension not built.")
        return

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 1
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    rotary_dim = head_size
    max_seq_len = 65536

    seq_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    cos_cache, sin_cache = compute_cos_sin_cache(
        max_seq_len, rotary_dim, dtype=dtype
    )
    cos_cache = cos_cache.to(device)
    sin_cache = sin_cache.to(device)

    # vLLM module
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

    # FlashAttention RotaryEmbedding (NeoX-style, operates on qkv)
    flash_rotary = None
    if HAS_FLASH_ATTN:
        flash_rotary = FlashRotaryEmbedding(rotary_dim, device=device)

    print(f"\nConfig: batch_size={batch_size}, heads={num_heads}/{num_kv_heads}, head_size={head_size}, dtype={dtype}")
    print("-" * 80)

    header = f"{'seq_len':>8}"
    header += f" | {'ours (ms)':>10}"
    if HAS_VLLM:
        header += f" | {'vLLM (ms)':>10}"
    if HAS_FLASH_ATTN:
        header += f" | {'flash_attn (ms)':>14}"
    if HAS_SGL_KERNEL:
        header += f" | {'sgl_kernel (ms)':>14}"
    if HAS_VLLM:
        header += f" | {'vLLM/SGL':>9}"
    print(header)
    print("-" * 80)

    results = []

    for seq_len in seq_lens:
        num_tokens = batch_size * seq_len
        query = torch.randn(
            num_tokens, num_heads * head_size, dtype=dtype, device=device
        )
        key = torch.randn(
            num_tokens, num_kv_heads * head_size, dtype=dtype, device=device
        )

        positions = torch.arange(seq_len, device=device, dtype=torch.int64).repeat(batch_size)
        cos = cos_cache[positions]
        sin = sin_cache[positions]

        row_str = f"{seq_len:8d}"
        ours_time = None
        vllm_time = None
        fa_time = None
        sgl_time = None

        # Ours
        # Warmup
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
        ours_time = (time.perf_counter() - start) / 100 * 1000.0  # ms
        row_str += f" | {ours_time:10.4f}"

        # vLLM
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
            vllm_time = (time.perf_counter() - start) / 100 * 1000.0
            row_str += f" | {vllm_time:10.4f}"
        elif HAS_VLLM:
            row_str += f" | {'N/A':>10}"

        # FlashAttention RotaryEmbedding (NeoX-style, QKV layout)
        if HAS_FLASH_ATTN and flash_rotary is not None:
            qkv = torch.randn(
                batch_size,
                seq_len,
                3,
                num_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
            # Warmup
            for _ in range(10):
                qkv_fa = qkv.clone()
                flash_rotary(qkv_fa, seqlen_offset=0)
            torch.cuda.synchronize()
            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                qkv_fa = qkv.clone()
                flash_rotary(qkv_fa, seqlen_offset=0)
            torch.cuda.synchronize()
            fa_time = (time.perf_counter() - start) / 100 * 1000.0
            row_str += f" | {fa_time:14.4f}"
        elif HAS_FLASH_ATTN:
            row_str += f" | {'N/A':>14}"

        # sgl_kernel
        if HAS_SGL_KERNEL and sgl_rotary_embedding is not None:
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
            sgl_time = (time.perf_counter() - start) / 100 * 1000.0
            row_str += f" | {sgl_time:14.4f}"
        elif HAS_SGL_KERNEL:
            row_str += f" | {'N/A':>14}"

        # Speedup vs vLLM
        if HAS_VLLM and ours_time is not None and vllm_time is not None:
            speedup = vllm_time / ours_time
            row_str += f" | {speedup:9.2f}x"

        print(row_str)
        results.append(
            {
                "seq_len": seq_len,
                "ours": ours_time,
                "vllm": vllm_time,
                "flash_attn": fa_time,
                "sgl_kernel": sgl_time,
            }
        )

    print("-" * 80)
    print("\nSUMMARY:")
    ours_vals = [r["ours"] for r in results if r["ours"] is not None]
    if ours_vals:
        print(f"  SGL      avg: {np.mean(ours_vals):.4f} ms")
    if HAS_VLLM:
        vllm_vals = [r["vllm"] for r in results if r["vllm"] is not None]
        if vllm_vals:
            print(f"  vLLM      avg: {np.mean(vllm_vals):.4f} ms")
    if HAS_FLASH_ATTN:
        fa_vals = [
            r["flash_attn"] for r in results if r["flash_attn"] is not None
        ]
        if fa_vals:
            print(f"  flash_attn avg: {np.mean(fa_vals):.4f} ms")
    if HAS_SGL_KERNEL:
        sgl_vals = [
            r["sgl_kernel"] for r in results if r["sgl_kernel"] is not None
        ]
        if sgl_vals:
            print(f"  sgl_kernel avg: {np.mean(sgl_vals):.4f} ms")

if __name__ == "__main__":
    run_all_tests()
    benchmark_rotary()
