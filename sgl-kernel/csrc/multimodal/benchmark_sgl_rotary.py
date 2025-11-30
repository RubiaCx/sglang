"""
Benchmark RoPE for flashinfer, sgl-kernel and vllm.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import triton
from vllm.model_executor.layers.rotary_embedding import (
    RotaryEmbedding as vLLMRotaryEmbedding,
)

from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
from flashinfer.testing.utils import bench_gpu_time

# Optional: sgl-kernel rotary embedding for comparison
import os
import sys

try:
    from sgl_kernel import rotary_embedding as sgl_rotary  

    HAS_SGL_KERNEL = True
except Exception:
    sgl_rotary = None
    HAS_SGL_KERNEL = False

    _here = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join("/workspace/sglang/sgl-kernel/csrc/multimodal"),
        os.path.normpath(
            os.path.join(_here, "..", "..", "sglang", "sgl-kernel", "csrc", "multimodal")
        ),
    ]
    for _path in _candidates:
        so_path = os.path.join(_path, "sgl_rotary.so")
        if os.path.exists(so_path):
            if _path not in sys.path:
                sys.path.append(_path)
            try:
                import sgl_rotary as _sgl_mod  # type: ignore[import]

                sgl_rotary = _sgl_mod.rotary_embedding
                HAS_SGL_KERNEL = True
                break
            except Exception:
                continue

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

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool,
    ) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, num_heads, head_size]
            cos: [num_tokens, head_size // 2]
            sin: [num_tokens, head_size // 2]
            is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
                positional embeddings.
        """
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            return torch.cat((o1, o2), dim=-1)
        else:
            return torch.stack((o1, o2), dim=-1).flatten(-2)

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


class SGLRotaryEmbedding(nn.Module):
    """Use sgl-kernel rotary_embedding kernel for benchmark."""

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
        if not HAS_SGL_KERNEL:
            raise RuntimeError("sgl_kernel is not available; cannot benchmark provider='sgl'")
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        self._cached_positions: Optional[torch.Tensor] = None
        self._cached_cos: Optional[torch.Tensor] = None
        self._cached_sin: Optional[torch.Tensor] = None

        self.register_buffer(
            "cos_cache",
            torch.randn(max_position_embeddings, head_size, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            torch.randn(max_position_embeddings, head_size, dtype=dtype),
            persistent=False,
        )

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: [num_tokens]
            query: [num_tokens, num_q_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        if (
            self._cached_positions is None
            or self._cached_positions.device != positions.device
            or self._cached_positions.dtype != positions.dtype
            or not torch.equal(self._cached_positions, positions)
        ):
            cos = self.cos_cache.index_select(0, positions)
            sin = self.sin_cache.index_select(0, positions)
            self._cached_positions = positions.detach().clone()
            self._cached_cos = cos
            self._cached_sin = sin
        else:
            cos = self._cached_cos
            sin = self._cached_sin

        num_tokens = positions.shape[0]
        q = query.view(num_tokens, self.num_q_heads, self.head_size)
        k = key.view(num_tokens, self.num_kv_heads, self.head_size)

        if hasattr(sgl_rotary, "rotary_embedding"):
            _rotary = sgl_rotary.rotary_embedding  # type: ignore[attr-defined]
        else:
            _rotary = sgl_rotary

        _rotary(
            cos,
            sin,
            q,
            k,
            self.head_size,
            self.is_neox_style,
        )

        query_out = q.view(num_tokens, self.num_q_heads * self.head_size)
        key_out = k.view(num_tokens, self.num_kv_heads * self.head_size)
        return query_out, key_out


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ],
        line_arg="provider",
        line_vals=["flashinfer", "vllm_native", "vllm", "sgl"],
        line_names=["FlashInfer", "vLLM_Native", "vLLM", "SGL"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("black", "-")],
        ylabel="Latency (ms)",
        plot_name="rope-latency",
        args={
            "head_size": 4096 // 32,
            "rotary_dim": 4096 // 32,
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
        f"provider: {provider}, head_size: {head_size}, rotary_dim: {rotary_dim}, max_position_embeddings: {max_position_embeddings}, base: {base}, is_neox_style: {is_neox_style}, dtype: {dtype}, device: {device}, batch_size: {batch_size}, seq_len: {seq_len}, num_q_heads: {num_q_heads}, num_kv_heads: {num_kv_heads}"
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
    elif provider == "sgl":
        rope = SGLRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
            num_q_heads,
            num_kv_heads,
        ).to(device)
        rope_forward = rope.forward_cuda

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

if __name__ == "__main__":
    benchmark.run(print_data=True)