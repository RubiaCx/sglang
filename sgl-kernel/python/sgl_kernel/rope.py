"""
Rotary Positional Embedding (RoPE) Module
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from sgl_kernel.elementwise import apply_rope_with_cos_sin_cache_inplace


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) module.
    
    This class provides a high-level interface for applying RoPE to query and key tensors.
    It manages the cos/sin cache internally and provides efficient CUDA kernel implementations.
    
    Args:
        head_size: The size of each attention head
        rotary_dim: The dimension to apply rotary embedding (usually same as head_size)
        max_position_embeddings: Maximum sequence length
        base: The base for computing inverse frequencies (default: 10000)
        is_neox_style: If True, uses GPT-NeoX style (non-interleaved). Default: True
        dtype: Data type for query/key tensors (for validation)
    """
    
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: Union[int, float] = 10000,
        is_neox_style: bool = True,
        dtype: torch.dtype = torch.bfloat16,
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
        """Compute inverse frequencies for RoPE."""
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim
            )
        )
        return inv_freq
    
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """
        Compute the cos/sin cache.
        
        Returns:
            Tensor of shape (max_position_embeddings, rotary_dim) where:
            - First half contains cos values
            - Second half contains sin values
        """
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)  # (max_seq_len, rotary_dim)
        return cache
    
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embedding to query and key tensors (in-place).
        
        Args:
            positions: Position indices, shape (num_tokens,), dtype int32/int64
            query: Query tensor, shape (num_tokens, num_q_heads * head_size)
            key: Key tensor, shape (num_tokens, num_kv_heads * head_size)
            offsets: Optional offsets for positions (not used, for API compatibility)
            
        Returns:
            Tuple of (query, key) - same tensors, modified in-place
        """
        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=self.is_neox_style,
        )
        return query, key
    
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for forward_cuda."""
        return self.forward_cuda(positions, query, key, offsets)


class RotaryEmbeddingWithScaling(RotaryEmbedding):
    """
    Rotary Embedding with scaling support (e.g., for long context models).
    
    Supports:
    - Linear scaling: scale positions by a factor
    - Dynamic NTK scaling: adjust base frequency based on sequence length
    
    Args:
        head_size: The size of each attention head
        rotary_dim: The dimension to apply rotary embedding
        max_position_embeddings: Maximum sequence length
        base: The base for computing inverse frequencies
        is_neox_style: If True, uses GPT-NeoX style (non-interleaved)
        dtype: Data type for query/key tensors
        scaling_factor: Factor to scale positions (for linear scaling)
        scaling_type: Type of scaling ("linear" or "dynamic")
    """
    
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: Union[int, float] = 10000,
        is_neox_style: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        scaling_factor: float = 1.0,
        scaling_type: str = "linear",
    ) -> None:
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type
        
        # For dynamic scaling, adjust the base
        if scaling_type == "dynamic" and scaling_factor > 1.0:
            base = base * (
                (scaling_factor * max_position_embeddings / max_position_embeddings) 
                - (scaling_factor - 1)
            ) ** (rotary_dim / (rotary_dim - 2))
        
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
    
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute cos/sin cache with scaling applied."""
        inv_freq = self._compute_inv_freq(self.base)
        
        if self.scaling_type == "linear":
            t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
            t = t / self.scaling_factor
        else:
            t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

SGLangRotaryEmbedding = RotaryEmbedding