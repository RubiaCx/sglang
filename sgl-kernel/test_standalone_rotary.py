#!/usr/bin/env python3
"""
独立测试 rotary_embedding kernel
"""
import torch
import sys

# 加载编译的扩展
try:
    import standalone_rotary
    print("✓ 成功加载 standalone_rotary 模块")
except ImportError as e:
    print(f"✗ 无法加载模块: {e}")
    print("请先运行: bash compile_standalone.sh")
    sys.exit(1)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, is_neox_style=False, unsqueeze_dim=1):
    """PyTorch native implementation"""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()

    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    
    if is_neox_style:
        # NEOX style: cos/sin only use first half dimensions
        # q_new[i] = q[i] * cos[i] - q[i+D/2] * sin[i]
        # q_new[i+D/2] = q[i+D/2] * cos[i] + q[i] * sin[i]
        half_dim = q.shape[-1] // 2
        q1, q2 = q[..., :half_dim], q[..., half_dim:]
        cos_half = cos[..., :half_dim]
        sin_half = sin[..., :half_dim]
        
        q_embed = torch.cat([
            q1 * cos_half - q2 * sin_half,
            q2 * cos_half + q1 * sin_half
        ], dim=-1)
        
        k1, k2 = k[..., :half_dim], k[..., half_dim:]
        k_embed = torch.cat([
            k1 * cos_half - k2 * sin_half,
            k2 * cos_half + k1 * sin_half
        ], dim=-1)
    else:
        # Non-NEOX style: standard rotary embedding
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)

    return q_embed, k_embed


def test_rotary_embedding(
    head_size=80,
    rotary_dim=80,
    is_neox_style=False,
    dtype=torch.bfloat16,
    batch_size=32,
    seq_len=32,
    num_q_heads=16,
    num_kv_heads=16,
):
    device = "cuda"
    print(f"\n测试配置:")
    print(f"  head_size={head_size}, rotary_dim={rotary_dim}")
    print(f"  is_neox_style={is_neox_style}, dtype={dtype}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}")

    # 创建输入张量
    query = torch.randn(
        batch_size * seq_len, num_q_heads, head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        batch_size * seq_len, num_kv_heads, head_size, dtype=dtype, device=device
    )
    cos = torch.randn(batch_size * seq_len, head_size, dtype=dtype, device=device)
    sin = torch.randn(batch_size * seq_len, head_size, dtype=dtype, device=device)

    # Native PyTorch 实现
    query_native_out, key_native_out = apply_rotary_pos_emb(
        query.clone(), key.clone(), cos, sin, is_neox_style=is_neox_style
    )

    # CUDA kernel 实现（in-place）
    cos_float = cos.float()
    sin_float = sin.float()
    query_kernel = query.float()
    key_kernel = key.float()
    
    standalone_rotary.rotary_embedding(
        cos_float, sin_float, query_kernel, key_kernel, head_size, is_neox_style
    )
    
    query_kernel_out = query_kernel.to(dtype)
    key_kernel_out = key_kernel.to(dtype)

    # 比较结果
    try:
        torch.testing.assert_close(
            query_native_out, query_kernel_out, atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            key_native_out, key_kernel_out, atol=1e-3, rtol=1e-3
        )
        print("✓ 测试通过! Query 和 Key 结果匹配")
        return True
    except AssertionError as e:
        print(f"✗ 测试失败!")
        print(f"  Query 最大差异: {(query_native_out - query_kernel_out).abs().max().item()}")
        print(f"  Key 最大差异: {(key_native_out - key_kernel_out).abs().max().item()}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("独立 Rotary Embedding Kernel 测试")
    print("=" * 50)

    # 运行多个测试配置
    test_configs = [
        {"head_size": 80, "rotary_dim": 80, "is_neox_style": False},
        {"head_size": 320, "rotary_dim": 230, "is_neox_style": False},
        {"head_size": 80, "rotary_dim": 80, "is_neox_style": True},
    ]

    all_passed = True
    for i, config in enumerate(test_configs, 1):
        print(f"\n【测试 {i}/{len(test_configs)}】")
        passed = test_rotary_embedding(**config)
        all_passed = all_passed and passed

    print("\n" + "=" * 50)
    if all_passed:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败")
        sys.exit(1)

