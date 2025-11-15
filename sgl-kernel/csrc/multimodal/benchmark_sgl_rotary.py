#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import torch

# ---------- 选择内核入口：sgl_kernel 优先，回退本地 sgl_rotary ----------
ROTARY_OP = None
try:
    from sgl_kernel import rotary_embedding as _rotary
    ROTARY_OP = _rotary
    print("✓ 使用 sgl_kernel.rotary_embedding")
except Exception as e1:
    try:
        import sgl_rotary  # 本目录编译产物
        ROTARY_OP = sgl_rotary.rotary_embedding
        print("✓ 使用本地模块 sgl_rotary.rotary_embedding")
    except Exception as e2:
        print("未找到 rotary kernel：既没有 sgl_kernel 也没有本地 sgl_rotary")
        print("错误信息：", e1, " / ", e2)
        raise SystemExit(1)


# ---------- 参考实现 ----------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope_gptj(q, cos, sin):
    # 参考：半维度拼接旋转
    qf = q.float()
    cos, sin = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
    x1, x2 = qf[..., : qf.shape[-1] // 2], qf[..., qf.shape[-1] // 2 :]
    return (qf * cos) + (torch.cat((-x2, x1), dim=-1) * sin)

def _apply_rope_neox(q, cos, sin):
    # 参考：偶/奇交织旋转
    qf = q.float()
    cos, sin = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
    q_even, q_odd = qf[..., ::2], qf[..., 1::2]
    cos_e, sin_e = cos[..., ::2], sin[..., ::2]
    rot_even = q_even * cos_e - q_odd * sin_e
    rot_odd  = q_odd  * cos_e + q_even * sin_e
    # 交织回去
    out = torch.empty_like(qf)
    out[..., ::2], out[..., 1::2] = rot_even, rot_odd
    return out

def detect_kernel_style():
    import torch
    from math import isfinite

    # 小尺寸运行一次：cos=1,sin=0，然后 cos=0,sin=1
    device, dtype = "cuda", torch.bfloat16
    T, H, D = 4, 3, 16
    q = torch.randn(T, H, D, device=device, dtype=dtype)
    k = torch.randn(T, H, D, device=device, dtype=dtype)
    cos0 = torch.ones(T, D, device=device, dtype=dtype)
    sin0 = torch.zeros(T, D, device=device, dtype=dtype)
    cos1 = torch.zeros(T, D, device=device, dtype=dtype)
    sin1 = torch.ones(T, D, device=device, dtype=dtype)

    # 引用可调用入口（统一使用全局已解析好的 ROTARY_OP）
    ROT = ROTARY_OP

    # 1) cos=1,sin=0 => 应该恒等
    q1 = q.clone().float(); k1 = k.clone().float()
    ROT(cos0.float(), sin0.float(), q1, k1, D, False)
    assert torch.allclose(q1.to(dtype), q, atol=1e-6, rtol=1e-6), "kernel 非恒等：形状或数据布局不兼容"

    # 2) cos=0,sin=1 => 取决于风格
    q2 = q.clone().float(); k2 = k.clone().float()
    ROT(cos1.float(), sin1.float(), q2, k2, D, False)  # 先用 False 调一次（开关谁真谁假稍后判断）

    ref_gptj = _apply_rope_gptj(q, cos1, sin1)
    ref_neox = _apply_rope_neox(q, cos1, sin1)

    e_gptj = (q2.to(dtype) - ref_gptj).abs().max().item()
    e_neox = (q2.to(dtype) - ref_neox).abs().max().item()

    style = "gptj" if e_gptj < e_neox else "neox"
    print(f"[detect] kernel 风格推断为: {style}  (max_err gptj={e_gptj:.3e}, neox={e_neox:.3e})")
    return style

def apply_rotary_pos_emb_ref(q, k, cos, sin, is_neox_style: bool):
    q_dtype, k_dtype = q.dtype, k.dtype
    if is_neox_style:
        q_out = _apply_rope_neox(q, cos, sin)
        k_out = _apply_rope_neox(k, cos, sin)
    else:
        q_out = _apply_rope_gptj(q, cos, sin)
        k_out = _apply_rope_gptj(k, cos, sin)
    return q_out.to(q_dtype), k_out.to(k_dtype)


## 保留在本脚本内的函数都为测试/基准所需，移除未使用的参考路径以保持精简


class RotaryEmbeddingRef(torch.nn.Module):
    """只保留单测用到的能力：构造/缓存 cos_sin，不在此脚本里使用 cache 索引分支"""
    def __init__(self, head_size: int, rotary_dim: int, max_position_embeddings: int, base: int,
                 dtype: torch.dtype, is_neox_style: bool = False):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

    def forward_native(self, cos, sin, query, key):
        return apply_rotary_pos_emb_ref(query, key, cos, sin, self.is_neox_style)

    def forward_kernel_inplace(self, cos, sin, query, key):
        cos, sin = cos.float().contiguous(), sin.float().contiguous()
        query, key = query.float(), key.float()
        ROTARY_OP(cos, sin, query, key, self.head_size, self.is_neox_style)
        return query.to(self.dtype), key.to(self.dtype)


# ---------- 正确性（与 test_correctness 参数一致） ----------
def run_correctness_cases(device="cuda"):
    cases = [
        (80, 80, int(1e6), int(1e6), False, torch.bfloat16, 32, 32, 16, 16),
        (320, 230, int(1e6), int(1e6), False, torch.bfloat16, 32, 32, 16, 16),
        (80, 80, int(1e6), int(1e6), True,  torch.bfloat16, 32, 32, 16, 16),
    ]
    for (head_size, rotary_dim, mpe, base, is_neox, dtype, bs, seqlen, Hq, Hkv) in cases:
        rope = RotaryEmbeddingRef(head_size, rotary_dim, mpe, base, dtype, is_neox_style=USE_NEOX).to(device)
        T = bs * seqlen
        q = torch.randn(T, Hq, head_size, dtype=dtype, device=device)
        k = torch.randn(T, Hkv, head_size, dtype=dtype, device=device)
        cos = torch.randn(T, head_size, dtype=dtype, device=device)
        sin = torch.randn(T, head_size, dtype=dtype, device=device)
        q_ref, k_ref = rope.forward_native(cos, sin, q.clone(), k.clone())
        q_out, k_out = rope.forward_kernel_inplace(cos, sin, q, k)
        torch.testing.assert_close(q_ref, q_out, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(k_ref, k_out, atol=1e-3, rtol=1e-3)
    print("✓ 正确性通过：对齐 test_correctness")


# ---------- 基准（与 test_rotary_embedding_benchmark 对齐） ----------
def bench_one(
    *,
    head_size=80,
    rotary_dim=80,
    is_neox_style=False,
    dtype=torch.bfloat16,
    batch_size=1,
    seq_len=8840,
    num_q_heads=16,
    num_kv_heads=16,
    warmup_rounds=5,
    bench_rounds=20000,
    reset_each_iter=False,  # 默认 False：与 pytest 一致（不恢复输入）
):
    device = "cuda"
    T = batch_size * seq_len

    rope = RotaryEmbeddingRef(head_size, rotary_dim, int(1e6), int(1e6), dtype, is_neox_style).to(device)

    q = torch.randn(T, num_q_heads, head_size, dtype=dtype, device=device)
    k = torch.randn(T, num_kv_heads, head_size, dtype=dtype, device=device)
    cos = torch.randn(T, head_size, dtype=dtype, device=device)
    sin = torch.randn(T, head_size, dtype=dtype, device=device)

    if reset_each_iter:
        q0, k0 = q.clone(), k.clone()
        cos0, sin0 = cos.clone(), sin.clone()

    for _ in range(warmup_rounds):
        if reset_each_iter:
            q.copy_(q0); k.copy_(k0); cos.copy_(cos0); sin.copy_(sin0)
        rope.forward_kernel_inplace(cos, sin, q, k)
        torch.cuda.synchronize()

    times = []
    for _ in range(bench_rounds):
        if reset_each_iter:
            q.copy_(q0); k.copy_(k0); cos.copy_(cos0); sin.copy_(sin0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rope.forward_kernel_inplace(cos, sin, q, k)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # us

    a = np.array(times, dtype=np.float64)
    stats = dict(mean=a.mean(), std=a.std(), min=a.min(), max=a.max(), median=np.median(a), rounds=bench_rounds)
    return stats


def accuracy_one(
    *,
    head_size=80,
    rotary_dim=80,
    is_neox_style=False,
    dtype=torch.bfloat16,
    batch_size=1,
    seq_len=8840,
    num_q_heads=16,
    num_kv_heads=16,
):
    """对齐 benchmark 规模做一次精度对比，返回误差指标。"""
    device = "cuda"
    T = batch_size * seq_len
    rope = RotaryEmbeddingRef(head_size, rotary_dim, int(1e6), int(1e6), dtype, is_neox_style).to(device)
    q = torch.randn(T, num_q_heads, head_size, dtype=dtype, device=device)
    k = torch.randn(T, num_kv_heads, head_size, dtype=dtype, device=device)
    cos = torch.randn(T, head_size, dtype=dtype, device=device)
    sin = torch.randn(T, head_size, dtype=dtype, device=device)
    q_ref, k_ref = rope.forward_native(cos, sin, q.clone(), k.clone())
    q_out, k_out = rope.forward_kernel_inplace(cos, sin, q.clone(), k.clone())
    q_err = (q_ref - q_out).abs()
    k_err = (k_ref - k_out).abs()
    return {
        "q_max_abs": float(q_err.max().item()),
        "q_mean_abs": float(q_err.mean().item()),
        "k_max_abs": float(k_err.max().item()),
        "k_mean_abs": float(k_err.mean().item()),
    }


def main():
    style = detect_kernel_style()
    global USE_NEOX
    USE_NEOX = (style == "neox")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-each-iter", action="store_true", help="每轮恢复输入（默认不恢复，以匹配 pytest 行为）")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")

    # 先做正确性（与 test_correctness 对齐）
    run_correctness_cases()

    # 三组基准（与 test_rotary_embedding_benchmark 三组参数一致/等价）
    configs = [
        dict(name="80/False/bs=1,seqlen=8840", head_size=80, rotary_dim=80, is_neox_style=False,
             dtype=torch.bfloat16, batch_size=1, seq_len=8840, num_q_heads=16, num_kv_heads=16),
        dict(name="80/False/bs=1,seqlen=4000", head_size=80, rotary_dim=80, is_neox_style=False,
             dtype=torch.bfloat16, batch_size=1, seq_len=4000, num_q_heads=16, num_kv_heads=16),
        dict(name="80/True/ bs=8,seqlen=8840", head_size=80, rotary_dim=80, is_neox_style=True,
             dtype=torch.bfloat16, batch_size=8, seq_len=8840, num_q_heads=16, num_kv_heads=16),
    ]

    print("\n=== Rotary Embedding Kernel Benchmark（us） ===")
    all_rows = []
    for i, cfg in enumerate(configs, 1):
        name = cfg.pop("name")
        print(f"\n[Case {i}] {name}")
        s = bench_one(**cfg, warmup_rounds=5, bench_rounds=20000,
                      reset_each_iter=args.reset_each_iter)
        print(f"  Min    {s['min']:10.2f} us")
        print(f"  Max    {s['max']:10.2f} us")
        print(f"  Mean   {s['mean']:10.2f} us")
        print(f"  Median {s['median']:10.2f} us")
        print(f"  StdDev {s['std']:10.2f} us")
        print(f"  Rounds {s['rounds']}")
        # 精度
        acc = accuracy_one(**cfg)
        # 吞吐（token/s）
        T = cfg["batch_size"] * cfg["seq_len"]
        tok_per_sec = 1e6 * T / s["mean"]
        all_rows.append((name, s, acc, tok_per_sec))

    # 汇总表格：精度 + 性能
    print("\n=== 汇总：精度与性能对比 ===")
    header = f"{'Case':<28} {'Mean(us)':>10} {'Min(us)':>10} {'Std(us)':>10} {'Tok/s':>12}  {'Q_max':>9} {'Q_mean':>9} {'K_max':>9} {'K_mean':>9}"
    print(header)
    print("-" * len(header))
    for name, s, acc, tps in all_rows:
        print(f"{name:<28} {s['mean']:10.2f} {s['min']:10.2f} {s['std']:10.2f} {tps:12.0f}  "
              f"{acc['q_max_abs']:9.3e} {acc['q_mean_abs']:9.3e} {acc['k_max_abs']:9.3e} {acc['k_mean_abs']:9.3e}")
    print("\n说明：默认与 pytest-benchmark 语义一致（不恢复输入；就地更新）。"
          "如需逐轮恢复输入以便横向对比，加 --reset-each-iter。")


if __name__ == "__main__":
    main()
