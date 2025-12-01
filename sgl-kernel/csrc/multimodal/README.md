# 高性能 RoPE Kernel

基于 FlashInfer 优化技术的 Rotary Position Embedding (RoPE) 实现。

## 各项目 RoPE 实现对比

### 接口总览

| 项目 | 接口 | 输入格式 | 特点 |
|------|------|----------|------|
| **vLLM** | `rotary_embedding(cos, sin, query, key, head_size, is_neox)` | 分开的 cos/sin | in-place, 2D input |
| **FlashInfer** | `apply_rope_with_cos_sin_cache_inplace(positions, query, key, head_size, cos_sin_cache, is_neox)` | 合并 cache + pos_ids | in-place, 2D input |
| **sgl_kernel** | `rotary_embedding` | 分开的 cos/sin | vLLM 兼容 |
| **sgl_kernel** | `apply_rope_with_cos_sin_cache_inplace` | 合并 cache + pos_ids | FlashInfer 兼容 |
| **本实现 (rope.cu)** | `apply_rope_pos_ids_cos_sin_cache` | 合并 cache + pos_ids | 3D input, 支持 out-of-place |
| **本实现 (rope.cu)** | `apply_rope_pos_ids` | 运行时计算 cos/sin | 无需 cache |
| **本实现 (rope.cu)** | `apply_rope` | indptr + offset | variable-length batch |
| **本实现 (rope.cu)** | `rope_quantize` | RoPE + FP8 量化 | MLA/DeepSeek 专用 |
| **本实现 (rotary_embedding.cu)** | `rotary_embedding` | 分开的 cos/sin | vLLM 兼容, NEOX/GPT-J |

### 详细说明

#### 1. vLLM 的实现

```python
# python/sgl_kernel/rotary_embedding.py
from sgl_kernel import rotary_embedding

rotary_embedding(
    cos: torch.Tensor,      # (seq_len, rotary_dim // 2) - 从 cache 索引得到
    sin: torch.Tensor,      # (seq_len, rotary_dim // 2) - 从 cache 索引得到
    query: torch.Tensor,    # (num_tokens, num_heads * head_size) - 2D, in-place 修改
    key: torch.Tensor,      # (num_tokens, num_heads * head_size) - 2D, in-place 修改
    head_size: int,
    is_neox: bool,          # True = NEOX style, False = GPT-J style
)
```

**特点**：需要调用方先根据 positions 索引 cos/sin cache。

#### 2. FlashInfer 的实现

```python
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,     # (num_tokens,) - 直接传 position ids
    query: torch.Tensor,         # (num_tokens, num_heads * head_size) - 2D, in-place
    key: torch.Tensor,           # (num_tokens, num_heads * head_size) - 2D, in-place
    head_size: int,
    cos_sin_cache: torch.Tensor, # (max_seq_len, rotary_dim) - [cos | sin] 格式
    is_neox: bool = True,
)
```

**特点**：内部根据 positions 索引 cache，更方便使用。

#### 3. sgl_kernel 的实现

```python

from sgl_kernel.rotary_embedding import rotary_embedding
rotary_embedding(cos, sin, query, key, head_size, is_neox)

from sgl_kernel.elementwise import apply_rope_with_cos_sin_cache_inplace
apply_rope_with_cos_sin_cache_inplace(
    positions, query, key, head_size, cos_sin_cache, is_neox,
    fused_set_kv_buffer_arg=None,  # 可选: fused KV cache 写入
    enable_pdl=None,               # 可选: PDL 优化
)
```

**底层 C++ API**:
```cpp
// 12 个参数 (已安装版本)
torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache(
    q, k, q_rope, k_rope, cos_sin_cache, pos_ids,
    interleave, enable_pdl,
    v, k_buffer, v_buffer, kv_cache_loc  // 可选, 用于 fused KV cache
)
```

### 输入格式差异

| 实现 | Query/Key Shape | Position IDs |
|------|-----------------|--------------|
| vLLM | `(nnz, num_heads * head_size)` - 2D | 不需要 (已索引 cos/sin) |
| FlashInfer | `(nnz, num_heads * head_size)` - 2D | `(nnz,)` int64 |
| sgl_kernel | `(nnz, num_heads * head_size)` - 2D | `(nnz,)` int64 |
| 本实现 | `(nnz, num_heads, head_size)` - 3D | `(nnz,)` int32/int64 |

---

## 背景

### 性能问题

原始实现存在以下性能问题：

| seq_len | FlashInfer | 原实现 | 差距 |
|---------|------------|--------|------|
| 2 | 0.006ms | 0.033ms | **~5x** |
| 65536 | 0.95ms | 1.63ms | **~1.7x** |

小序列时性能差距尤为明显（decode 阶段最常见的场景）。

### 原因分析

| 问题 | 原实现 | 优化后 |
|------|--------|--------|
| 内存访问 | 逐元素标量加载 | 向量化加载 (16 bytes) |
| 并行度 | 1 block = 1 token | 自适应 head 并行 |
| Grid 配置 | `dim3(num_tokens)` | `dim3(nblks_x, num_heads)` |
| 三角函数 | 预计算 cache | 支持 `__sincosf` 运行时计算 |

## 文件结构

```
sgl-kernel/csrc/multimodal/
├── rope_kernels.cuh              # FlashInfer 风格 kernel (合并 cos_sin_cache)
├── rope.cu                       # FlashInfer 风格 PyTorch 绑定
├── compile.sh                    # 编译 rope.so
├── test_rope.py                  # 测试 rope.so
├── bench_rope.py                 # 性能测试 rope.so
│
├── rotary_embedding.cu           # vLLM 风格 kernel (分离 cos/sin)
├── compile_rotary_embedding.sh   # 编译 rotary_embedding.so
├── test_rotary_embedding.py      # 测试 rotary_embedding.so
├── bench_rotary.py               # 性能测试 (vLLM vs sgl_kernel)
│
└── README.md                     # 本文档
```

### 两种实现的区别

| 实现 | 文件 | 接口风格 | cos/sin 格式 | 典型用户 |
|------|------|----------|--------------|----------|
| FlashInfer 风格 | `rope.cu` | `apply_rope_pos_ids_cos_sin_cache(q, k, q_out, k_out, cos_sin_cache, pos_ids)` | 合并 `[cos..., sin...]` | FlashInfer, sgl_kernel |
| vLLM 风格 | `rotary_embedding.cu` | `rotary_embedding(cos, sin, query, key, head_size, is_neox)` | 分离 cos, sin | vLLM, sgl_kernel |

## API 设计

### apply_rope_pos_ids_cos_sin_cache

预计算 cos/sin cache 模式，**性能最优**。

```python
rope.apply_rope_pos_ids_cos_sin_cache(
    q: torch.Tensor,           # (nnz, num_qo_heads, head_dim), fp16/bf16
    k: torch.Tensor,           # (nnz, num_kv_heads, head_dim), fp16/bf16
    q_rope: torch.Tensor,      # 输出，同 q shape
    k_rope: torch.Tensor,      # 输出，同 k shape
    cos_sin_cache: torch.Tensor,  # (max_seq_len, rotary_dim), fp32
    pos_ids: torch.Tensor,     # (nnz,), int32/int64
    interleave: bool = False   # NEOX style (False) 或 GPT-J style (True)
)
```

**cos_sin_cache 格式**: `[cos_0, cos_1, ..., cos_{d/2-1}, sin_0, sin_1, ..., sin_{d/2-1}]`

### apply_rope_pos_ids

运行时计算 cos/sin，适用于无 cache 的场景。

```python
rope.apply_rope_pos_ids(
    q, k, q_rope, k_rope,
    pos_ids: torch.Tensor,     # (nnz,)
    rotary_dim: int,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0
)
```

### apply_rope

支持 variable-length batching 的 indptr + offset 模式。

```python
rope.apply_rope(
    q, k, q_rope, k_rope,
    indptr: torch.Tensor,      # (batch_size + 1,), 每个 batch 的起始位置
    offsets: torch.Tensor,     # (batch_size,), 每个 batch 的位置偏移
    rotary_dim: int,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 10000.0
)
```

### rope_quantize

RoPE + FP8 量化融合，用于 DeepSeek/MLA 架构。

```python
rope.rope_quantize(
    # 输入 (fp16/bf16)
    q_rope_in: torch.Tensor,   # (nnz, num_qo_heads, rope_dim)
    k_rope_in: torch.Tensor,   # (nnz, num_kv_heads, rope_dim) 或 (nnz, rope_dim) for MLA
    q_nope_in: torch.Tensor,   # (nnz, num_qo_heads, no_rope_dim)
    k_nope_in: torch.Tensor,   # (nnz, num_kv_heads, no_rope_dim) 或 (nnz, no_rope_dim) for MLA
    # 输出 (fp8)
    q_rope_out: torch.Tensor,
    k_rope_out: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope_out: torch.Tensor,
    # RoPE 参数
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    interleave: bool = False
)
```

---

## rotary_embedding.cu (vLLM 风格)

vLLM 兼容的分离 cos/sin 接口。

### 编译

```bash
cd sgl-kernel/csrc/multimodal
bash compile_rotary_embedding.sh
```

### API

```python
import rotary_embedding

rotary_embedding.rotary_embedding(
    cos: torch.Tensor,      # (num_tokens, rotary_dim/2), 已按 positions 索引
    sin: torch.Tensor,      # (num_tokens, rotary_dim/2), 已按 positions 索引
    query: torch.Tensor,    # (num_tokens, num_heads * head_size), in-place 修改
    key: torch.Tensor,      # (num_tokens, num_kv_heads * head_size), in-place 修改, 可为 None
    head_size: int,
    is_neox: bool = True    # True = NeoX style (pairs at 0,1,2,3...), False = GPT-J style
)
```

### 测试

```bash
python test_rotary_embedding.py
```

### 与 sgl_kernel.rotary_embedding 对比

两者接口完全一致，可直接替换：

```python
# sgl_kernel 已有实现
from sgl_kernel.rotary_embedding import rotary_embedding
rotary_embedding(cos, sin, query, key, head_size, is_neox)

# 本实现 (编译后)
import rotary_embedding
rotary_embedding.rotary_embedding(cos, sin, query, key, head_size, is_neox)
```

---

## 核心优化技术

### 向量化内存访问

使用 `flashinfer::vec_t<T, vec_size>` 进行向量化加载/存储：

```cuda
// 一次处理 vec_size 个元素 (fp16 时为 8)
constexpr uint32_t vec_size = 32 / sizeof(DType);  // = 16 bytes
vec_t<float, vec_size> vec;
vec.cast_load(x + threadIdx.x * vec_size);  // 向量化加载
```

### 自适应 Head 并行

根据 SM occupancy 自动选择最优 kernel：

```cuda
// 计算可用 CTA 数量
int num_blocks_per_sm = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, num_threads, 0);
uint32_t num_ctas = num_blocks_per_sm * num_sms;

// 自适应选择
if (nblks_x >= num_ctas) {
    // Token 数量多，使用单维度 grid
    dim3 nblks(nblks_x);
    kernel_0<<<nblks, nthrs, 0, stream>>>(...);
} else {
    // Token 数量少，启用 head 并行
    dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
    kernel_1<<<nblks, nthrs, 0, stream>>>(...);  // HeadParallelism kernel
}
```

### 2D 线程块配置

使用 2D 线程块同时处理多个 token：

```cuda
// Grid/Block 配置
constexpr uint32_t bdx = HEAD_DIM / vec_size;  // X方向: 处理一个 head
uint32_t bdy = num_threads / bdx;              // Y方向: 多 token 并行
dim3 nthrs(bdx, bdy);

// Kernel 中的索引计算
const uint32_t idx = bx * bdy + ty;  // 当前处理的 token
```

### 高效三角函数计算

使用 `__sincosf` 同时计算 sin 和 cos：

```cuda
float embed = float(pos) * freq[i];
float cos_val, sin_val;
__sincosf(embed, &sin_val, &cos_val);  // 一次调用得到两个值
```

## RoPE 公式

**非交错模式 (NEOX style, interleave=False)**:
```
x' = [x_0, x_1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]
y' = [x_{d/2}, ..., x_{d-1}, x_0, ..., x_{d/2-1}]  (permuted)
out = x' * cos - y' * sin  (for first half)
out = x' * cos + y' * sin  (for second half)
```

**交错模式 (GPT-J style, interleave=True)**:
```
x' = [x_0, x_1, x_2, x_3, ...]
out[0] = x[0] * cos[0] - x[1] * sin[0]
out[1] = x[1] * cos[0] + x[0] * sin[0]
out[2] = x[2] * cos[1] - x[3] * sin[1]
...
```

## 编译和使用

### 编译

```bash
cd sgl-kernel/csrc/multimodal
chmod +x compile.sh
./compile.sh
```

**注意**: 编译需要 FlashInfer 的头文件。如果找不到，编译脚本会自动克隆 FlashInfer 仓库。

### 测试

```bash
# 测试优化后的实现
python test_rope.py

# Benchmark 对比 (需要安装 vllm, flashinfer, sgl_kernel)
python bench_rope.py
```

### Benchmark 覆盖的实现

`bench_rope.py` 支持以下 provider:

| Provider | 说明 | 依赖 |
|----------|------|------|
| `flashinfer` | FlashInfer 原生实现 | `pip install flashinfer` |
| `vllm` | vLLM CUDA 实现 | `pip install vllm` |
| `vllm_native` | vLLM PyTorch 原生实现 | `pip install vllm` |
| `sgl_existing` | sgl_kernel 已有实现 | `pip install sgl-kernel` |
| `sgl_opt` | 本项目优化实现 | `bash compile.sh` |
| `sgl_opt_inplace` | 本项目优化实现 (in-place) | `bash compile.sh` |
| `sgl_opt_onthefly` | 本项目优化实现 (运行时计算) | `bash compile.sh` |

### 集成使用

```python
import torch
import sys
sys.path.insert(0, '/path/to/sgl-kernel/csrc/multimodal')
import rope

# 创建 cos/sin cache
def compute_cos_sin_cache(max_seq_len, rotary_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat((cos, sin), dim=-1).cuda()

cos_sin_cache = compute_cos_sin_cache(8192, 128)

# 使用 RoPE
q = torch.randn(256, 32, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(256, 8, 128, dtype=torch.bfloat16, device='cuda')
q_out = torch.empty_like(q)
k_out = torch.empty_like(k)
pos_ids = torch.arange(256, device='cuda', dtype=torch.int32)

rope.apply_rope_pos_ids_cos_sin_cache(
    q, k, q_out, k_out,
    cos_sin_cache, pos_ids,
    interleave=False  # NEOX style
)
```

## 依赖

- CUDA >= 11.8
- PyTorch >= 2.0
- FlashInfer (用于 `vec_dtypes.cuh`)
- GPU: SM >= 80 (建议 SM 90 for Hopper)

## 性能对比

| seq_len | FlashInfer | 优化后 | 预期加速 |
|---------|------------|--------|----------|
| 2 | 0.006ms | ~0.007ms | ~1x |
| 64 | 0.007ms | ~0.008ms | ~1x |
| 1024 | 0.022ms | ~0.025ms | ~1x |
| 65536 | 0.95ms | ~1.0ms | ~1x |

优化后的实现应与 FlashInfer 性能相当。

## 后续工作

1. [ ] 支持 Llama 3.1 风格的频率插值 (`smooth_a`, `smooth_b`)
2. [ ] 支持 PDL (Programmatic Dependent Launch) 以优化 kernel 调度
3. [ ] 集成 RopeQuantizeAppendPagedKVCache 以支持 fused KV cache 写入
4. [ ] 添加更多 head_dim 支持 (32, 48, 96 等)

