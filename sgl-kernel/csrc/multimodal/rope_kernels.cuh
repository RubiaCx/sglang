/*
 * Copyright (c) 2025 by SGLang team.
 * Adapted from FlashInfer project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef ROPE_KERNELS_CUH_
#define ROPE_KERNELS_CUH_

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

#include <flashinfer/vec_dtypes.cuh>

namespace rope {

using flashinfer::vec_t;

/******************* Helper macros *******************/

#define ROPE_CUDA_CALL(func)                                             \
  {                                                                      \
    cudaError_t e = (func);                                              \
    if (e != cudaSuccess) {                                              \
      return e;                                                          \
    }                                                                    \
  }

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)                       \
  [&]() {                                                                \
    if (head_dim == 64) {                                                \
      constexpr uint32_t HEAD_DIM = 64;                                  \
      __VA_ARGS__;                                                       \
    } else if (head_dim == 128) {                                        \
      constexpr uint32_t HEAD_DIM = 128;                                 \
      __VA_ARGS__;                                                       \
    } else if (head_dim == 256) {                                        \
      constexpr uint32_t HEAD_DIM = 256;                                 \
      __VA_ARGS__;                                                       \
    } else {                                                             \
      constexpr uint32_t HEAD_DIM = 128;                                 \
      __VA_ARGS__;                                                       \
    }                                                                    \
  }()

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...)                 \
  [&]() {                                                                \
    if (interleave) {                                                    \
      constexpr bool INTERLEAVE = true;                                  \
      __VA_ARGS__;                                                       \
    } else {                                                             \
      constexpr bool INTERLEAVE = false;                                 \
      __VA_ARGS__;                                                       \
    }                                                                    \
  }()

#define DISPATCH_ROPE_DIM(rope_dim, ROPE_DIM, ...)                       \
  [&]() {                                                                \
    if (rope_dim == 64) {                                                \
      constexpr uint32_t ROPE_DIM = 64;                                  \
      __VA_ARGS__;                                                       \
    } else if (rope_dim == 128) {                                        \
      constexpr uint32_t ROPE_DIM = 128;                                 \
      __VA_ARGS__;                                                       \
    } else if (rope_dim == 256) {                                        \
      constexpr uint32_t ROPE_DIM = 256;                                 \
      __VA_ARGS__;                                                       \
    } else {                                                             \
      constexpr uint32_t ROPE_DIM = 128;                                 \
      __VA_ARGS__;                                                       \
    }                                                                    \
  }()

/******************* Helper functions *******************/

__device__ __forceinline__ size_t get_elem_offset_impl(
    uint32_t idx, uint32_t head_idx, uint32_t feat_idx,
    size_t stride_n, size_t stride_h) {
  return idx * stride_n + head_idx * stride_h + feat_idx;
}

/******************* Core RoPE functions *******************/

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin(
    const T* x, const vec_t<float, vec_size>& cos, const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(x + ((threadIdx.x * vec_size < rotary_dim / 2)
                                    ? threadIdx.x * vec_size + rotary_dim / 2
                                    : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] = vec[i] * cos[i] +
          ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin[i];
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin_interleave_reuse_half(
    const T* x, const vec_t<float, vec_size>& cos, const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] = vec[i] * cos[i / 2] +
               ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i / 2];
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(x + ((threadIdx.x * vec_size < rotary_dim / 2)
                                    ? threadIdx.x * vec_size + rotary_dim / 2
                                    : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos_val, sin_val;
      __sincosf(embed, &sin_val, &cos_val);
      vec[i] = vec[i] * cos_val +
          ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin_val;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_interleave(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos_val, sin_val;
      __sincosf(embed, &sin_val, &cos_val);
      vec[i] = vec[i] * cos_val + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin_val;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin_interleave(
    const T* x, const vec_t<float, vec_size>& cos, const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] = vec[i] * cos[i] + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i];
    }
  }
  return vec;
}

/******************* Kernel: BatchQKApplyRotaryPosIdsCosSinCache *******************/

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];
    const int half_rotary_dim = rotary_dim / 2;

    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr = q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr = k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];
    const int half_rotary_dim = rotary_dim / 2;

    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr = q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr = k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

/******************* Kernel: BatchQKApplyRotaryPosIds *******************/

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    float rope_rcp_scale, float rope_rcp_theta) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }
      freq[i] = freq[i] * rope_rcp_scale;
    }
  }

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float embed = float(pos) * freq[i];
        __sincosf(embed, &sin[i], &cos[i]);
      }
    }

#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr = q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr = k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsHeadParallelismKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    float rope_rcp_scale, float rope_rcp_theta) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }
      freq[i] = freq[i] * rope_rcp_scale;
    }
  }

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float embed = float(pos) * freq[i];
        __sincosf(embed, &sin[i], &cos[i]);
      }
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr = q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr = k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

/******************* Kernel: BatchQKApplyRotary (indptr + offset) *******************/

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    float rope_rcp_scale, float rope_rcp_theta) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }
      freq[i] = freq[i] * rope_rcp_scale;
    }
  }

  if (bx < batch_size * num_qo_heads) {
    const uint32_t batch_idx = bx / num_qo_heads;
    const uint32_t qo_head_idx = bx % num_qo_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> q_vec;
      if (i * bdy + ty < seq_len) {
        DType* q_ptr = q + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                                q_stride_n, q_stride_h);
        DType* q_rope_ptr = q_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                          q_rope_stride_n, q_rope_stride_h);
        if constexpr (interleave) {
          q_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty, rotary_dim);
        } else {
          q_vec = vec_apply_llama_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        q_vec.cast_store(q_rope_ptr + tx * vec_size);
      }
    }
  } else {
    uint32_t batch_idx = (bx - batch_size * num_qo_heads) / num_kv_heads;
    uint32_t kv_head_idx = (bx - batch_size * num_qo_heads) % num_kv_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> k_vec;
      if (i * bdy + ty < seq_len) {
        DType* k_ptr = k + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                                k_stride_n, k_stride_h);
        DType* k_rope_ptr = k_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                          k_rope_stride_n, k_rope_stride_h);
        if constexpr (interleave) {
          k_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty, rotary_dim);
        } else {
          k_vec = vec_apply_llama_rope<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        k_vec.cast_store(k_rope_ptr + tx * vec_size);
      }
    }
  }
}

/******************* Kernel: RopeQuantize *******************/

template <bool interleave, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType,
          typename QuantType>
__global__ void RopeQuantizeKernel(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, QuantType* q_rope_out,
    QuantType* k_rope_out, QuantType* q_nope_out, QuantType* k_nope_out,
    float* __restrict__ cos_sin_cache, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rope_dim, uint32_t no_rope_dim,
    size_t q_rope_in_stride_n, size_t q_rope_in_stride_h, size_t q_nope_in_stride_n,
    size_t q_nope_in_stride_h, size_t q_rope_out_stride_n, size_t q_rope_out_stride_h,
    size_t q_nope_out_stride_n, size_t q_nope_out_stride_h, size_t k_rope_in_stride,
    size_t k_rope_in_stride_h, size_t k_nope_in_stride, size_t k_nope_in_stride_h,
    size_t k_rope_out_stride, size_t k_rope_out_stride_h, size_t k_nope_out_stride,
    size_t k_nope_out_stride_h, float quant_scale_q, float quant_scale_kv) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  uint32_t bdy = blockDim.y;

  uint32_t rope_chunks = 1;
  uint32_t no_rope_chunks = (no_rope_dim + rope_dim - 1) / rope_dim;

  uint32_t q_rope_end = num_qo_heads * rope_chunks;
  uint32_t k_rope_end = q_rope_end + num_kv_heads * rope_chunks;
  uint32_t k_nope_end = k_rope_end + num_kv_heads * no_rope_chunks;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];
    const int half_rope_dim = rope_dim / 2;

    if ((tx * vec_size < rope_dim) && (by < k_rope_end)) {
      int sin_offset = rope_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rope_dim;
      }
      cos.load(cos_sin_cache + (pos * rope_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rope_dim) + (sin_offset + vec_idx));
    }

    if (by < q_rope_end) {
      // Q RoPE
      uint32_t q_head_idx = by;
      DType* q_rope_in_ptr = q_rope_in + get_elem_offset_impl(idx, q_head_idx, 0, q_rope_in_stride_n, q_rope_in_stride_h);
      QuantType* q_rope_out_ptr = q_rope_out + get_elem_offset_impl(idx, q_head_idx, 0, q_rope_out_stride_n, q_rope_out_stride_h);

      vec_t<float, vec_size> q_rope_vec;
      if constexpr (interleave) {
        q_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_rope_in_ptr, cos, sin, rope_dim);
      } else {
        q_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        q_rope_vec[i] = q_rope_vec[i] * quant_scale_q;
      }
      q_rope_vec.cast_store(q_rope_out_ptr + tx * vec_size);

    } else if (by < k_rope_end) {
      // K RoPE
      uint32_t k_head_idx = by - q_rope_end;
      DType* k_rope_in_ptr = k_rope_in + get_elem_offset_impl(idx, k_head_idx, 0, k_rope_in_stride, k_rope_in_stride_h);
      QuantType* k_rope_out_ptr = k_rope_out + get_elem_offset_impl(idx, k_head_idx, 0, k_rope_out_stride, k_rope_out_stride_h);

      vec_t<float, vec_size> k_rope_vec;
      if constexpr (interleave) {
        k_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_rope_in_ptr, cos, sin, rope_dim);
      } else {
        k_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        k_rope_vec[i] = k_rope_vec[i] * quant_scale_kv;
      }
      k_rope_vec.cast_store(k_rope_out_ptr + tx * vec_size);

    } else if (by < k_nope_end) {
      // K Non-RoPE
      uint32_t k_head_idx = (by - k_rope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_rope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_dim;

      DType* k_nope_in_ptr = k_nope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset, k_nope_in_stride, k_nope_in_stride_h);
      QuantType* k_nope_out_ptr = k_nope_out + get_elem_offset_impl(idx, k_head_idx, elem_offset, k_nope_out_stride, k_nope_out_stride_h);

      vec_t<float, vec_size> k_nope_vec;
      k_nope_vec.cast_load(k_nope_in_ptr + tx * vec_size);
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        k_nope_vec[i] = k_nope_vec[i] * quant_scale_kv;
      }
      k_nope_vec.cast_store(k_nope_out_ptr + tx * vec_size);

    } else {
      // Q Non-RoPE
      uint32_t q_head_idx = (by - k_nope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_nope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_dim;

      DType* q_nope_in_ptr = q_nope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_in_stride_n, q_nope_in_stride_h);
      QuantType* q_nope_out_ptr = q_nope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_out_stride_n, q_nope_out_stride_h);

      vec_t<float, vec_size> q_nope_vec;
      q_nope_vec.cast_load(q_nope_in_ptr + tx * vec_size);
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        q_nope_vec[i] = q_nope_vec[i] * quant_scale_q;
      }
      q_nope_vec.cast_store(q_nope_out_ptr + tx * vec_size);
    }
  }
}

/******************* Host functions *******************/

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryPosIdsCosSinCache(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* cos_sin_cache, IdType* pos_ids,
    uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim,
    uint32_t head_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    bool interleave, cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  ROPE_CUDA_CALL(cudaGetDevice(&dev_id));
  ROPE_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  if (interleave) {
    constexpr bool INTERLEAVE = true;
    if (head_dim == 64) {
      constexpr uint32_t HEAD_DIM = 64;
      constexpr uint32_t vec_size = (16 / sizeof(DType) > HEAD_DIM / 32) ? (16 / sizeof(DType)) : (HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = (128U > bdx) ? 128U : bdx;
      uint32_t bdy = num_threads / bdx;
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;

      auto kernel_0 = BatchQKApplyRotaryPosIdsCosSinCacheKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      int num_blocks_per_sm_0 = 0;
      ROPE_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm_0, kernel_0, num_threads, 0));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

      if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
        dim3 nblks(nblks_x);
        dim3 nthrs(bdx, bdy);
        kernel_0<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      } else {
        dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel_1 = BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
        kernel_1<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      }
    } else if (head_dim == 256) {
      constexpr uint32_t HEAD_DIM = 256;
      constexpr uint32_t vec_size = (16 / sizeof(DType) > HEAD_DIM / 32) ? (16 / sizeof(DType)) : (HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = (128U > bdx) ? 128U : bdx;
      uint32_t bdy = num_threads / bdx;
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;

      auto kernel_0 = BatchQKApplyRotaryPosIdsCosSinCacheKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      int num_blocks_per_sm_0 = 0;
      ROPE_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm_0, kernel_0, num_threads, 0));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

      if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
        dim3 nblks(nblks_x);
        dim3 nthrs(bdx, bdy);
        kernel_0<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      } else {
        dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel_1 = BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
        kernel_1<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      }
    } else {
      // Default: head_dim == 128
      constexpr uint32_t HEAD_DIM = 128;
      constexpr uint32_t vec_size = (16 / sizeof(DType) > HEAD_DIM / 32) ? (16 / sizeof(DType)) : (HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = (128U > bdx) ? 128U : bdx;
      uint32_t bdy = num_threads / bdx;
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;

      auto kernel_0 = BatchQKApplyRotaryPosIdsCosSinCacheKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      int num_blocks_per_sm_0 = 0;
      ROPE_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm_0, kernel_0, num_threads, 0));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

      if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
        dim3 nblks(nblks_x);
        dim3 nthrs(bdx, bdy);
        kernel_0<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      } else {
        dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel_1 = BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
        kernel_1<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      }
    }
  } else {
    constexpr bool INTERLEAVE = false;
    if (head_dim == 64) {
      constexpr uint32_t HEAD_DIM = 64;
      constexpr uint32_t vec_size = (16 / sizeof(DType) > HEAD_DIM / 32) ? (16 / sizeof(DType)) : (HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = (128U > bdx) ? 128U : bdx;
      uint32_t bdy = num_threads / bdx;
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;

      auto kernel_0 = BatchQKApplyRotaryPosIdsCosSinCacheKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      int num_blocks_per_sm_0 = 0;
      ROPE_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm_0, kernel_0, num_threads, 0));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

      if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
        dim3 nblks(nblks_x);
        dim3 nthrs(bdx, bdy);
        kernel_0<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      } else {
        dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel_1 = BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
        kernel_1<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      }
    } else if (head_dim == 256) {
      constexpr uint32_t HEAD_DIM = 256;
      constexpr uint32_t vec_size = (16 / sizeof(DType) > HEAD_DIM / 32) ? (16 / sizeof(DType)) : (HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = (128U > bdx) ? 128U : bdx;
      uint32_t bdy = num_threads / bdx;
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;

      auto kernel_0 = BatchQKApplyRotaryPosIdsCosSinCacheKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      int num_blocks_per_sm_0 = 0;
      ROPE_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm_0, kernel_0, num_threads, 0));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

      if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
        dim3 nblks(nblks_x);
        dim3 nthrs(bdx, bdy);
        kernel_0<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      } else {
        dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel_1 = BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
        kernel_1<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      }
    } else {
      // Default: head_dim == 128
      constexpr uint32_t HEAD_DIM = 128;
      constexpr uint32_t vec_size = (16 / sizeof(DType) > HEAD_DIM / 32) ? (16 / sizeof(DType)) : (HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = (128U > bdx) ? 128U : bdx;
      uint32_t bdy = num_threads / bdx;
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;

      auto kernel_0 = BatchQKApplyRotaryPosIdsCosSinCacheKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      int num_blocks_per_sm_0 = 0;
      ROPE_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm_0, kernel_0, num_threads, 0));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

      if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
        dim3 nblks(nblks_x);
        dim3 nthrs(bdx, bdy);
        kernel_0<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      } else {
        dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel_1 = BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
        kernel_1<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h);
      }
    }
  }

  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryPosIds(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim,
    size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    bool interleave, float rope_scale, float rope_theta, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  int dev_id = 0;
  int num_sms = 0;
  ROPE_CUDA_CALL(cudaGetDevice(&dev_id));
  ROPE_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  auto launch_pos_ids_kernel = [&]<bool INTERLEAVE, uint32_t HEAD_DIM>() {
    constexpr uint32_t vec_size = (16 / sizeof(DType) > HEAD_DIM / 32) ? (16 / sizeof(DType)) : (HEAD_DIM / 32);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t num_threads = (128U > bdx) ? 128U : bdx;
    uint32_t bdy = num_threads / bdx;
    uint32_t nblks_x = (nnz + bdy - 1) / bdy;

    auto kernel_0 = BatchQKApplyRotaryPosIdsKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
    int num_blocks_per_sm_0 = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm_0, kernel_0, num_threads, 0);
    uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

    if (nblks_x >= num_ctas_0) {
      dim3 nblks(nblks_x);
      dim3 nthrs(bdx, bdy);
      kernel_0<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h, rope_rcp_scale, rope_rcp_theta);
    } else {
      dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
      dim3 nthrs(bdx, bdy);
      auto kernel_1 = BatchQKApplyRotaryPosIdsHeadParallelismKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      kernel_1<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, pos_ids, nnz, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h, rope_rcp_scale, rope_rcp_theta);
    }
  };

  if (interleave) {
    if (head_dim == 64) launch_pos_ids_kernel.template operator()<true, 64>();
    else if (head_dim == 256) launch_pos_ids_kernel.template operator()<true, 256>();
    else launch_pos_ids_kernel.template operator()<true, 128>();
  } else {
    if (head_dim == 64) launch_pos_ids_kernel.template operator()<false, 64>();
    else if (head_dim == 256) launch_pos_ids_kernel.template operator()<false, 256>();
    else launch_pos_ids_kernel.template operator()<false, 128>();
  }

  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotary(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* indptr, IdType* offsets,
    uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim,
    uint32_t head_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    bool interleave, float rope_scale, float rope_theta, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;

  auto launch_rotary_kernel = [&]<bool INTERLEAVE, uint32_t HEAD_DIM>() {
    constexpr uint32_t vec_size = (16 / sizeof(DType) > HEAD_DIM / 32) ? (16 / sizeof(DType)) : (HEAD_DIM / 32);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t num_threads = (128U > bdx) ? 128U : bdx;
    uint32_t bdy = num_threads / bdx;
    dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
    dim3 nthrs(bdx, bdy);
    auto kernel = BatchQKApplyRotaryKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
    kernel<<<nblks, nthrs, 0, stream>>>(q, k, q_rope, k_rope, indptr, offsets, batch_size, num_qo_heads, num_kv_heads, rotary_dim, q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h, rope_rcp_scale, rope_rcp_theta);
  };

  if (interleave) {
    if (head_dim == 64) launch_rotary_kernel.template operator()<true, 64>();
    else if (head_dim == 256) launch_rotary_kernel.template operator()<true, 256>();
    else launch_rotary_kernel.template operator()<true, 128>();
  } else {
    if (head_dim == 64) launch_rotary_kernel.template operator()<false, 64>();
    else if (head_dim == 256) launch_rotary_kernel.template operator()<false, 256>();
    else launch_rotary_kernel.template operator()<false, 128>();
  }

  return cudaSuccess;
}

template <typename DType, typename IdType, typename QuantType>
cudaError_t RopeQuantize(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, QuantType* q_rope_out,
    QuantType* k_rope_out, QuantType* q_nope_out, QuantType* k_nope_out, float* cos_sin_cache,
    IdType* pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rope_dim,
    uint32_t no_rope_dim, size_t q_rope_in_stride_n, size_t q_rope_in_stride_h,
    size_t q_nope_in_stride_n, size_t q_nope_in_stride_h, size_t q_rope_out_stride_n,
    size_t q_rope_out_stride_h, size_t q_nope_out_stride_n, size_t q_nope_out_stride_h,
    size_t k_rope_in_stride, size_t k_rope_in_stride_h, size_t k_nope_in_stride,
    size_t k_nope_in_stride_h, size_t k_rope_out_stride, size_t k_rope_out_stride_h,
    size_t k_nope_out_stride, size_t k_nope_out_stride_h, float quant_scale_q, float quant_scale_kv,
    bool interleave, cudaStream_t stream = nullptr) {
  DISPATCH_ROPE_DIM(rope_dim, ROPE_DIM, {
    DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
      constexpr uint32_t vec_size = 32 / sizeof(DType);
      constexpr uint32_t bdx = ROPE_DIM / vec_size;
      uint32_t num_threads = 128U;
      uint32_t bdy = num_threads / bdx;
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;
      uint32_t rope_chunks = 1;
      uint32_t no_rope_chunks = (no_rope_dim + rope_dim - 1) / rope_dim;
      uint32_t total_blocks_y = num_qo_heads * rope_chunks + num_kv_heads * rope_chunks +
                                num_kv_heads * no_rope_chunks + num_qo_heads * no_rope_chunks;

      dim3 nblks(nblks_x, total_blocks_y);
      dim3 nthrs(bdx, bdy);

      auto kernel = RopeQuantizeKernel<INTERLEAVE, vec_size, bdx, DType, IdType, QuantType>;
      kernel<<<nblks, nthrs, 0, stream>>>(
          q_rope_in, k_rope_in, q_nope_in, k_nope_in, q_rope_out, k_rope_out, q_nope_out,
          k_nope_out, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rope_dim, no_rope_dim,
          q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n, q_nope_in_stride_h,
          q_rope_out_stride_n, q_rope_out_stride_h, q_nope_out_stride_n, q_nope_out_stride_h,
          k_rope_in_stride, k_rope_in_stride_h, k_nope_in_stride, k_nope_in_stride_h,
          k_rope_out_stride, k_rope_out_stride_h, k_nope_out_stride, k_nope_out_stride_h,
          quant_scale_q, quant_scale_kv);
    });
  });

  return cudaSuccess;
}

}  // namespace rope

#endif  // ROPE_KERNELS_CUH_

