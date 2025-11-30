/*
 * Copyright (c) 2025 by SGLang team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "rope_kernels.cuh"

// Dispatch dtype macro - use CUDA native types for FlashInfer vec_t compatibility
// at::Half is binary compatible with half, at::BFloat16 with nv_bfloat16
#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                         \
  [&] {                                                                  \
    switch (TYPE) {                                                      \
      case at::ScalarType::Half: {                                       \
        using scalar_t = half;                                           \
        return __VA_ARGS__();                                            \
      }                                                                  \
      case at::ScalarType::BFloat16: {                                   \
        using scalar_t = nv_bfloat16;                                    \
        return __VA_ARGS__();                                            \
      }                                                                  \
      case at::ScalarType::Float: {                                      \
        using scalar_t = float;                                          \
        return __VA_ARGS__();                                            \
      }                                                                  \
      default:                                                           \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");  \
    }                                                                    \
  }()

#define DISPATCH_INDEX_TYPES(TYPE, NAME, ...)                            \
  [&] {                                                                  \
    switch (TYPE) {                                                      \
      case at::ScalarType::Int: {                                        \
        using index_t = int32_t;                                         \
        return __VA_ARGS__();                                            \
      }                                                                  \
      case at::ScalarType::Long: {                                       \
        using index_t = int64_t;                                         \
        return __VA_ARGS__();                                            \
      }                                                                  \
      default:                                                           \
        AT_ERROR(#NAME, " not implemented for index type '", toString(TYPE), "'"); \
    }                                                                    \
  }()

/******************* API: apply_rope_pos_ids_cos_sin_cache *******************/

void apply_rope_pos_ids_cos_sin_cache(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor q_rope,
    torch::Tensor k_rope,
    torch::Tensor cos_sin_cache,
    torch::Tensor pos_ids,
    bool interleave) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be a CUDA tensor");
  TORCH_CHECK(pos_ids.is_cuda(), "pos_ids must be a CUDA tensor");
  
  TORCH_CHECK(q.dim() == 3, "q must be 3D: (nnz, num_qo_heads, head_dim)");
  TORCH_CHECK(k.dim() == 3, "k must be 3D: (nnz, num_kv_heads, head_dim)");
  TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must be 2D: (max_seq_len, rotary_dim)");
  
  uint32_t nnz = q.size(0);
  uint32_t num_qo_heads = q.size(1);
  uint32_t num_kv_heads = k.size(1);
  uint32_t head_dim = q.size(2);
  uint32_t rotary_dim = cos_sin_cache.size(1);
  
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  DISPATCH_FLOATING_TYPES(q.scalar_type(), "apply_rope_pos_ids_cos_sin_cache", [&] {
    DISPATCH_INDEX_TYPES(pos_ids.scalar_type(), "apply_rope_pos_ids_cos_sin_cache", [&] {
      cudaError_t status = rope::BatchQKApplyRotaryPosIdsCosSinCache(
          static_cast<scalar_t*>(q.data_ptr()),
          static_cast<scalar_t*>(k.data_ptr()),
          static_cast<scalar_t*>(q_rope.data_ptr()),
          static_cast<scalar_t*>(k_rope.data_ptr()),
          static_cast<float*>(cos_sin_cache.data_ptr()),
          static_cast<index_t*>(pos_ids.data_ptr()),
          nnz, num_qo_heads, num_kv_heads, rotary_dim, head_dim,
          q_stride_n, q_stride_h, k_stride_n, k_stride_h,
          q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
          interleave, stream);
      TORCH_CHECK(status == cudaSuccess, "BatchQKApplyRotaryPosIdsCosSinCache failed: ", cudaGetErrorString(status));
      return true;
    });
    return true;
  });
}

/******************* API: apply_rope_pos_ids *******************/

void apply_rope_pos_ids(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor q_rope,
    torch::Tensor k_rope,
    torch::Tensor pos_ids,
    int64_t rotary_dim,
    bool interleave,
    double rope_scale,
    double rope_theta) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(pos_ids.is_cuda(), "pos_ids must be a CUDA tensor");
  
  TORCH_CHECK(q.dim() == 3, "q must be 3D: (nnz, num_qo_heads, head_dim)");
  TORCH_CHECK(k.dim() == 3, "k must be 3D: (nnz, num_kv_heads, head_dim)");
  
  uint32_t nnz = q.size(0);
  uint32_t num_qo_heads = q.size(1);
  uint32_t num_kv_heads = k.size(1);
  uint32_t head_dim = q.size(2);
  
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  DISPATCH_FLOATING_TYPES(q.scalar_type(), "apply_rope_pos_ids", [&] {
    DISPATCH_INDEX_TYPES(pos_ids.scalar_type(), "apply_rope_pos_ids", [&] {
      cudaError_t status = rope::BatchQKApplyRotaryPosIds(
          static_cast<scalar_t*>(q.data_ptr()),
          static_cast<scalar_t*>(k.data_ptr()),
          static_cast<scalar_t*>(q_rope.data_ptr()),
          static_cast<scalar_t*>(k_rope.data_ptr()),
          static_cast<index_t*>(pos_ids.data_ptr()),
          nnz, num_qo_heads, num_kv_heads, rotary_dim, head_dim,
          q_stride_n, q_stride_h, k_stride_n, k_stride_h,
          q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
          interleave, (float)rope_scale, (float)rope_theta, stream);
      TORCH_CHECK(status == cudaSuccess, "BatchQKApplyRotaryPosIds failed: ", cudaGetErrorString(status));
      return true;
    });
    return true;
  });
}

/******************* API: apply_rope (indptr + offset) *******************/

void apply_rope(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor q_rope,
    torch::Tensor k_rope,
    torch::Tensor indptr,
    torch::Tensor offsets,
    int64_t rotary_dim,
    bool interleave,
    double rope_scale,
    double rope_theta) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(indptr.is_cuda(), "indptr must be a CUDA tensor");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor");
  
  TORCH_CHECK(q.dim() == 3, "q must be 3D: (nnz, num_qo_heads, head_dim)");
  TORCH_CHECK(k.dim() == 3, "k must be 3D: (nnz, num_kv_heads, head_dim)");
  TORCH_CHECK(indptr.dim() == 1, "indptr must be 1D: (batch_size + 1,)");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D: (batch_size,)");
  
  uint32_t batch_size = offsets.size(0);
  uint32_t num_qo_heads = q.size(1);
  uint32_t num_kv_heads = k.size(1);
  uint32_t head_dim = q.size(2);
  
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);
  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  DISPATCH_FLOATING_TYPES(q.scalar_type(), "apply_rope", [&] {
    DISPATCH_INDEX_TYPES(indptr.scalar_type(), "apply_rope", [&] {
      cudaError_t status = rope::BatchQKApplyRotary(
          static_cast<scalar_t*>(q.data_ptr()),
          static_cast<scalar_t*>(k.data_ptr()),
          static_cast<scalar_t*>(q_rope.data_ptr()),
          static_cast<scalar_t*>(k_rope.data_ptr()),
          static_cast<index_t*>(indptr.data_ptr()),
          static_cast<index_t*>(offsets.data_ptr()),
          batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim,
          q_stride_n, q_stride_h, k_stride_n, k_stride_h,
          q_rope_stride_n, q_rope_stride_h, k_rope_stride_n, k_rope_stride_h,
          interleave, (float)rope_scale, (float)rope_theta, stream);
      TORCH_CHECK(status == cudaSuccess, "BatchQKApplyRotary failed: ", cudaGetErrorString(status));
      return true;
    });
    return true;
  });
}

/******************* API: rope_quantize *******************/

void rope_quantize(
    torch::Tensor q_rope_in,
    torch::Tensor k_rope_in,
    torch::Tensor q_nope_in,
    torch::Tensor k_nope_in,
    torch::Tensor q_rope_out,
    torch::Tensor k_rope_out,
    torch::Tensor q_nope_out,
    torch::Tensor k_nope_out,
    torch::Tensor cos_sin_cache,
    torch::Tensor pos_ids,
    double quant_scale_q,
    double quant_scale_kv,
    bool interleave) {
  TORCH_CHECK(q_rope_in.is_cuda(), "q_rope_in must be a CUDA tensor");
  TORCH_CHECK(k_rope_in.is_cuda(), "k_rope_in must be a CUDA tensor");
  TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be a CUDA tensor");
  TORCH_CHECK(pos_ids.is_cuda(), "pos_ids must be a CUDA tensor");
  
  // Extract dimensions
  uint32_t nnz = q_rope_in.size(0);
  uint32_t num_qo_heads = q_rope_in.size(1);
  uint32_t rope_dim = q_rope_in.size(2);
  uint32_t no_rope_dim = q_nope_in.size(2);
  
  // For K tensors, handle both 2D (MLA) and 3D (GQA/MHA) cases
  uint32_t num_kv_heads;
  if (k_rope_in.dim() == 2) {
    num_kv_heads = 1;
  } else {
    num_kv_heads = k_rope_in.size(1);
  }
  
  // Q strides
  size_t q_rope_in_stride_n = q_rope_in.stride(0);
  size_t q_rope_in_stride_h = q_rope_in.stride(1);
  size_t q_nope_in_stride_n = q_nope_in.stride(0);
  size_t q_nope_in_stride_h = q_nope_in.stride(1);
  size_t q_rope_out_stride_n = q_rope_out.stride(0);
  size_t q_rope_out_stride_h = q_rope_out.stride(1);
  size_t q_nope_out_stride_n = q_nope_out.stride(0);
  size_t q_nope_out_stride_h = q_nope_out.stride(1);
  
  // K strides
  size_t k_rope_in_stride, k_rope_in_stride_h;
  size_t k_nope_in_stride, k_nope_in_stride_h;
  size_t k_rope_out_stride, k_rope_out_stride_h;
  size_t k_nope_out_stride, k_nope_out_stride_h;
  
  if (k_rope_in.dim() == 2) {
    k_rope_in_stride = k_rope_in.stride(0);
    k_rope_in_stride_h = k_rope_in_stride;
    k_nope_in_stride = k_nope_in.stride(0);
    k_nope_in_stride_h = k_nope_in_stride;
    k_rope_out_stride = k_rope_out.stride(0);
    k_rope_out_stride_h = k_rope_out_stride;
    k_nope_out_stride = k_nope_out.stride(0);
    k_nope_out_stride_h = k_nope_out_stride;
  } else {
    k_rope_in_stride = k_rope_in.stride(0);
    k_rope_in_stride_h = k_rope_in.stride(1);
    k_nope_in_stride = k_nope_in.stride(0);
    k_nope_in_stride_h = k_nope_in.stride(1);
    k_rope_out_stride = k_rope_out.stride(0);
    k_rope_out_stride_h = k_rope_out.stride(1);
    k_nope_out_stride = k_nope_out.stride(0);
    k_nope_out_stride_h = k_nope_out.stride(1);
  }
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // Dispatch based on input dtype (fp16/bf16) and output dtype (fp8)
  auto out_dtype = q_rope_out.scalar_type();
  
  DISPATCH_FLOATING_TYPES(q_rope_in.scalar_type(), "rope_quantize", [&] {
    DISPATCH_INDEX_TYPES(pos_ids.scalar_type(), "rope_quantize", [&] {
      if (out_dtype == at::ScalarType::Float8_e4m3fn) {
        cudaError_t status = rope::RopeQuantize(
            static_cast<scalar_t*>(q_rope_in.data_ptr()),
            static_cast<scalar_t*>(k_rope_in.data_ptr()),
            static_cast<scalar_t*>(q_nope_in.data_ptr()),
            static_cast<scalar_t*>(k_nope_in.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(q_rope_out.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(k_rope_out.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(q_nope_out.data_ptr()),
            static_cast<__nv_fp8_e4m3*>(k_nope_out.data_ptr()),
            static_cast<float*>(cos_sin_cache.data_ptr()),
            static_cast<index_t*>(pos_ids.data_ptr()),
            nnz, num_qo_heads, num_kv_heads, rope_dim, no_rope_dim,
            q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n, q_nope_in_stride_h,
            q_rope_out_stride_n, q_rope_out_stride_h, q_nope_out_stride_n, q_nope_out_stride_h,
            k_rope_in_stride, k_rope_in_stride_h, k_nope_in_stride, k_nope_in_stride_h,
            k_rope_out_stride, k_rope_out_stride_h, k_nope_out_stride, k_nope_out_stride_h,
            (float)quant_scale_q, (float)quant_scale_kv, interleave, stream);
        TORCH_CHECK(status == cudaSuccess, "RopeQuantize failed: ", cudaGetErrorString(status));
      } else {
        AT_ERROR("rope_quantize only supports Float8_e4m3fn output dtype");
      }
      return true;
    });
    return true;
  });
}

/******************* Python bindings *******************/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("apply_rope_pos_ids_cos_sin_cache", &apply_rope_pos_ids_cos_sin_cache,
        "Apply RoPE with precomputed cos/sin cache and position IDs",
        py::arg("q"), py::arg("k"), py::arg("q_rope"), py::arg("k_rope"),
        py::arg("cos_sin_cache"), py::arg("pos_ids"), py::arg("interleave") = false);
  
  m.def("apply_rope_pos_ids", &apply_rope_pos_ids,
        "Apply RoPE with position IDs (compute cos/sin on-the-fly)",
        py::arg("q"), py::arg("k"), py::arg("q_rope"), py::arg("k_rope"),
        py::arg("pos_ids"), py::arg("rotary_dim"), py::arg("interleave") = false,
        py::arg("rope_scale") = 1.0, py::arg("rope_theta") = 10000.0);
  
  m.def("apply_rope", &apply_rope,
        "Apply RoPE with indptr and offsets",
        py::arg("q"), py::arg("k"), py::arg("q_rope"), py::arg("k_rope"),
        py::arg("indptr"), py::arg("offsets"), py::arg("rotary_dim"),
        py::arg("interleave") = false, py::arg("rope_scale") = 1.0,
        py::arg("rope_theta") = 10000.0);
  
  m.def("rope_quantize", &rope_quantize,
        "Apply RoPE and quantize to FP8",
        py::arg("q_rope_in"), py::arg("k_rope_in"), py::arg("q_nope_in"), py::arg("k_nope_in"),
        py::arg("q_rope_out"), py::arg("k_rope_out"), py::arg("q_nope_out"), py::arg("k_nope_out"),
        py::arg("cos_sin_cache"), py::arg("pos_ids"),
        py::arg("quant_scale_q") = 1.0, py::arg("quant_scale_kv") = 1.0,
        py::arg("interleave") = false);
}

