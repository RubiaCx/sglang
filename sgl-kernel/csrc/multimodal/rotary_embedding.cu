/*
 * Rotary Positional Embedding (RoPE) CUDA Kernel
 *
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/extension.h>

using namespace flashinfer;

template <typename scalar_t>
__global__ void rotary_embedding_neox_kernel(
    scalar_t* __restrict__ query,            // (num_tokens, num_heads * head_size)
    scalar_t* __restrict__ key,              // (num_tokens, num_kv_heads * head_size) or nullptr
    const scalar_t* __restrict__ cos_cache,  // (num_tokens, rotary_dim/2)
    const scalar_t* __restrict__ sin_cache,  // (num_tokens, rotary_dim/2)
    const int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int rotary_dim) {
  const int embed_dim = rotary_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const int q_total = num_tokens * num_heads * embed_dim;
  if (idx < q_total) {
    const int token_idx = idx / (num_heads * embed_dim);
    const int remainder = idx % (num_heads * embed_dim);
    const int head_idx = remainder / embed_dim;
    const int rot_idx = remainder % embed_dim;

    const int q_offset = token_idx * num_heads * head_size + head_idx * head_size;
    const int x_idx = q_offset + rot_idx * 2;
    const int y_idx = x_idx + 1;

    const int cs_idx = token_idx * embed_dim + rot_idx;
    float cos_val = static_cast<float>(cos_cache[cs_idx]);
    float sin_val = static_cast<float>(sin_cache[cs_idx]);

    float x_f = static_cast<float>(query[x_idx]);
    float y_f = static_cast<float>(query[y_idx]);
    query[x_idx] = static_cast<scalar_t>(x_f * cos_val - y_f * sin_val);
    query[y_idx] = static_cast<scalar_t>(y_f * cos_val + x_f * sin_val);
  }

  if (key != nullptr) {
    const int k_total = num_tokens * num_kv_heads * embed_dim;
    const int k_idx = idx - q_total;
    if (k_idx >= 0 && k_idx < k_total) {
      const int token_idx = k_idx / (num_kv_heads * embed_dim);
      const int remainder = k_idx % (num_kv_heads * embed_dim);
      const int head_idx = remainder / embed_dim;
      const int rot_idx = remainder % embed_dim;

      const int k_offset = token_idx * num_kv_heads * head_size + head_idx * head_size;
      const int x_idx = k_offset + rot_idx * 2;
      const int y_idx = x_idx + 1;

      const int cs_idx = token_idx * embed_dim + rot_idx;
      float cos_val = static_cast<float>(cos_cache[cs_idx]);
      float sin_val = static_cast<float>(sin_cache[cs_idx]);

      float x_f = static_cast<float>(key[x_idx]);
      float y_f = static_cast<float>(key[y_idx]);
      key[x_idx] = static_cast<scalar_t>(x_f * cos_val - y_f * sin_val);
      key[y_idx] = static_cast<scalar_t>(y_f * cos_val + x_f * sin_val);
    }
  }
}

template <typename scalar_t>
__global__ void rotary_embedding_gptj_kernel(
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ cos_cache,
    const scalar_t* __restrict__ sin_cache,
    const int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int rotary_dim) {
  const int embed_dim = rotary_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int q_total = num_tokens * num_heads * embed_dim;
  if (idx < q_total) {
    const int token_idx = idx / (num_heads * embed_dim);
    const int remainder = idx % (num_heads * embed_dim);
    const int head_idx = remainder / embed_dim;
    const int rot_idx = remainder % embed_dim;

    const int q_offset = token_idx * num_heads * head_size + head_idx * head_size;
    const int x_idx = q_offset + rot_idx;
    const int y_idx = q_offset + rot_idx + embed_dim;

    const int cs_idx = token_idx * embed_dim + rot_idx;
    float cos_val = static_cast<float>(cos_cache[cs_idx]);
    float sin_val = static_cast<float>(sin_cache[cs_idx]);

    float x_f = static_cast<float>(query[x_idx]);
    float y_f = static_cast<float>(query[y_idx]);
    query[x_idx] = static_cast<scalar_t>(x_f * cos_val - y_f * sin_val);
    query[y_idx] = static_cast<scalar_t>(y_f * sin_val + x_f * cos_val);
  }

  if (key != nullptr) {
    const int k_total = num_tokens * num_kv_heads * embed_dim;
    const int k_idx = idx - q_total;
    if (k_idx >= 0 && k_idx < k_total) {
      const int token_idx = k_idx / (num_kv_heads * embed_dim);
      const int remainder = k_idx % (num_kv_heads * embed_dim);
      const int head_idx = remainder / embed_dim;
      const int rot_idx = remainder % embed_dim;

      const int k_offset = token_idx * num_kv_heads * head_size + head_idx * head_size;
      const int x_idx = k_offset + rot_idx;
      const int y_idx = k_offset + rot_idx + embed_dim;

      const int cs_idx = token_idx * embed_dim + rot_idx;
      float cos_val = static_cast<float>(cos_cache[cs_idx]);
      float sin_val = static_cast<float>(sin_cache[cs_idx]);

      float x_f = static_cast<float>(key[x_idx]);
      float y_f = static_cast<float>(key[y_idx]);
      key[x_idx] = static_cast<scalar_t>(x_f * cos_val - y_f * sin_val);
      key[y_idx] = static_cast<scalar_t>(y_f * sin_val + x_f * cos_val);
    }
  }
}

void rotary_embedding(
    at::Tensor cos_cache,
    at::Tensor sin_cache,
    at::Tensor query,
    const std::optional<at::Tensor>& key,
    int64_t head_size,
    bool is_neox) {
  TORCH_CHECK(cos_cache.is_cuda(), "cos_cache must be a CUDA tensor");
  TORCH_CHECK(sin_cache.is_cuda(), "sin_cache must be a CUDA tensor");
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");

  TORCH_CHECK(cos_cache.dim() == 2, "cos_cache must be 2D (num_tokens, rotary_dim/2)");
  TORCH_CHECK(sin_cache.dim() == 2, "sin_cache must be 2D (num_tokens, rotary_dim/2)");
  TORCH_CHECK(query.dim() == 2, "query must be 2D (num_tokens, num_heads * head_size)");

  const int num_tokens = query.size(0);
  const int query_hidden = query.size(1);
  const int num_heads = query_hidden / head_size;
  const int rotary_dim = cos_cache.size(1) * 2;  // cos has rotary_dim/2

  TORCH_CHECK(query_hidden % head_size == 0, "query hidden size must be divisible by head_size");
  TORCH_CHECK(cos_cache.size(0) == num_tokens, "cos_cache num_tokens mismatch");
  TORCH_CHECK(sin_cache.size(0) == num_tokens, "sin_cache num_tokens mismatch");
  TORCH_CHECK(sin_cache.size(1) == cos_cache.size(1), "cos/sin cache size mismatch");

  at::Tensor key_tensor;
  int num_kv_heads = 0;
  bool has_key = key.has_value() && key->defined();
  if (has_key) {
    key_tensor = key.value();
    TORCH_CHECK(key_tensor.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(key_tensor.dim() == 2, "key must be 2D");
    TORCH_CHECK(key_tensor.size(0) == num_tokens, "key num_tokens mismatch");
    num_kv_heads = key_tensor.size(1) / head_size;
  }

  const int embed_dim = rotary_dim / 2;

  const int q_work = num_tokens * num_heads * embed_dim;
  const int k_work = has_key ? num_tokens * num_kv_heads * embed_dim : 0;
  const int total_work = q_work + k_work;

  const int block_size = 256;
  const int num_blocks = (total_work + block_size - 1) / block_size;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(), "rotary_embedding", [&] {
        scalar_t* key_ptr = has_key ? key_tensor.data_ptr<scalar_t>() : nullptr;
        if (is_neox) {
          rotary_embedding_neox_kernel<scalar_t><<<num_blocks, block_size, 0, stream>>>(
              query.data_ptr<scalar_t>(),
              key_ptr,
              cos_cache.data_ptr<scalar_t>(),
              sin_cache.data_ptr<scalar_t>(),
              num_tokens,
              num_heads,
              num_kv_heads,
              head_size,
              rotary_dim);
        } else {
          rotary_embedding_gptj_kernel<scalar_t><<<num_blocks, block_size, 0, stream>>>(
              query.data_ptr<scalar_t>(),
              key_ptr,
              cos_cache.data_ptr<scalar_t>(),
              sin_cache.data_ptr<scalar_t>(),
              num_tokens,
              num_heads,
              num_kv_heads,
              head_size,
              rotary_dim);
        }
      });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "rotary_embedding",
      &rotary_embedding,
      "Apply RoPE with separate cos/sin tensors",
      py::arg("cos_cache"),
      py::arg("sin_cache"),
      py::arg("query"),
      py::arg("key"),
      py::arg("head_size"),
      py::arg("is_neox") = true);
}
