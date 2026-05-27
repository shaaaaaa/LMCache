// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include "mem_kernels.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#ifdef USE_ROCM
  #include <hip/hip_fp8.h>
#else
  #include <cuda_fp8.h>
#endif

#ifndef CHECK_CUDA_CALL
  #define CHECK_CUDA_CALL(call)                                             \
    do {                                                                    \
      cudaError_t err = call;                                               \
      if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        throw std::runtime_error(                                           \
            std::string("CUDA error in file '") + __FILE__ + "' in line " + \
            std::to_string(__LINE__) + " : " + cudaGetErrorString(err));    \
      }                                                                     \
    } while (0)
#endif

namespace lmc {

// inline helper to check MLA (callable from device and host)
__host__ __device__ __forceinline__ bool is_mla(
    const GPUKVFormat gpu_kv_format) {
  return gpu_kv_format == GPUKVFormat::NL_X_NB_BS_HS ||   // vllm MLA
         gpu_kv_format == GPUKVFormat::NL_X_NBBS_ONE_HS;  // SGLang MLA
}

template <typename scalar_t>
__global__ void load_and_reshape_flash_kernel(
    scalar_t* __restrict__ key_value,  // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ key_cache,    // [num_blocks, block_size,
                                               // num_heads, head_size]
    const scalar_t* __restrict__ value_cache,  // [num_blocks, block_size,
                                               // num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride_in_64bit, const int key_value_stride,
    const int num_heads, const int head_size_in_64bit, const int block_size,
    const int key_layer_offset, const int value_layer_offset) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t tgt_key_idx =
        key_layer_offset + token_idx * key_value_stride + i;
    const int64_t tgt_value_idx =
        value_layer_offset + token_idx * key_value_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t src_key_value_idx =
        block_idx * block_stride_in_64bit +
        block_offset * num_heads * head_size_in_64bit +
        head_idx * head_size_in_64bit + head_offset;

    scalar_t tgt_key = key_cache[src_key_value_idx];
    scalar_t tgt_value = value_cache[src_key_value_idx];

    key_value[tgt_key_idx] = tgt_key;
    key_value[tgt_value_idx] = tgt_value;
  }
}

template <typename scalar_t>
__global__ void reshape_and_cache_back_flash_kernel(
    const scalar_t* __restrict__ key_value,  // [num_tokens, num_heads,
                                             // head_size]
    scalar_t* __restrict__ key_cache,    // [num_blocks, block_size, num_heads,
                                         // head_size]
    scalar_t* __restrict__ value_cache,  // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride_in_64bit, const int key_value_stride,
    const int num_heads, const int head_size_in_64bit, const int block_size,
    const int key_layer_offset, const int value_layer_offset) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t tgt_key_idx =
        key_layer_offset + token_idx * key_value_stride + i;
    const int64_t tgt_value_idx =
        value_layer_offset + token_idx * key_value_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t src_key_value_idx =
        block_idx * block_stride_in_64bit +
        block_offset * num_heads * head_size_in_64bit +
        head_idx * head_size_in_64bit + head_offset;

    scalar_t tgt_key = key_value[tgt_key_idx];
    scalar_t tgt_value = key_value[tgt_value_idx];

    key_cache[src_key_value_idx] = tgt_key;
    value_cache[src_key_value_idx] = tgt_value;
  }
}

template <typename scalar_t, bool USE_MLA>
__global__ void single_layer_kv_transfer_kernel(
    // scalar_t* __restrict__ lmc_key_cache,    // [num_tokens,
    // num_heads*head_size] scalar_t* __restrict__ lmc_value_cache,  //
    // [num_tokens, num_heads*head_size]
    scalar_t* __restrict__ lmc_key_value_cache,   // [num_tokens, 2,
                                                  // num_heads*head_size]
                                                  // or
                                                  // [2, num_tokens,
                                                  // num_heads*head_size]
                                                  // or for MLA:
                                                  // [num_tokens,
                                                  // aligned_head_size]
    scalar_t* __restrict__ vllm_key_value_cache,  // [2, num_blocks, block_size,
                                                  // num_heads, head_size] or
                                                  // [num_blocks, 2, block_size,
                                                  // num_heads, head_size]
                                                  // or for MLA:
                                                  // [num_blocks, block_size,
                                                  // head_size]

    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int vllm_block_key_stride_in_64bit, const int vllm_value_offset,
    const int lmc_stride, const int lmc_value_offset, const int num_heads,
    const int head_size_in_64bit, const int block_size,
    const TransferDirection direction) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t lmc_key_idx = token_idx * lmc_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t vllm_key_idx = block_idx * vllm_block_key_stride_in_64bit +
                                 block_offset * num_heads * head_size_in_64bit +
                                 head_idx * head_size_in_64bit + head_offset;

    if (direction == TransferDirection::D2H) {
      // GPU to LMCache
      lmc_key_value_cache[lmc_key_idx] = vllm_key_value_cache[vllm_key_idx];
      // For non-MLA, also copy the value component
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        lmc_key_value_cache[lmc_value_idx] =
            vllm_key_value_cache[vllm_value_idx];
      }
    } else {
      // LMCache to GPU
      vllm_key_value_cache[vllm_key_idx] = lmc_key_value_cache[lmc_key_idx];
      // For non-MLA, also copy the value component
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        vllm_key_value_cache[vllm_value_idx] =
            lmc_key_value_cache[lmc_value_idx];
      }
    }
  }
}

template <typename scalar_t, bool USE_MLA>
__global__ void single_layer_head_token_wise_kv_transfer_kernel(
    // scalar_t* __restrict__ lmc_key_cache,    // [num_tokens,
    // num_heads*head_size] scalar_t* __restrict__ lmc_value_cache,  //
    // [num_tokens, num_heads*head_size]
    scalar_t* __restrict__ lmc_key_value_cache,   // [num_tokens, 2,
                                                  // num_heads*head_size]
                                                  // or
                                                  // [2, num_tokens,
                                                  // num_heads*head_size]
                                                  // or for MLA:
                                                  // [num_tokens,
                                                  // aligned_head_size]
    scalar_t* __restrict__ vllm_key_value_cache,  // [2, num_blocks, block_size,
                                                  // num_heads, head_size] or
                                                  // [num_blocks, 2, block_size,
                                                  // num_heads, head_size]
                                                  // or for MLA:
                                                  // [num_blocks, block_size,
                                                  // head_size]

    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t* __restrict__ selected_tokens,
    const int vllm_block_key_stride_in_64bit, const int vllm_value_offset,
    const int lmc_stride, const int lmc_value_offset, const int num_heads,
    const int head_size_in_64bit, const int block_size,
    const TransferDirection direction,
    const int fixed_head) {

    int64_t token_idx;
    int64_t head_idx;

  if (fixed_head >= 0) {
    // grid size num_tokens
    token_idx = blockIdx.x;
    head_idx = fixed_head;
  } else {
    // grid size num_tokens * num_heads
    int64_t token_head_idx = blockIdx.x;
    token_idx = token_head_idx / num_heads;
    head_idx = token_head_idx % num_heads;
  }

  const int64_t slot_idx = slot_mapping[token_idx];
  // 从 selected_tokens 获取 LMCache 中的真实 token 索引
  const int64_t lmc_token_idx = selected_tokens[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  // const int n = num_heads * head_size_in_64bit;

  const int64_t k_offset_in_token = head_idx * head_size_in_64bit;

  for (int i = threadIdx.x; i < head_size_in_64bit; i += blockDim.x) {
    const int64_t lmc_key_idx = lmc_token_idx * lmc_stride + k_offset_in_token + i;

    const int64_t vllm_key_idx = block_idx * vllm_block_key_stride_in_64bit +
                                 block_offset * num_heads * head_size_in_64bit +
                                 head_idx * head_size_in_64bit + i;

    if (direction == TransferDirection::D2H) {
      // GPU to LMCache
      lmc_key_value_cache[lmc_key_idx] = vllm_key_value_cache[vllm_key_idx];
      // For non-MLA, also copy the value component
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        lmc_key_value_cache[lmc_value_idx] =
            vllm_key_value_cache[vllm_value_idx];
      }
    } else {
      // LMCache to GPU
      vllm_key_value_cache[vllm_key_idx] = lmc_key_value_cache[lmc_key_idx];
      // For non-MLA, also copy the value component
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        vllm_key_value_cache[vllm_value_idx] =
            lmc_key_value_cache[lmc_value_idx];
      }
    }
  }
}


template <typename T>
__global__ void single_layer_sparse_kv_transfer_kernel(
    int64_t* __restrict__ lmcache_kv_device_ptrs,   // [num_chunks]
    T* __restrict__ vllm_k,                         // vLLM K cache: [num_blocks, block_size, num_heads, head_dim]
    T* __restrict__ vllm_v,                         // vLLM V cache
    const int64_t* __restrict__ slot_mapping,       // [num_all_tokens]
    const int64_t* __restrict__ selected_tokens,    // [num_heads, num_selected_tokens]
    const int64_t token_start_index,
    const int num_chunks,
    const int num_selected_tokens,
    const int num_heads,
    const int num_tokens_per_chunk,
    const int block_size,
    const int head_dim,
    const int stride_vllm_block,                    // = block_size * num_heads * head_dim
    const int stride_vllm_slot,                     // = num_heads * head_dim
    const int stride_vllm_head,                     // = head_dim
    const int stride_lm_token,                      // = 2 * num_heads * head_dim
    const int stride_lm_kv,                         // = num_heads * head_dim
    const int stride_lm_dim,                        // = head_dim
    const int stride_sel_head                       // = num_selected_tokens
) {
    const int head_idx = blockIdx.x;
    const int token_j   = blockIdx.y;

    if (head_idx >= num_heads || token_j >= num_selected_tokens)
        return;

    // 1.token index for this thread block
    const int64_t global_token_idx = selected_tokens[head_idx * stride_sel_head + token_j];

    // 2. lmcache chunk indexing
    const int chunk_idx = global_token_idx / num_tokens_per_chunk;
    const int token_offset = global_token_idx % num_tokens_per_chunk;

    // 3. src cpu base ptr（CPU pinned memory）
    if (threadIdx.x == 0 && chunk_idx >= num_chunks) {
      printf("[single_layer_sparse_kv_transfer_kernel] ERROR: head_idx: %d token_idx: %d, chunk_idx = %d >= num_chunks = %d, global_token_idx = %d\n", head_idx, token_j, chunk_idx, num_chunks, global_token_idx);
    }
    T* chunk_ptr = reinterpret_cast<T*>(lmcache_kv_device_ptrs[chunk_idx]);

    // 4. vllm slot indexing
    const int64_t slot = slot_mapping[token_start_index + token_j];
    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;

    // 5. src
    T* src_k = chunk_ptr + token_offset * stride_lm_token
            + 0 * stride_lm_kv + head_idx * stride_lm_dim;
    T* src_v = chunk_ptr + token_offset * stride_lm_token
            + 1 * stride_lm_kv + head_idx * stride_lm_dim;

    // 6. dst
    T* dst_k = vllm_k + block_idx * stride_vllm_block
             + block_offset * stride_vllm_slot + head_idx * stride_vllm_head;
    T* dst_v = vllm_v + block_idx * stride_vllm_block
             + block_offset * stride_vllm_slot + head_idx * stride_vllm_head;

    // 7. copy
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dst_k[d] = src_k[d];
        dst_v[d] = src_v[d];
    }
}


template <typename T>
__global__ void single_layer_sparse_clustered_flattened_kv_transfer_kernel(
    int64_t* __restrict__ lmcache_kv_device_ptrs,   // [num_chunks]
    T* __restrict__ vllm_k,                         // vLLM K cache: [num_blocks, block_size, num_heads, head_dim]
    T* __restrict__ vllm_v,                         // vLLM V cache
    const int64_t* __restrict__ slot_mapping,       // [num_all_tokens]
    const int32_t* __restrict__ clusters,           // [num_heads, num_clusters, max_cluster_size]
    const int64_t* __restrict__ selected_clusters,  // [num_heads, num_selected_clusters]
    const int32_t* __restrict__ cluster_size,       // [num_heads, num_clusters]
    const int32_t* __restrict__ cluster_start_index,// [num_heads, num_selected_clusters]
    const int32_t retrieve_budget,
    const int num_chunks,
    const int token_start_index,
    const int num_selected_clusters,
    const int num_heads,
    const int num_tokens_per_chunk,
    const int block_size,
    const int head_dim,
    const int stride_vllm_block,                    // = block_size * num_heads * head_dim
    const int stride_vllm_slot,                     // = num_heads * head_dim
    const int stride_vllm_head,                     // = head_dim
    const int stride_lm_token,                      // = 2 * num_heads * head_dim
    const int stride_lm_kv,                         // = num_heads * head_dim
    const int stride_lm_dim,                        // = head_dim
    const int stride_cluster_head,                  // = num_clusters * max_cluster_size
    const int stride_cluster_num,                   // = max_cluster_size
    const int stride_cs_head,                       // = num_clusters
    const int stride_sel_head                       // = num_selected_clusters
) {
    extern __shared__ int32_t s_cumsum[];

    const int head_idx = blockIdx.x;
    const int token_idx = blockIdx.y;

    for (int i = threadIdx.x; i < num_selected_clusters; i += blockDim.x) {
        s_cumsum[i] = cluster_start_index[head_idx * stride_sel_head + i];
    }
    __syncthreads();

    int total_tokens = s_cumsum[num_selected_clusters - 1];
    if (token_idx >= total_tokens) {
      // printf("[single_layer_sparse_clustered_kv_transfer_kernel] WARNING: head_idx: %d token_idx:%d, total_tokens=%d is less than token_idx\n", head_idx, token_idx, total_tokens);
      return;
    }

    int l = 0, r = num_selected_clusters - 1;
#pragma unroll
    while (l < r) {
        int mid = (l + r) >> 1;
        if (s_cumsum[mid] <= token_idx)
            l = mid + 1;
        else
            r = mid;
    }
    int cluster_idx = l;

    int64_t real_cluster_id = selected_clusters[head_idx * stride_sel_head + cluster_idx];
    int32_t cur_cluster_size = cluster_size[head_idx * stride_cs_head + real_cluster_id];

    int32_t cluster_physical_start_index = s_cumsum[cluster_idx] - cur_cluster_size;
    if (cluster_physical_start_index >= retrieve_budget)
      return;

    // 1.token index for this thread block
    int32_t token_i = token_idx - cluster_physical_start_index;
    const int32_t global_token_idx = clusters[head_idx * stride_cluster_head + real_cluster_id * stride_cluster_num + token_i];

    // 2. lmcache chunk indexing
    const int32_t chunk_idx = global_token_idx / num_tokens_per_chunk;
    const int32_t token_offset = global_token_idx % num_tokens_per_chunk;

    // 3. src cpu base ptr（CPU pinned memory）
    if (chunk_idx >= num_chunks) {
      printf("[single_layer_sparse_clustered_kv_transfer_kernel] ERROR: head_idx: %d cluster_idx: %d, chunk_idx=%d >= num_chunks=%d, global_token_idx = %d\n", head_idx, cluster_idx, chunk_idx, num_chunks, global_token_idx);
      return;
    }
    T* chunk_ptr = reinterpret_cast<T*>(lmcache_kv_device_ptrs[chunk_idx]);

    // 4. vllm slot indexing
    const int64_t slot = slot_mapping[token_start_index + token_idx];
    const int64_t block_idx = slot / block_size;
    const int64_t block_offset = slot % block_size;

    // 5. src
    T* src_k = chunk_ptr + token_offset * stride_lm_token
            + 0 * stride_lm_kv + head_idx * stride_lm_dim;
    T* src_v = chunk_ptr + token_offset * stride_lm_token
            + 1 * stride_lm_kv + head_idx * stride_lm_dim;

    // 6. dst
    T* dst_k = vllm_k + block_idx * stride_vllm_block
            + block_offset * stride_vllm_slot + head_idx * stride_vllm_head;
    T* dst_v = vllm_v + block_idx * stride_vllm_block
            + block_offset * stride_vllm_slot + head_idx * stride_vllm_head;

    // 7. copy
#pragma unroll
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dst_k[d] = src_k[d];
        dst_v[d] = src_v[d];
    }
}


template <typename T>
__global__ void single_layer_sparse_clustered_kv_transfer_kernel(
    int64_t* __restrict__ lmcache_kv_device_ptrs,   // [num_chunks]
    T* __restrict__ vllm_k,                         // vLLM K cache: [num_blocks, block_size, num_heads, head_dim]
    T* __restrict__ vllm_v,                         // vLLM V cache
    const int64_t* __restrict__ slot_mapping,       // [num_all_tokens]
    const int32_t* __restrict__ clusters,           // [num_heads, num_clusters, max_cluster_size]
    const int64_t* __restrict__ selected_clusters,  // [num_heads, num_selected_clusters]
    const int32_t* __restrict__ cluster_size,       // [num_heads, num_clusters]
    const int32_t* __restrict__ cluster_start_index,// [num_heads, num_selected_clusters]
    const int32_t retrieve_budget,
    const int num_chunks,
    const int token_start_index,
    const int num_selected_clusters,
    const int num_heads,
    const int num_tokens_per_chunk,
    const int block_size,
    const int head_dim,
    const int stride_vllm_block,                    // = block_size * num_heads * head_dim
    const int stride_vllm_slot,                     // = num_heads * head_dim
    const int stride_vllm_head,                     // = head_dim
    const int stride_lm_token,                      // = 2 * num_heads * head_dim
    const int stride_lm_kv,                         // = num_heads * head_dim
    const int stride_lm_dim,                        // = head_dim
    const int stride_cluster_head,                  // = num_clusters * max_cluster_size
    const int stride_cluster_num,                   // = max_cluster_size
    const int stride_cs_head,                       // = num_clusters
    const int stride_sel_head                       // = num_selected_clusters
) {
    const int head_idx = blockIdx.x;
    const int cluster_idx = blockIdx.y;

    int64_t real_cluster_id = selected_clusters[head_idx * stride_sel_head + cluster_idx];
    int32_t cur_cluster_size = cluster_size[head_idx * stride_cs_head + real_cluster_id];

    int32_t cluster_physical_start_index = cluster_start_index[head_idx * stride_sel_head + cluster_idx] - cur_cluster_size;
    if (cluster_physical_start_index >= retrieve_budget)
      return;

    int32_t num_tokens = std::min(cur_cluster_size, retrieve_budget - cluster_physical_start_index);
    for (int32_t token_i = 0; token_i < num_tokens; token_i++) {
      // 1.token index for this thread block
      const int32_t global_token_idx = clusters[head_idx * stride_cluster_head + real_cluster_id * stride_cluster_num + token_i];

      // 2. lmcache chunk indexing
      const int32_t chunk_idx = global_token_idx / num_tokens_per_chunk;
      const int32_t token_offset = global_token_idx % num_tokens_per_chunk;

      // 3. src cpu base ptr（CPU pinned memory）
      if (chunk_idx >= num_chunks) {
        printf("[single_layer_sparse_clustered_kv_transfer_kernel] ERROR: head_idx: %d cluster_idx: %d, chunk_idx=%d >= num_chunks=%d, global_token_idx = %d\n", head_idx, cluster_idx, chunk_idx, num_chunks, global_token_idx);
        continue;
      }
      T* chunk_ptr = reinterpret_cast<T*>(lmcache_kv_device_ptrs[chunk_idx]);

      // 4. vllm slot indexing
      const int64_t slot = slot_mapping[token_start_index + cluster_physical_start_index + token_i];
      const int64_t block_idx = slot / block_size;
      const int64_t block_offset = slot % block_size;

      // 5. src
      T* src_k = chunk_ptr + token_offset * stride_lm_token
              + 0 * stride_lm_kv + head_idx * stride_lm_dim;
      T* src_v = chunk_ptr + token_offset * stride_lm_token
              + 1 * stride_lm_kv + head_idx * stride_lm_dim;

      // 6. dst
      T* dst_k = vllm_k + block_idx * stride_vllm_block
              + block_offset * stride_vllm_slot + head_idx * stride_vllm_head;
      T* dst_v = vllm_v + block_idx * stride_vllm_block
              + block_offset * stride_vllm_slot + head_idx * stride_vllm_head;

      // 7. copy
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
          dst_k[d] = src_k[d];
          dst_v[d] = src_v[d];
      }
    }
}


template <GPUKVFormat format>
__device__ __forceinline__ int64_t
page_buffer_offset(const int k_or_v, const int token_idx,
                   const int scalar_offset, const int scalars_per_token,
                   const int page_buffer_size, const int block_size) {
  // vllm cross layer
  if constexpr (format == GPUKVFormat::NB_NL_TWO_BS_NH_HS) {
    return k_or_v * page_buffer_size * scalars_per_token +
           token_idx * scalars_per_token + scalar_offset;
  }
  // vllm flash attention
  else if constexpr (format == GPUKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    return k_or_v * page_buffer_size * scalars_per_token +
           token_idx * scalars_per_token + scalar_offset;
  }
  // vllm flash infer
  else if constexpr (format == GPUKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    const int block_idx = token_idx / block_size;
    const int block_offset = token_idx % block_size;
    return block_idx * 2 * block_size * scalars_per_token +
           k_or_v * block_size * scalars_per_token +
           block_offset * scalars_per_token + scalar_offset;
  }
  // MLA formats: vLLM (NL_X_NB_BS_HS) and SGLang (NL_X_NBBS_ONE_HS)
  else if constexpr (format == GPUKVFormat::NL_X_NB_BS_HS ||
                     format == GPUKVFormat::NL_X_NBBS_ONE_HS) {
    return token_idx * scalars_per_token + scalar_offset;
  }
}

__device__ __forceinline__ int64_t page_buffer_offset_unilateral(
    const int token_idx, const int scalar_offset, const int scalars_per_token) {
  return token_idx * scalars_per_token + scalar_offset;
}

__device__ __forceinline__ int64_t
key_value_offset(const int k_or_v, const int layer_idx, const int token_idx,
                 const int scalar_offset, const int scalars_per_token,
                 const int num_tokens, const int num_layers) {
  return k_or_v * num_layers * num_tokens * scalars_per_token +
         layer_idx * num_tokens * scalars_per_token +
         token_idx * scalars_per_token + scalar_offset;
}

template <typename scalar_t>
__global__ void single_layer_kv_transfer_sgl_kernel(
    // scalar_t* __restrict__ lmc_key_cache,    // [num_tokens,
    // num_heads*head_size] scalar_t* __restrict__ lmc_value_cache,  //
    // [num_tokens, num_heads*head_size]
    scalar_t* __restrict__ lmc_key_value_cache,  // [num_tokens, 2,
                                                 // num_heads*head_size]
                                                 // or
                                                 // [2, num_tokens,
                                                 // num_heads*head_size]
    scalar_t* __restrict__ sgl_key_cache,        // [num_blocks, block_size,
                                                 // num_heads, head_size]
    scalar_t* __restrict__ sgl_value_cache,      // [num_blocks, block_size,
                                                 // num_heads, head_size]
    const int64_t* __restrict__ slot_mapping,    // [num_tokens]
    const int block_stride_in_64bit, const int lmc_stride,
    const int lmc_value_offset, const int num_heads,
    const int head_size_in_64bit, const int block_size,
    const TransferDirection direction) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t lmc_key_idx = token_idx * lmc_stride + i;
    const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;
    const int64_t sgl_key_value_idx =
        block_idx * block_stride_in_64bit +
        block_offset * num_heads * head_size_in_64bit +
        head_idx * head_size_in_64bit + head_offset;

    if (direction == TransferDirection::D2H) {
      lmc_key_value_cache[lmc_key_idx] = sgl_key_cache[sgl_key_value_idx];
      lmc_key_value_cache[lmc_value_idx] = sgl_value_cache[sgl_key_value_idx];
    } else {  // direction == TransferDirection::H2D
      sgl_key_cache[sgl_key_value_idx] = lmc_key_value_cache[lmc_key_idx];
      sgl_value_cache[sgl_key_value_idx] = lmc_key_value_cache[lmc_value_idx];
    }
  }
}

/**
 * Quickly load KV cache between vLLM paged memory and offloading buffer
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] <=> ptrs[block.y][block.z,
 * slot_id, thread.x]
 */
template <typename scalar_t, bool DIRECTION, GPUKVFormat format>
__global__ void load_and_reshape_multi_layer_kernel(
    scalar_t* __restrict__ key_value,           // [2, num_layer, num_tokens,
                                                // scalars_per_token]
    scalar_t** __restrict__ paged_buffer_ptrs,  // [num_layers] * [2,
                                                // PAGE_BUFFER_SIZE,
                                                // scalars_per_token]
                                                // or
                                                // [num_layers] * [num_blocks,
                                                // 2, block_size,
                                                // scalars_per_token]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int scalars_per_token, const int num_tokens, const int num_layers,
    const int page_buffer_size, const int block_size,
    const int skip_prefix_n_tokens) {
  const int token_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int k_or_v = blockIdx.z;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int kv_token_id = token_id + skip_prefix_n_tokens;
  const int64_t slot_idx = slot_mapping[kv_token_id];
  scalar_t* paged_buffer_ptr = paged_buffer_ptrs[layer_id];

  if (slot_idx < 0) {
    return;
  }

  /** Copy the data from page buffer to key_value **/
  for (int i = tid; i < scalars_per_token; i += num_threads) {
    const int64_t lmcache_offset =
        key_value_offset(k_or_v, layer_id, kv_token_id, i, scalars_per_token,
                         num_tokens, num_layers);

    const int64_t vllm_offset = page_buffer_offset<format>(
        k_or_v, slot_idx, i, scalars_per_token, page_buffer_size, block_size);

    if (DIRECTION)  // 1 is paged buffer to LMCache
      key_value[lmcache_offset] = paged_buffer_ptr[vllm_offset];
    else  // 0 is LMCache to paged buffer
      paged_buffer_ptr[vllm_offset] = key_value[lmcache_offset];
  }
}

/*
 * handle sglang MHA offload between CPU and GPU
 * DIRECTION = 1 (true) means paged buffer to LMCache (D2H)
 * DIRECTION = 0 (false) means LMCache to paged buffer (H2D)
 */
template <typename scalar_t, bool DIRECTION>
__global__ void load_and_reshape_multi_layer_kernel_unilateral(
    scalar_t* __restrict__ key_value,           // [2, num_layer, num_tokens,
                                                // scalars_per_token]
    scalar_t** __restrict__ paged_buffer_ptrs,  // [num_layers *2] *
                                                // [PAGE_BUFFER_SIZE,
                                                // scalars_per_token]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int scalars_per_token, const int num_tokens, const int num_layers,
    const int page_buffer_size) {
  const int token_id = blockIdx.x;
  const int layer_id = blockIdx.y;
  const int k_or_v = blockIdx.z;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int64_t slot_idx = slot_mapping[token_id];
  scalar_t* key_ptr = paged_buffer_ptrs[layer_id];
  scalar_t* value_ptr = paged_buffer_ptrs[layer_id + num_layers];

  if (slot_idx < 0) {
    return;
  }

  /** Copy the data from page buffer to key_value **/
  for (int i = tid; i < scalars_per_token; i += num_threads) {
    const int64_t lmcache_offset =
        key_value_offset(k_or_v, layer_id, token_id, i, scalars_per_token,
                         num_tokens, num_layers);

    const int64_t sgl_offset =
        page_buffer_offset_unilateral(slot_idx, i, scalars_per_token);

    if (k_or_v == 0) {
      if (DIRECTION)  // 1 is paged buffer to LMCache
        key_value[lmcache_offset] = key_ptr[sgl_offset];
      else  // 0 is LMCache to paged buffer
        key_ptr[sgl_offset] = key_value[lmcache_offset];
    } else {
      if (DIRECTION)  // 1 is paged buffer to LMCache
        key_value[lmcache_offset] = value_ptr[sgl_offset];
      else  // 0 is LMCache to paged buffer
        value_ptr[sgl_offset] = key_value[lmcache_offset];
    }
  }
}

}  // namespace lmc

template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
  // Get the kernel-accessible pointer of the given type T
  // Returns NULL if the tensor is on CPU and non-pinned
  torch::Device device = tensor.device();
  if (device.is_cuda()) {
    return static_cast<T*>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    T* ptr;
    auto st = cudaHostGetDevicePointer(
        (void**)&ptr, static_cast<void*>(tensor.data_ptr()), 0);
    TORCH_CHECK(st == cudaSuccess,
                "Host tensor not registered/pinned (or bad ptr)");
    return ptr;
  } else {
    TORCH_CHECK(false, "Invalid device. Device must be cuda or pinned cpu.");
  }
}

/**
 * Quickly offload KV cache from vLLM paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in vLLM's KV buffer has a shape of
 * [2, PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each thread block processes the copy for a token
 * The grid size should be (num_tokens, num_layers, 2)
 *
 * Therefore:
 *  - k/v -- block.z
 *  - layer id -- block.y
 *  - token id -- block.x
 *  - offset within a token -- thread.x
 *
 * The function does:
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] = ptrs[block.y][block.z,
 * slot_id, thread.x]
 *
 * Param:
 *  - direction: H2D  means LMCache to PagedBuffer, D2H  means PagedBuffer to
 * LMCache
 */
#define LAUNCH_KERNEL_WITH_FORMAT(T, DIRECTION, FORMAT)                      \
  lmc::load_and_reshape_multi_layer_kernel<T, DIRECTION, FORMAT>             \
      <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,          \
                                   slot_mapping_ptr, num_xwords, num_tokens, \
                                   num_layers, page_buffer_size, block_size, \
                                   skip_prefix_n_tokens);                    \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

template <typename T>
void multi_layer_kv_transfer_templated(
    torch::Tensor&
        key_value,  // key/value must be on gpu/pinned cpu.
                    // [2, num_layer, num_tokens, num_heads*head_size] for
                    // flash_attn.
                    // [1, num_layer, num_tokens, aligned_head_size]
                    // for MLA.
    const torch::Tensor& key_value_ptrs,  // [num_layers]
    const torch::Tensor& slot_mapping,    // [num_tokens],
    const torch::Device& paged_memory_device, const int page_buffer_size,
    const TransferDirection direction, const GPUKVFormat gpu_kv_format,
    const int block_size, const int skip_prefix_n_tokens) {
  T* key_value_ptr = get_kernel_ptr<T, torch::Tensor>(key_value);
  T** page_buffer_ptrs =
      get_kernel_ptr<T*, const torch::Tensor>(key_value_ptrs);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int num_layers = key_value.size(1);
  int num_tokens = key_value.size(2);
  int num_transfer_tokens = num_tokens - skip_prefix_n_tokens;
  int num_origin_elements = key_value.size(3);
  int elements_per_xword = sizeof(T) / key_value.element_size();
  int num_xwords = num_origin_elements / elements_per_xword;

  int k_or_v_size = lmc::is_mla(gpu_kv_format) ? 1 : 2;

  dim3 grid(num_transfer_tokens, num_layers, k_or_v_size);
  dim3 block(std::min(num_xwords, 128));

  const at::cuda::OptionalCUDAGuard device_guard(paged_memory_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (direction == TransferDirection::H2D) {
    switch (gpu_kv_format) {
      case GPUKVFormat::NB_NL_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NB_NL_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NL_X_NB_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_BS_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NL_X_NB_BS_HS);
        break;
      case GPUKVFormat::NL_X_NBBS_ONE_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, false, GPUKVFormat::NL_X_NBBS_ONE_HS);
        break;
      default:
        throw std::runtime_error("Unsupported GPUKVFormat");
    }
  } else {
    switch (gpu_kv_format) {
      case GPUKVFormat::NB_NL_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, GPUKVFormat::NB_NL_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, GPUKVFormat::NL_X_NB_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_BS_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, GPUKVFormat::NL_X_NB_BS_HS);
        break;
      case GPUKVFormat::NL_X_NBBS_ONE_HS:
        LAUNCH_KERNEL_WITH_FORMAT(T, true, GPUKVFormat::NL_X_NBBS_ONE_HS);
        break;
      default:
        throw std::runtime_error("Unsupported GPUKVFormat");
    }
  }
}

#undef LAUNCH_KERNEL_WITH_FORMAT

/**
 * @see multi_layer_kv_transfer_templated
 */
void multi_layer_kv_transfer(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const GPUKVFormat gpu_kv_format, const int block_size,
    const int skip_prefix_n_tokens) {
  int num_origin_elements = key_value.size(3);
  int copy_size = num_origin_elements * key_value.element_size();
#ifndef LAUNCH_MULTI_LAYER_KV_TRANSFER
  #define LAUNCH_MULTI_LAYER_KV_TRANSFER(type)                          \
    do {                                                                \
      multi_layer_kv_transfer_templated<type>(                          \
          key_value, key_value_ptrs, slot_mapping, paged_memory_device, \
          page_buffer_size, direction, gpu_kv_format, block_size,       \
          skip_prefix_n_tokens);                                        \
    } while (0)
#endif
  if (copy_size % 8 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int64_t);
  } else if (copy_size % 4 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int32_t);
  } else if (copy_size % 2 == 0) {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int16_t);
  } else {
    LAUNCH_MULTI_LAYER_KV_TRANSFER(int8_t);
  }
#undef LAUNCH_MULTI_LAYER_KV_TRANSFER
}

/**
 * Quickly offload KV cache from SGLang paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in SGLang's K/V buffer has a shape of
 * [PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each thread block processes the copy for a token
 * The grid size should be (num_tokens, num_layers, 2)
 *
 * Therefore:
 *  - k/v -- block.z
 *  - layer id -- block.y
 *  - token id -- block.x
 *  - offset within a token -- thread.x
 *
 * The function does:
 * slot_id = slot_mapping[block.x]
 * key_value[block.z, block.y, block.x, thread.x] = ptrs[block.y][block.z,
 * slot_id, thread.x]
 *
 * Param:
 *  - direction: H2D  means LMCache to PagedBuffer, D2H  means PagedBuffer to
 * LMCache
 */
void multi_layer_kv_transfer_unilateral(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size] for
                    // flash_attn [1, num_layer, num_tokens, aligned_head_size]
                    // for MLA key/value must be on gpu/pinned cpu

    const torch::Tensor& key_value_ptrs,  // [num_layers*2]
    const torch::Tensor& slot_mapping,    // [num_tokens],
    const torch::Device& paged_memory_device, const int page_buffer_size,
    const TransferDirection direction, const GPUKVFormat gpu_kv_format) {
  const bool use_mla = lmc::is_mla(gpu_kv_format);
  // MLA case collapses back to multi_layer_kv_transfer
  // (vLLM and SGLang indexing are compatible)
  if (use_mla) {
    return multi_layer_kv_transfer(key_value, key_value_ptrs, slot_mapping,
                                   paged_memory_device, page_buffer_size,
                                   direction, gpu_kv_format);
  }

  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);
  int64_t** page_buffer_ptrs =
      get_kernel_ptr<int64_t*, const torch::Tensor>(key_value_ptrs);
  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int num_layers = key_value.size(1);
  int num_tokens = slot_mapping.size(0);
  int num_origin_elements = key_value.size(3);
  int elements_per_qword = 8 / key_value.element_size();
  int num_qwords = num_origin_elements / elements_per_qword;

  int k_or_v_size = 2;

  dim3 grid(key_value.size(2), key_value.size(1), k_or_v_size);
  dim3 block(std::min(num_qwords, 128));

  const at::cuda::OptionalCUDAGuard device_guard(paged_memory_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (direction == TransferDirection::H2D) {
    lmc::load_and_reshape_multi_layer_kernel_unilateral<int64_t, false>
        <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,
                                     slot_mapping_ptr, num_qwords, num_tokens,
                                     num_layers, page_buffer_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    lmc::load_and_reshape_multi_layer_kernel_unilateral<int64_t, true>
        <<<grid, block, 0, stream>>>(key_value_ptr, page_buffer_ptrs,
                                     slot_mapping_ptr, num_qwords, num_tokens,
                                     num_layers, page_buffer_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void single_layer_kv_transfer(
    // torch::Tensor& lmc_key_cache,  // [num_tokens, num_heads*head_size]
    //  key/value must be on gpu/pinned cpu
    // torch::Tensor& lmc_value_cache,  // [num_tokens, num_heads*head_size]

    torch::Tensor& lmc_key_value_cache,  // [num_tokens, 2, num_heads*head_size]
                                         // or
                                         // [2, num_tokens, num_heads*head_size]
                                         // or for MLA:
                                         // [num_tokens, aligned_head_size]

    // torch::Tensor&
    //     vllm_key_cache,  // [num_blocks, block_size, num_heads, head_size]
    // torch::Tensor&
    //     vllm_value_cache,  // [num_blocks, block_size, num_heads, head_size]
    //  key_cache/value_cache must be on gpu
    torch::Tensor&
        vllm_key_value_cache,  // [2, num_blocks, block_size, num_heads,
                               // head_size] for flash attention
    // [num_blocks, 2, block_size, num_heads, head_size] for flash infer
    // [num_blocks, block_size, head_size] for MLA

    torch::Tensor& slot_mapping,  // [num_tokens]
    const TransferDirection direction, const GPUKVFormat gpu_kv_format,
    const bool token_major  // true: lmc_key_value_cache is
                            // [num_tokens, 2, num_heads*head_size]
                            // false: lmc_key_value_cache is
                            // [2, num_tokens, num_heads*head_size]
) {
  // int64_t* lmc_key_cache_ptr = get_kernel_ptr<int64_t,
  // torch::Tensor>(lmc_key_cache); int64_t* lmc_value_cache_ptr =
  // get_kernel_ptr<int64_t, torch::Tensor>(lmc_value_cache);
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);

  int64_t* vllm_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(vllm_key_value_cache);
  // int64_t* vllm_value_cache_ptr =
  //     get_kernel_ptr<int64_t, torch::Tensor>(vllm_value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / vllm_key_value_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads;
  int head_size_in_64bit;
  int block_size;

  const bool use_mla = lmc::is_mla(gpu_kv_format);

  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    num_heads = 1;
    block_size = vllm_key_value_cache.size(1);
    head_size_in_64bit = vllm_key_value_cache.size(2) / elements_per_entry;
  } else {
    num_heads = vllm_key_value_cache.size(3);
    head_size_in_64bit = vllm_key_value_cache.size(4) / elements_per_entry;
    block_size = vllm_key_value_cache.size(2);
  }

  int lmc_stride;
  int lmc_value_offset;
  if (use_mla) {
    // MLA format: [num_tokens, aligned_head_size]
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = 0;  // No separate K/V for MLA
  } else if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int vllm_block_key_stride_in_64bit;
  int vllm_value_offset;
  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = 0;  // No separate K/V for MLA
  } else if (gpu_kv_format == GPUKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(1) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(0) / elements_per_entry;
  } else {  // gpu_kv_format == GPUKVFormat::NL_X_NB_TWO_BS_NH_HS
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(1) / elements_per_entry;
  }

  // int block_stride_in_64bit = vllm_key_cache.stride(0) / elements_per_entry;
  // TORCH_CHECK(vllm_key_cache.stride(0) == vllm_value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(
      device_of(vllm_key_value_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Dispatch to the appropriate template specialization based on use_mla
  if (use_mla) {
    lmc::single_layer_kv_transfer_kernel<int64_t, true>
        <<<grid, block, 0, stream>>>(
            lmc_key_value_cache_ptr, vllm_key_value_cache_ptr, slot_mapping_ptr,
            vllm_block_key_stride_in_64bit, vllm_value_offset, lmc_stride,
            lmc_value_offset, num_heads, head_size_in_64bit, block_size,
            direction);
  } else {
    lmc::single_layer_kv_transfer_kernel<int64_t, false>
        <<<grid, block, 0, stream>>>(
            lmc_key_value_cache_ptr, vllm_key_value_cache_ptr, slot_mapping_ptr,
            vllm_block_key_stride_in_64bit, vllm_value_offset, lmc_stride,
            lmc_value_offset, num_heads, head_size_in_64bit, block_size,
            direction);
  }
}

void single_layer_head_token_wise_kv_transfer(
    torch::Tensor& lmc_key_value_cache,  // [num_tokens, 2, num_heads*head_size]
                                         // or
                                         // [2, num_tokens, num_heads*head_size]
                                         // or for MLA:
                                         // [num_tokens, aligned_head_size]
    torch::Tensor&
        vllm_key_value_cache,  // [2, num_blocks, block_size, num_heads,
                               // head_size] for flash attention
                               // [num_blocks, 2, block_size, num_heads, head_size] for flash infer
                               // [num_blocks, block_size, head_size] for MLA

    torch::Tensor& slot_mapping,  // [num_selected_tokens]
    const torch::Tensor& selected_tokens_per_chunk,  // [num_selected_tokens], selected token indices in lmc_key_value_cache
    const TransferDirection direction, const GPUKVFormat gpu_kv_format,
    const bool token_major, // true: lmc_key_value_cache is
                            // [num_tokens, 2, num_heads*head_size]
                            // false: lmc_key_value_cache is
                            // [2, num_tokens, num_heads*head_size]
    const int target_head // selected kv head to transfer
) {
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);

  int64_t* vllm_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(vllm_key_value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  const int64_t* selected_tokens_ptr = get_kernel_ptr<const int64_t, const torch::Tensor>(selected_tokens_per_chunk);

  int elements_per_entry = 8 / vllm_key_value_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads;
  int head_size_in_64bit;
  int block_size;

  const bool use_mla = lmc::is_mla(gpu_kv_format);

  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    num_heads = 1;
    block_size = vllm_key_value_cache.size(1);
    head_size_in_64bit = vllm_key_value_cache.size(2) / elements_per_entry;
  } else {
    num_heads = vllm_key_value_cache.size(3);
    head_size_in_64bit = vllm_key_value_cache.size(4) / elements_per_entry;
    block_size = vllm_key_value_cache.size(2);
  }

  int lmc_stride;
  int lmc_value_offset;
  if (use_mla) {
    // MLA format: [num_tokens, aligned_head_size]
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = 0;  // No separate K/V for MLA
  } else if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int vllm_block_key_stride_in_64bit;
  int vllm_value_offset;
  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = 0;  // No separate K/V for MLA
  } else if (gpu_kv_format == GPUKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(1) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(0) / elements_per_entry;
  } else {  // gpu_kv_format == GPUKVFormat::NL_X_NB_TWO_BS_NH_HS
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(1) / elements_per_entry;
  }

  int grid_size;
  if (target_head >= 0 && target_head < num_heads) {
    grid_size = num_tokens;
  } else {
    grid_size = num_tokens * num_heads; // one block per (token, head)
  }
  dim3 grid(grid_size);

  int threads_per_block = std::min(head_size_in_64bit, 256);
  dim3 block(threads_per_block);
  const at::cuda::OptionalCUDAGuard device_guard(
      device_of(vllm_key_value_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Dispatch to the appropriate template specialization based on use_mla
  if (use_mla) {
    lmc::single_layer_head_token_wise_kv_transfer_kernel<int64_t, true>
        <<<grid, block, 0, stream>>>(
            lmc_key_value_cache_ptr, vllm_key_value_cache_ptr, slot_mapping_ptr, selected_tokens_ptr,
            vllm_block_key_stride_in_64bit, vllm_value_offset, lmc_stride,
            lmc_value_offset, num_heads, head_size_in_64bit, block_size,
            direction, target_head);
  } else {
    lmc::single_layer_head_token_wise_kv_transfer_kernel<int64_t, false>
        <<<grid, block, 0, stream>>>(
            lmc_key_value_cache_ptr, vllm_key_value_cache_ptr, slot_mapping_ptr, selected_tokens_ptr,
            vllm_block_key_stride_in_64bit, vllm_value_offset, lmc_stride,
            lmc_value_offset, num_heads, head_size_in_64bit, block_size,
            direction, target_head);
  }
}


void single_layer_sparse_kv_transfer_64_bit(
    std::vector<torch::Tensor>& lmcache_tensors,  // [num_chunks] int64
    torch::Tensor& vllm_kv_cache,                 // [2, num_blocks, block_size, num_heads, head_dim]
    torch::Tensor& slot_mapping,                  // [num_all_tokens] int64
    torch::Tensor& selected_tokens_per_head,      // [num_heads, num_selected_tokens] int64
    int64_t token_start_index,
    int64_t num_tokens_per_chunk
) {
    TORCH_CHECK(vllm_kv_cache.is_cuda(), "vllm_kv_cache must be on GPU");
    TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be on GPU");
    TORCH_CHECK(selected_tokens_per_head.is_cuda(), "selected_tokens_per_head must be on GPU");

    const int64_t* slot_mapping_ptr =
        get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);
    const int64_t* selected_tokens_ptr =
        get_kernel_ptr<const int64_t, const torch::Tensor>(selected_tokens_per_head);

    // Get device pointer from host pointer
    int64_t num_chunks = lmcache_tensors.size();
    auto dev_ptrs_tensor = torch::empty(
        {num_chunks},
        torch::TensorOptions().dtype(torch::kInt64).device(vllm_kv_cache.device())
    );
    int64_t* dev_ptrs_array = dev_ptrs_tensor.data_ptr<int64_t>();
    std::vector<int64_t> host_dev_ptrs(num_chunks);
    for (int64_t i = 0; i < num_chunks; ++i) {
        void* d_ptr = nullptr;
        cudaError_t err = cudaHostGetDevicePointer(
            &d_ptr,
            reinterpret_cast<void*>(lmcache_tensors[i].data_ptr()),
            0
        );
        TORCH_CHECK(err == cudaSuccess,
                    "cudaHostGetDevicePointer failed for chunk ", i,
                    ": ", cudaGetErrorString(err));
        host_dev_ptrs[i] = reinterpret_cast<int64_t>(d_ptr); // device pointer
    }
    cudaMemcpy(dev_ptrs_array,
               host_dev_ptrs.data(),
               num_chunks * sizeof(int64_t),
               cudaMemcpyHostToDevice);

    const auto num_heads = selected_tokens_per_head.size(0);
    const auto num_selected_tokens = selected_tokens_per_head.size(1);
    const auto block_size = vllm_kv_cache.size(2);
    const auto head_dim = vllm_kv_cache.size(4);
    auto vllm_k = vllm_kv_cache[0].contiguous();
    auto vllm_v = vllm_kv_cache[1].contiguous();

    int64_t* vllm_key_cache_ptr =
        get_kernel_ptr<int64_t, torch::Tensor>(vllm_k);
    int64_t* vllm_value_cache_ptr =
        get_kernel_ptr<int64_t, torch::Tensor>(vllm_v);

    auto element_size = vllm_kv_cache.element_size();
    int32_t elements_per_entry = 8 / element_size;
    int32_t head_dim_in_64bit = head_dim / elements_per_entry;

    // stride
    // vllm tensor format: NL_X_TWO_NB_BS_NH_HS
    const int32_t stride_vllm_block = vllm_k.stride(0) / elements_per_entry;   // block_size * num_heads * head_dim
    const int32_t stride_vllm_slot  = vllm_k.stride(1) / elements_per_entry;   // num_heads * head_dim
    const int32_t stride_vllm_head  = vllm_k.stride(2) / elements_per_entry;   // head_dim
    // lmc memory_obj: [num_tokens, 2, num_heads, head_dim]
    const int32_t stride_lm_token = 2 * num_heads * head_dim / elements_per_entry;
    const int32_t stride_lm_kv = num_heads * head_dim / elements_per_entry;
    const int32_t stride_lm_dim = head_dim / elements_per_entry;
    // selected_tokens_per_head [num_heads, num_selected_tokens]
    const int32_t stride_sel_head = selected_tokens_per_head.stride(0);

    // grid / block
    dim3 grid(num_heads, num_selected_tokens);
    dim3 block(std::min(head_dim_in_64bit, 128));

    lmc::single_layer_sparse_kv_transfer_kernel<int64_t><<<grid, block>>>(
        dev_ptrs_array,
        vllm_key_cache_ptr,
        vllm_value_cache_ptr,
        slot_mapping_ptr,
        selected_tokens_ptr,
        token_start_index,
        static_cast<int32_t>(num_chunks),
        static_cast<int32_t>(num_selected_tokens),
        static_cast<int32_t>(num_heads),
        static_cast<int32_t>(num_tokens_per_chunk),
        static_cast<int32_t>(block_size),
        static_cast<int32_t>(head_dim_in_64bit),
        stride_vllm_block,
        stride_vllm_slot,
        stride_vllm_head,
        stride_lm_token,
        stride_lm_kv,
        stride_lm_dim,
        stride_sel_head
    );
}


void single_layer_sparse_kv_transfer_64_bit_addr(
    std::vector<int64_t>& lmcache_tensor_ptrs,    // [num_chunks] device ptrs
    torch::Tensor& vllm_kv_cache,                 // [2, num_blocks, block_size, num_heads, head_dim]
    torch::Tensor& slot_mapping,                  // [num_all_tokens] int64
    torch::Tensor& selected_tokens_per_head,      // [num_heads, num_selected_tokens] int64
    int64_t token_start_index,
    int64_t num_tokens_per_chunk
) {
    TORCH_CHECK(vllm_kv_cache.is_cuda(), "vllm_kv_cache must be on GPU");
    TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be on GPU");
    TORCH_CHECK(selected_tokens_per_head.is_cuda(), "selected_tokens_per_head must be on GPU");

    const int64_t* slot_mapping_ptr =
        get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);
    const int64_t* selected_tokens_ptr =
        get_kernel_ptr<const int64_t, const torch::Tensor>(selected_tokens_per_head);

    // Get device pointer from host pointer
    int64_t num_chunks = lmcache_tensor_ptrs.size();
    auto dev_ptrs_tensor = torch::empty(
        {num_chunks},
        torch::TensorOptions().dtype(torch::kInt64).device(vllm_kv_cache.device())
    );
    int64_t* dev_ptrs_array = dev_ptrs_tensor.data_ptr<int64_t>();
    cudaMemcpy(dev_ptrs_array,
               lmcache_tensor_ptrs.data(),
               num_chunks * sizeof(int64_t),
               cudaMemcpyHostToDevice);

    const auto num_heads = selected_tokens_per_head.size(0);
    const auto num_selected_tokens = selected_tokens_per_head.size(1);
    const auto block_size = vllm_kv_cache.size(2);
    const auto head_dim = vllm_kv_cache.size(4);
    auto vllm_k = vllm_kv_cache[0].contiguous();
    auto vllm_v = vllm_kv_cache[1].contiguous();

    int64_t* vllm_key_cache_ptr =
        get_kernel_ptr<int64_t, torch::Tensor>(vllm_k);
    int64_t* vllm_value_cache_ptr =
        get_kernel_ptr<int64_t, torch::Tensor>(vllm_v);

    auto element_size = vllm_kv_cache.element_size();
    int32_t elements_per_entry = 8 / element_size;
    int32_t head_dim_in_64bit = head_dim / elements_per_entry;

    // stride
    // vllm tensor format: NL_X_TWO_NB_BS_NH_HS
    const int32_t stride_vllm_block = vllm_k.stride(0) / elements_per_entry;   // block_size * num_heads * head_dim
    const int32_t stride_vllm_slot  = vllm_k.stride(1) / elements_per_entry;   // num_heads * head_dim
    const int32_t stride_vllm_head  = vllm_k.stride(2) / elements_per_entry;   // head_dim
    // lmc memory_obj: [num_tokens, 2, num_heads, head_dim]
    const int32_t stride_lm_token = 2 * num_heads * head_dim / elements_per_entry;
    const int32_t stride_lm_kv = num_heads * head_dim / elements_per_entry;
    const int32_t stride_lm_dim = head_dim / elements_per_entry;
    // selected_tokens_per_head [num_heads, num_selected_tokens]
    const int32_t stride_sel_head = selected_tokens_per_head.stride(0);

    // grid / block
    dim3 grid(num_heads, num_selected_tokens);
    dim3 block(std::min(head_dim_in_64bit, 128));

    lmc::single_layer_sparse_kv_transfer_kernel<int64_t><<<grid, block>>>(
        dev_ptrs_array,
        vllm_key_cache_ptr,
        vllm_value_cache_ptr,
        slot_mapping_ptr,
        selected_tokens_ptr,
        token_start_index,
        static_cast<int32_t>(num_chunks),
        static_cast<int32_t>(num_selected_tokens),
        static_cast<int32_t>(num_heads),
        static_cast<int32_t>(num_tokens_per_chunk),
        static_cast<int32_t>(block_size),
        static_cast<int32_t>(head_dim_in_64bit),
        stride_vllm_block,
        stride_vllm_slot,
        stride_vllm_head,
        stride_lm_token,
        stride_lm_kv,
        stride_lm_dim,
        stride_sel_head
    );
}


void single_layer_sparse_clustered_flattened_kv_transfer_64_bit_addr(
    std::vector<int64_t>& lmcache_tensor_ptrs,    // [num_chunks] device ptrs
    torch::Tensor& vllm_kv_cache,                 // [2, num_blocks, block_size, num_heads, head_dim]
    torch::Tensor& slot_mapping,                  // [num_all_tokens] int64
    torch::Tensor& selected_clusters,             // int64 [num_heads, num_selected_clusters]
    torch::Tensor& clusters,                      // int32 [num_heads, num_clusters, max_cluster_size]
    torch::Tensor& cluster_size,                  // int32 [num_heads, num_clusters]
    torch::Tensor& cluster_start_index,           // int32 [num_heads, num_selected_clusters]
    int32_t retrieve_budget,
    int64_t token_start_index,
    int64_t num_tokens_per_chunk
) {
    TORCH_CHECK(vllm_kv_cache.is_cuda(), "vllm_kv_cache must be on GPU");
    TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be on GPU");
    TORCH_CHECK(selected_clusters.is_cuda(), "selected_clusters must be on GPU");
    TORCH_CHECK(clusters.is_cuda(), "clusters must be on GPU");
    TORCH_CHECK(cluster_size.is_cuda(), "cluster_size must be on GPU");
    TORCH_CHECK(cluster_start_index.is_cuda(), "cluster_start_index must be on GPU");

    const int64_t* slot_mapping_ptr =
        get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);
    const int64_t* selected_clusters_ptr =
        get_kernel_ptr<const int64_t, const torch::Tensor>(selected_clusters);
    const int32_t* clusters_ptr =
        get_kernel_ptr<const int32_t, const torch::Tensor>(clusters);
    const int32_t* cluster_size_ptr =
        get_kernel_ptr<const int32_t, const torch::Tensor>(cluster_size);
    const int32_t* cluster_start_index_ptr =
        get_kernel_ptr<const int32_t, const torch::Tensor>(cluster_start_index);

    // Get device pointer from host pointer
    int64_t num_chunks = lmcache_tensor_ptrs.size();
    auto dev_ptrs_tensor = torch::empty(
        {num_chunks},
        torch::TensorOptions().dtype(torch::kInt64).device(vllm_kv_cache.device())
    );
    int64_t* dev_ptrs_array = dev_ptrs_tensor.data_ptr<int64_t>();
    cudaMemcpy(dev_ptrs_array,
               lmcache_tensor_ptrs.data(),
               num_chunks * sizeof(int64_t),
               cudaMemcpyHostToDevice);

    // selected info
    const auto num_heads = selected_clusters.size(0);
    const auto num_selected_clusters = selected_clusters.size(1);

    // vllm info
    auto vllm_k = vllm_kv_cache[0].contiguous();
    auto vllm_v = vllm_kv_cache[1].contiguous();
    int64_t* vllm_key_cache_ptr =
        get_kernel_ptr<int64_t, torch::Tensor>(vllm_k);
    int64_t* vllm_value_cache_ptr =
        get_kernel_ptr<int64_t, torch::Tensor>(vllm_v);
    const auto block_size = vllm_kv_cache.size(2);
    const auto head_dim = vllm_kv_cache.size(4);
    auto element_size = vllm_kv_cache.element_size();
    int32_t elements_per_entry = 8 / element_size;
    int32_t head_dim_in_64bit = head_dim / elements_per_entry;

    // stride
    // vllm tensor format: NL_X_TWO_NB_BS_NH_HS
    const int32_t stride_vllm_block = vllm_k.stride(0) / elements_per_entry;   // block_size * num_heads * head_dim
    const int32_t stride_vllm_slot  = vllm_k.stride(1) / elements_per_entry;   // num_heads * head_dim
    const int32_t stride_vllm_head  = vllm_k.stride(2) / elements_per_entry;   // head_dim
    // lmc memory_obj: [num_tokens, 2, num_heads, head_dim]
    const int32_t stride_lm_token = 2 * num_heads * head_dim / elements_per_entry;
    const int32_t stride_lm_kv = num_heads * head_dim / elements_per_entry;
    const int32_t stride_lm_dim = head_dim / elements_per_entry;
    // clusters [num_heads, num_clusters, max_cluster_size]
    const int32_t stride_cluster_head = clusters.stride(0);
    const int32_t stride_cluster_num = clusters.stride(1);
    // selected_clusters [num_heads, num_selected_clusters]
    const int32_t stride_sel_head = selected_clusters.stride(0);
    // cluster_size [num_heads, num_clusters]
    const int32_t stride_cs_head = cluster_size.stride(0);

    // grid / block
    dim3 grid(num_heads, retrieve_budget);
    dim3 block(std::min(head_dim_in_64bit, 128));

    lmc::single_layer_sparse_clustered_flattened_kv_transfer_kernel<int64_t><<<grid, block, num_selected_clusters * sizeof(int)>>>(
        dev_ptrs_array,
        vllm_key_cache_ptr,
        vllm_value_cache_ptr,
        slot_mapping_ptr,
        clusters_ptr,
        selected_clusters_ptr,
        cluster_size_ptr,
        cluster_start_index_ptr,
        retrieve_budget,
        static_cast<int32_t>(num_chunks),
        static_cast<int32_t>(token_start_index),
        static_cast<int32_t>(num_selected_clusters),
        static_cast<int32_t>(num_heads),
        static_cast<int32_t>(num_tokens_per_chunk),
        static_cast<int32_t>(block_size),
        static_cast<int32_t>(head_dim_in_64bit),
        stride_vllm_block,
        stride_vllm_slot,
        stride_vllm_head,
        stride_lm_token,
        stride_lm_kv,
        stride_lm_dim,
        stride_cluster_head,
        stride_cluster_num,
        stride_cs_head,
        stride_sel_head
    );
}


void single_layer_sparse_clustered_kv_transfer_64_bit_addr(
    std::vector<int64_t>& lmcache_tensor_ptrs,    // [num_chunks] device ptrs
    torch::Tensor& vllm_kv_cache,                 // [2, num_blocks, block_size, num_heads, head_dim]
    torch::Tensor& slot_mapping,                  // [num_all_tokens] int64
    torch::Tensor& selected_clusters,             // int64 [num_heads, num_selected_clusters]
    torch::Tensor& clusters,                      // int32 [num_heads, num_clusters, max_cluster_size]
    torch::Tensor& cluster_size,                  // int32 [num_heads, num_clusters]
    torch::Tensor& cluster_start_index,           // int32 [num_heads, num_selected_clusters]
    int32_t retrieve_budget,
    int64_t token_start_index,
    int64_t num_tokens_per_chunk
) {
    TORCH_CHECK(vllm_kv_cache.is_cuda(), "vllm_kv_cache must be on GPU");
    TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be on GPU");
    TORCH_CHECK(selected_clusters.is_cuda(), "selected_clusters must be on GPU");
    TORCH_CHECK(clusters.is_cuda(), "clusters must be on GPU");
    TORCH_CHECK(cluster_size.is_cuda(), "cluster_size must be on GPU");
    TORCH_CHECK(cluster_start_index.is_cuda(), "cluster_start_index must be on GPU");

    const int64_t* slot_mapping_ptr =
        get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);
    const int64_t* selected_clusters_ptr =
        get_kernel_ptr<const int64_t, const torch::Tensor>(selected_clusters);
    const int32_t* clusters_ptr =
        get_kernel_ptr<const int32_t, const torch::Tensor>(clusters);
    const int32_t* cluster_size_ptr =
        get_kernel_ptr<const int32_t, const torch::Tensor>(cluster_size);
    const int32_t* cluster_start_index_ptr =
        get_kernel_ptr<const int32_t, const torch::Tensor>(cluster_start_index);

    // Get device pointer from host pointer
    int64_t num_chunks = lmcache_tensor_ptrs.size();
    auto dev_ptrs_tensor = torch::empty(
        {num_chunks},
        torch::TensorOptions().dtype(torch::kInt64).device(vllm_kv_cache.device())
    );
    int64_t* dev_ptrs_array = dev_ptrs_tensor.data_ptr<int64_t>();
    cudaMemcpy(dev_ptrs_array,
               lmcache_tensor_ptrs.data(),
               num_chunks * sizeof(int64_t),
               cudaMemcpyHostToDevice);

    // selected info
    const auto num_heads = selected_clusters.size(0);
    const auto num_selected_clusters = selected_clusters.size(1);

    // vllm info
    auto vllm_k = vllm_kv_cache[0].contiguous();
    auto vllm_v = vllm_kv_cache[1].contiguous();
    int64_t* vllm_key_cache_ptr =
        get_kernel_ptr<int64_t, torch::Tensor>(vllm_k);
    int64_t* vllm_value_cache_ptr =
        get_kernel_ptr<int64_t, torch::Tensor>(vllm_v);
    const auto block_size = vllm_kv_cache.size(2);
    const auto head_dim = vllm_kv_cache.size(4);
    auto element_size = vllm_kv_cache.element_size();
    int32_t elements_per_entry = 8 / element_size;
    int32_t head_dim_in_64bit = head_dim / elements_per_entry;

    // stride
    // vllm tensor format: NL_X_TWO_NB_BS_NH_HS
    const int32_t stride_vllm_block = vllm_k.stride(0) / elements_per_entry;   // block_size * num_heads * head_dim
    const int32_t stride_vllm_slot  = vllm_k.stride(1) / elements_per_entry;   // num_heads * head_dim
    const int32_t stride_vllm_head  = vllm_k.stride(2) / elements_per_entry;   // head_dim
    // lmc memory_obj: [num_tokens, 2, num_heads, head_dim]
    const int32_t stride_lm_token = 2 * num_heads * head_dim / elements_per_entry;
    const int32_t stride_lm_kv = num_heads * head_dim / elements_per_entry;
    const int32_t stride_lm_dim = head_dim / elements_per_entry;
    // clusters [num_heads, num_clusters, max_cluster_size]
    const int32_t stride_cluster_head = clusters.stride(0);
    const int32_t stride_cluster_num = clusters.stride(1);
    // selected_clusters [num_heads, num_selected_clusters]
    const int32_t stride_sel_head = selected_clusters.stride(0);
    // cluster_size [num_heads, num_clusters]
    const int32_t stride_cs_head = cluster_size.stride(0);

    // grid / block
    dim3 grid(num_heads, num_selected_clusters);
    dim3 block(std::min(head_dim_in_64bit, 128));

    lmc::single_layer_sparse_clustered_kv_transfer_kernel<int64_t><<<grid, block>>>(
        dev_ptrs_array,
        vllm_key_cache_ptr,
        vllm_value_cache_ptr,
        slot_mapping_ptr,
        clusters_ptr,
        selected_clusters_ptr,
        cluster_size_ptr,
        cluster_start_index_ptr,
        retrieve_budget,
        static_cast<int32_t>(num_chunks),
        static_cast<int32_t>(token_start_index),
        static_cast<int32_t>(num_selected_clusters),
        static_cast<int32_t>(num_heads),
        static_cast<int32_t>(num_tokens_per_chunk),
        static_cast<int32_t>(block_size),
        static_cast<int32_t>(head_dim_in_64bit),
        stride_vllm_block,
        stride_vllm_slot,
        stride_vllm_head,
        stride_lm_token,
        stride_lm_kv,
        stride_lm_dim,
        stride_cluster_head,
        stride_cluster_num,
        stride_cs_head,
        stride_sel_head
    );
}


void single_layer_sparse_kv_transfer(
    std::vector<torch::Tensor>& lmcache_tensors,           // [num_chunks] int64
    torch::Tensor& vllm_kv_cache,          // [2, num_blocks, block_size, num_heads, head_dim]
    torch::Tensor& slot_mapping,            // [num_all_tokens] int64
    torch::Tensor& selected_tokens_per_head,// [num_heads, num_selected_tokens] int64
    int64_t token_start_index,
    int64_t num_tokens_per_chunk
) {
    TORCH_CHECK(vllm_kv_cache.is_cuda(), "vllm_kv_cache must be on GPU");
    TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be on GPU");
    TORCH_CHECK(selected_tokens_per_head.is_cuda(), "selected_tokens_per_head must be on GPU");

    const int64_t* slot_mapping_ptr =
        get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);
    const int64_t* selected_tokens_ptr =
        get_kernel_ptr<const int64_t, const torch::Tensor>(selected_tokens_per_head);

    // Get device pointer from host pointer
    int64_t num_chunks = lmcache_tensors.size();
    auto dev_ptrs_tensor = torch::empty(
        {num_chunks},
        torch::TensorOptions().dtype(torch::kInt64).device(vllm_kv_cache.device())
    );
    int64_t* dev_ptrs_array = dev_ptrs_tensor.data_ptr<int64_t>();
    std::vector<int64_t> host_dev_ptrs(num_chunks);
    for (int64_t i = 0; i < num_chunks; ++i) {
        void* d_ptr = nullptr;
        cudaError_t err = cudaHostGetDevicePointer(
            &d_ptr,
            reinterpret_cast<void*>(lmcache_tensors[i].data_ptr()),
            0
        );
        TORCH_CHECK(err == cudaSuccess,
                    "cudaHostGetDevicePointer failed for chunk ", i,
                    ": ", cudaGetErrorString(err));
        host_dev_ptrs[i] = reinterpret_cast<int64_t>(d_ptr); // device pointer
    }
    cudaMemcpy(dev_ptrs_array,
               host_dev_ptrs.data(),
               num_chunks * sizeof(int64_t),
               cudaMemcpyHostToDevice);

    const auto num_heads = selected_tokens_per_head.size(0);
    const auto num_selected_tokens = selected_tokens_per_head.size(1);
    const auto num_blocks = vllm_kv_cache.size(1);
    const auto block_size = vllm_kv_cache.size(2);
    const auto head_dim = vllm_kv_cache.size(4);
    auto vllm_k = vllm_kv_cache[0].contiguous();
    auto vllm_v = vllm_kv_cache[1].contiguous();

    // stride
    const int stride_vllm_block = vllm_k.stride(0);   // block_size * num_heads * head_dim
    const int stride_vllm_slot  = vllm_k.stride(1);   // num_heads * head_dim
    const int stride_vllm_head  = vllm_k.stride(2);   // head_dim

    const int stride_lm_token = 2 * num_heads * head_dim;
    const int stride_lm_kv    = num_heads * head_dim;
    const int stride_lm_dim  = head_dim;

    const int stride_sel_head = selected_tokens_per_head.stride(0);

    // grid / block
    dim3 grid(num_heads, num_selected_tokens);
    int block_threads = std::min(head_dim, static_cast<int64_t>(1024));
    dim3 block(block_threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      vllm_kv_cache.scalar_type(),
      "single_layer_sparse_kv_transfer_kernel",
      [&] {
        lmc::single_layer_sparse_kv_transfer_kernel<scalar_t><<<grid, block>>>(
            dev_ptrs_array,
            vllm_k.data_ptr<scalar_t>(),
            vllm_v.data_ptr<scalar_t>(),
            slot_mapping_ptr,
            selected_tokens_ptr,
            token_start_index,
            static_cast<int32_t>(num_chunks),
            static_cast<int>(num_selected_tokens),
            static_cast<int>(num_heads),
            static_cast<int>(num_tokens_per_chunk),
            static_cast<int>(block_size),
            static_cast<int>(head_dim),
            stride_vllm_block,
            stride_vllm_slot,
            stride_vllm_head,
            stride_lm_token,
            stride_lm_kv,
            stride_lm_dim,
            stride_sel_head
        );
      }
    );
    // cudaDeviceSynchronize();
}

void load_and_reshape_flash(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size]
                    // key/value must be on gpu/pinned cpu

    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
                      // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens],
    const int layer_idx) {
  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);

  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;

  int block_size = key_cache.size(1);

  int key_value_stride = key_value.stride(2) / elements_per_entry;

  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset =
      (layer_idx + num_layers) * key_value.stride(1) / elements_per_entry;

  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::load_and_reshape_flash_kernel<int64_t><<<grid, block, 0, stream>>>(
      key_value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
      block_stride_in_64bit, key_value_stride, num_heads, head_size_in_64bit,
      block_size, key_layer_offset, value_layer_offset);
}

void reshape_and_cache_back_flash(
    torch::Tensor&
        key_value,  // [2, num_layer, num_tokens, num_heads*head_size]
                    // key/value must be on gpu/pinned cpu

    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
                      // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens]
    const int layer_idx) {
  int64_t* key_cache_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_cache);
  int64_t* value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(value_cache);

  int64_t* key_value_ptr = get_kernel_ptr<int64_t, torch::Tensor>(key_value);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = key_cache.size(2);
  int head_size_in_64bit = key_cache.size(3) / elements_per_entry;

  int block_size = key_cache.size(1);

  int key_value_stride = key_value.stride(2) / elements_per_entry;

  int num_layers = key_value.size(1);
  int key_layer_offset = layer_idx * key_value.stride(1) / elements_per_entry;
  int value_layer_offset =
      (layer_idx + num_layers) * key_value.stride(1) / elements_per_entry;

  int block_stride_in_64bit = key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::reshape_and_cache_back_flash_kernel<int64_t><<<grid, block, 0, stream>>>(
      key_value_ptr, key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
      block_stride_in_64bit, key_value_stride, num_heads, head_size_in_64bit,
      block_size, key_layer_offset, value_layer_offset);
}

void single_layer_kv_transfer_sgl(
    // torch::Tensor& lmc_key_cache,  // [num_tokens, num_heads*head_size]
    //  key/value must be on gpu/pinned cpu
    // torch::Tensor& lmc_value_cache,  // [num_tokens, num_heads*head_size]

    torch::Tensor& lmc_key_value_cache,  // [num_tokens, 2, num_heads*head_size]
                                         // or
                                         // [2, num_tokens, num_heads*head_size]

    torch::Tensor&
        sgl_key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        sgl_value_cache,  // [num_blocks, block_size, num_heads, head_size]
                          // key_cache/value_cache must be on gpu
    torch::Tensor& slot_mapping,  // [num_tokens]
    const TransferDirection direction,
    const bool token_major  // true: lmc_key_value_cache is
                            // [num_tokens, 2, num_heads*head_size]
                            // false: lmc_key_value_cache is
                            // [2, num_tokens, num_heads*head_size]
) {
  // int64_t* lmc_key_cache_ptr = get_kernel_ptr<int64_t,
  // torch::Tensor>(lmc_key_cache); int64_t* lmc_value_cache_ptr =
  // get_kernel_ptr<int64_t, torch::Tensor>(lmc_value_cache);
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);

  int64_t* sgl_key_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(sgl_key_cache);
  int64_t* sgl_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(sgl_value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / sgl_key_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads = sgl_key_cache.size(2);
  int head_size_in_64bit = sgl_key_cache.size(3) / elements_per_entry;

  int block_size = sgl_key_cache.size(1);

  int lmc_stride;
  int lmc_value_offset;
  if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int block_stride_in_64bit = sgl_key_cache.stride(0) / elements_per_entry;
  TORCH_CHECK(sgl_key_cache.stride(0) == sgl_value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(sgl_key_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  lmc::single_layer_kv_transfer_sgl_kernel<int64_t><<<grid, block, 0, stream>>>(
      lmc_key_value_cache_ptr, sgl_key_cache_ptr, sgl_value_cache_ptr,
      slot_mapping_ptr, block_stride_in_64bit, lmc_stride, lmc_value_offset,
      num_heads, head_size_in_64bit, block_size, direction);
}

/**
 * Perform asynchronous memory copy between lmcache host buffer (memory obj)
 * and a device buffer.
 * The copy will be performed asynchronously on the current CUDA stream.
 * They copy will be split into multiple smaller copies based on the host buffer
 * offset and host buffer alignment requirements.
 *
 * @param dest Destination pointer (device or host)
 * @param src Source pointer (device or host)
 * @param nbytes Number of bytes to copy
 * @param direction H2D or D2H
 * @param host_buffer_offset the virtual offset in the lmcache memory allocator
 * @param host_buffer_alignments the alignment (i.e., cudaHostRegister
 * granularity) requirement of the host buffer. Must be power of two.
 */
void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments) {
  // Check that host_buffer_alignments is power of two
  TORCH_CHECK((host_buffer_alignments & (host_buffer_alignments - 1)) == 0,
              "host_buffer_alignments must be power of two");

  size_t offset = 0;
  const size_t mask = host_buffer_alignments - 1;
  cudaMemcpyKind kind = (direction == TransferDirection::H2D)
                            ? cudaMemcpyHostToDevice
                            : cudaMemcpyDeviceToHost;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  while (offset < nbytes) {
    size_t current_src = src + offset;
    size_t current_dest = dest + offset;

    size_t aligned_area_end =
        ((offset + host_buffer_offset) & ~mask) + host_buffer_alignments;
    size_t real_end = min(host_buffer_offset + nbytes, aligned_area_end);
    size_t max_nbytes = real_end - offset - host_buffer_offset;

    CHECK_CUDA_CALL(cudaMemcpyAsync(reinterpret_cast<void*>(current_dest),
                                    reinterpret_cast<const void*>(current_src),
                                    max_nbytes, kind, stream));

    offset += max_nbytes;
  }
}

std::future<std::vector<uintptr_t> > AsyncClusterMetaManager::BatchGetDevicePtr(std::vector<std::string>& keys) {
  return std::async(std::launch::async, [this, keys]() {
    std::vector<uintptr_t> res;
    for (auto &key : keys) {
      if (chunk_storage.find(key) != chunk_storage.end()) {
        res.emplace_back(reinterpret_cast<uintptr_t>(chunk_storage[key]));
      }
    }
    return res;
  });
}

void AsyncClusterMetaManager::Put(const std::string &key, torch::Tensor& obj) {
  int64_t* ptr =
    get_kernel_ptr<int64_t, const torch::Tensor>(obj);

  chunk_storage[key] = ptr;
}


std::future<std::vector<uintptr_t> > ThreadPoolAsyncClusterMetaManager::BatchGetDevicePtr(std::vector<std::string>& keys) {

  // 创建 promise 和 future
  auto promise = std::make_shared<std::promise<std::vector<uintptr_t>>>();
  std::future<std::vector<uintptr_t>> future = promise->get_future();

  // 封装任务
  auto task = [this, keys, promise]() {
    try {
      std::vector<uintptr_t> res;
      {
        std::shared_lock<std::shared_mutex> lock(storage_mutex_);
        for (const auto& key : keys) {
          auto it = chunk_storage.find(key);
          if (it != chunk_storage.end()) {
            res.emplace_back(reinterpret_cast<uintptr_t>(it->second));
          } else {
            std::cout << "[BatchGetDevicePtr] ERROR: key " << key << " not found!" << std::endl;
          }
        }
      }
      promise->set_value(std::move(res));
    } catch (...) {
      promise->set_exception(std::current_exception());
    }
  };

  // 将任务放入队列
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    task_queue_.push(std::move(task));
  }
  cv_.notify_one();

  return future;
}

void ThreadPoolAsyncClusterMetaManager::Put(const std::string &key, torch::Tensor& obj) {
  int64_t* ptr =
    get_kernel_ptr<int64_t, const torch::Tensor>(obj);

  std::unique_lock<std::shared_mutex> lock(storage_mutex_);
  chunk_storage[key] = ptr;
}
