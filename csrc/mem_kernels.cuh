// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <vector>
#include <unordered_map>
#include <iostream>
#include <chrono>
#include <future>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <functional>

// #ifndef MEM_KERNELS_CUH
// #define MEM_KERNELS_CUH

enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

/*
Symbol Reference:
NL: number of layers
NB: number of blocks/pages
BS: block/page size
NBBS: block/page buffer size = NB * BS
NH: number of heads
HS: head size
TWO: 2
ONE: 1

_ means a dimension within the same tensor
_X_ means a dimension across a list

A_X_B_X_C_D_E means:
kv_cache: List[List[torch.Tensor]]
len(kv_cache) = A
len(kv_cache[0]) = B
kv_cache[0][0].shape = (C, D, E)

The logic for identifying the format currently lives in
`lmcache/v1/gpu_connector/utils.py`
*/
enum class GPUKVFormat : int {
  NB_NL_TWO_BS_NH_HS = 0,
  /*
  used by:
  - vLLM CROSS_LAYER mode
  */

  NL_X_TWO_NB_BS_NH_HS = 1,
  /*
  used by:
  - vLLM non-MLA flash attention
  */

  NL_X_NB_TWO_BS_NH_HS = 2,
  /*
  used by:
  - vLLM non-MLA flash infer
  */

  NL_X_NB_BS_HS = 3,
  /*
  used by:
  - vLLM MLA
  */

  TWO_X_NL_X_NBBS_NH_HS = 4,
  /*
  used by:
  - SGLang MHA (flash attention and flash infer)
  */

  NL_X_NBBS_ONE_HS = 5,
  /*
  used by:
  - SGLang MLA
  */
};

void multi_layer_kv_transfer(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const GPUKVFormat gpu_kv_format, const int block_size = 0,
    const int skip_prefix_n_tokens = 0);

// collapses to multi_layer_kv_transfer for MLA
void multi_layer_kv_transfer_unilateral(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const GPUKVFormat gpu_kv_format);

void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_value_cache,
                              torch::Tensor& slot_mapping,
                              const TransferDirection direction,
                              const GPUKVFormat gpu_kv_format,
                              const bool token_major = false);

void single_layer_head_token_wise_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_value_cache,
                              torch::Tensor& slot_mapping,
                              const torch::Tensor& selected_tokens_per_chunk,
                              const TransferDirection direction,
                              const GPUKVFormat gpu_kv_format,
                              const bool token_major = false,
                              const int target_head = -1);

void single_layer_kv_transfer_sgl(torch::Tensor& lmc_key_value_cache,
                                  torch::Tensor& sgl_key_cache,
                                  torch::Tensor& sgl_value_cache,
                                  torch::Tensor& slot_mapping,
                                  const TransferDirection direction,
                                  const bool token_major = false);

void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments);

// deprecated / unused except in unit tests
void load_and_reshape_flash(torch::Tensor& key_value, torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping, const int layer_idx);

// deprecated / unused except in unit tests
void reshape_and_cache_back_flash(torch::Tensor& key_value,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache,
                                  torch::Tensor& slot_mapping,
                                  const int layer_idx);

void single_layer_sparse_kv_transfer_64_bit(std::vector<torch::Tensor>& lmcache_tensors, // list[num_chunks] int64
                                            torch::Tensor& vllm_kv_cache,                // [2, num_blocks, block_size, num_heads, head_dim]
                                            torch::Tensor& slot_mapping,                 // [num_all_tokens] int64
                                            torch::Tensor& selected_tokens_per_head,     // [num_heads, num_selected_tokens] int64
                                            int64_t token_start_index,
                                            int64_t num_tokens_per_chunk);

void single_layer_sparse_clustered_kv_transfer_64_bit_addr(std::vector<int64_t>& lmcache_tensor_ptrs,    // [num_chunks] device ptrs
                                                           torch::Tensor& vllm_kv_cache,                 // [2, num_blocks, block_size, num_heads, head_dim]
                                                           torch::Tensor& slot_mapping,                  // [num_all_tokens] int64
                                                           torch::Tensor& selected_clusters,             // cpu [num_heads, num_selected_clusters]
                                                           torch::Tensor& clusters,                      // cpu [num_heads, num_clusters, max_cluster_size]
                                                           torch::Tensor& cluster_size,
                                                           torch::Tensor& cluster_start_index,           // [num_heads, num_selected_clusters]
                                                           int32_t retrieve_budget,
                                                           int64_t token_start_index,
                                                           int64_t num_tokens_per_chunk);

void single_layer_sparse_kv_transfer_64_bit_addr(std::vector<int64_t>& lmcache_tensor_ptrs, // list[num_chunks] int64
                                                 torch::Tensor& vllm_kv_cache,                // [2, num_blocks, block_size, num_heads, head_dim]
                                                 torch::Tensor& slot_mapping,                 // [num_all_tokens] int64
                                                 torch::Tensor& selected_tokens_per_head,     // [num_heads, num_selected_tokens] int64
                                                 int64_t token_start_index,
                                                 int64_t num_tokens_per_chunk);

void single_layer_sparse_kv_transfer(std::vector<torch::Tensor>& lmcache_tensors, // list[num_chunks] int64
                                     torch::Tensor& vllm_kv_cache,                // [2, num_blocks, block_size, num_heads, head_dim]
                                     torch::Tensor& slot_mapping,                 // [num_all_tokens] int64
                                     torch::Tensor& selected_tokens_per_head,     // [num_heads, num_selected_tokens] int64
                                     int64_t token_start_index,
                                     int64_t num_tokens_per_chunk);

class AsyncClusterMetaManager {
public:
  AsyncClusterMetaManager() {}

  void Put(const std::string &key, torch::Tensor& obj);
  std::future<std::vector<uintptr_t> > BatchGetDevicePtr(std::vector<std::string>& keys);

  std::unordered_map<std::string, int64_t*> chunk_storage; // key is cache_key, unique for request / layer / chunk
};


class ThreadPoolAsyncClusterMetaManager {
public:
  ThreadPoolAsyncClusterMetaManager()
    : stop_(false) {
    // 1. main thread CPU
    main_cpu_ = sched_getcpu();
    if (main_cpu_ == -1) main_cpu_ = 0; // fallback

    // 2. another cpu
    int num_cpus = sysconf(_SC_NPROCESSORS_CONF);
    chosen_cpu_ = (main_cpu_ + 1) % num_cpus;

    worker_ = std::thread(&ThreadPoolAsyncClusterMetaManager::workerLoop, this);
  }

  ~ThreadPoolAsyncClusterMetaManager() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }
  ThreadPoolAsyncClusterMetaManager(const ThreadPoolAsyncClusterMetaManager&) = delete;
  ThreadPoolAsyncClusterMetaManager& operator=(const ThreadPoolAsyncClusterMetaManager&) = delete;

  void Put(const std::string &key, torch::Tensor& obj);
  std::future<std::vector<uintptr_t> > BatchGetDevicePtr(std::vector<std::string>& keys);

private:
  void workerLoop() {
    // bindToCpu(chosen_cpu_);

    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cv_.wait(lock, [this] { return stop_ || !task_queue_.empty(); });
        if (stop_ && task_queue_.empty()) {
            break;
        }
        task = std::move(task_queue_.front());
        task_queue_.pop();
      }
      task(); // 执行任务
    }
  }

  void bindToCpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_t thread = pthread_self();
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
      // do nothing
    }
  }

  std::unordered_map<std::string, int64_t*> chunk_storage; // key is cache_key, unique for request / layer / chunk
  mutable std::shared_mutex storage_mutex_;   // 保护 chunk_storage

  std::queue<std::function<void()>> task_queue_;
  std::mutex queue_mutex_;
  std::condition_variable cv_;
  std::thread worker_;
  std::atomic<bool> stop_;

  // bind cpu
  int main_cpu_;
  int chosen_cpu_;
};
