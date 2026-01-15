#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <vector>

#include "../kernels/kvx_paged_kv.h"
#include "../kvx_abi.h"

static void check_cuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
    exit(1);
  }
}

static bool nearly_equal(float a, float b, float tol) {
  float diff = fabsf(a - b);
  if (diff <= tol) {
    return true;
  }
  float denom = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
  return diff / denom <= tol;
}

template <typename T>
struct DTypeTraits {};

template <>
struct DTypeTraits<float> {
  static constexpr kvx_dtype_t dtype = KVX_DTYPE_F32;
  static float to_float(float v) { return v; }
  static float from_float(float v) { return v; }
  static float tol() { return 1e-5f; }
};

template <>
struct DTypeTraits<__half> {
  static constexpr kvx_dtype_t dtype = KVX_DTYPE_F16;
  static float to_float(__half v) { return __half2float(v); }
  static __half from_float(float v) { return __float2half_rn(v); }
  static float tol() { return 1e-2f; }
};

template <>
struct DTypeTraits<__nv_bfloat16> {
  static constexpr kvx_dtype_t dtype = KVX_DTYPE_BF16;
  static float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
  static float tol() { return 1e-2f; }
};

static void set_tensor_desc_nhd(kvx_tensor_desc_t *tensor, uint32_t num_blocks,
                                uint32_t block_size, uint32_t num_heads,
                                uint32_t head_dim, kvx_dtype_t dtype,
                                void *data) {
  memset(tensor, 0, sizeof(*tensor));
  tensor->size = sizeof(*tensor);
  tensor->dtype = dtype;
  tensor->layout = KVX_LAYOUT_BLOCK_NHD;
  tensor->memory = KVX_MEMORY_DEVICE;
  tensor->ndim = 4;
  tensor->shape[0] = num_blocks;
  tensor->shape[1] = block_size;
  tensor->shape[2] = num_heads;
  tensor->shape[3] = head_dim;
  tensor->stride[3] = 1;
  tensor->stride[2] = head_dim;
  tensor->stride[1] = (int64_t)num_heads * head_dim;
  tensor->stride[0] = (int64_t)block_size * num_heads * head_dim;
  tensor->data = data;
}

static void set_tensor_desc_hnd(kvx_tensor_desc_t *tensor, uint32_t num_blocks,
                                uint32_t block_size, uint32_t num_heads,
                                uint32_t head_dim, kvx_dtype_t dtype,
                                void *data) {
  memset(tensor, 0, sizeof(*tensor));
  tensor->size = sizeof(*tensor);
  tensor->dtype = dtype;
  tensor->layout = KVX_LAYOUT_BLOCK_HND;
  tensor->memory = KVX_MEMORY_DEVICE;
  tensor->ndim = 4;
  tensor->shape[0] = num_blocks;
  tensor->shape[1] = num_heads;
  tensor->shape[2] = block_size;
  tensor->shape[3] = head_dim;
  tensor->stride[3] = 1;
  tensor->stride[2] = head_dim;
  tensor->stride[1] = (int64_t)block_size * head_dim;
  tensor->stride[0] = (int64_t)num_heads * block_size * head_dim;
  tensor->data = data;
}

static void set_tensor_desc_tokens(kvx_tensor_desc_t *tensor, uint32_t num_tokens,
                                   uint32_t num_heads, uint32_t head_dim,
                                   kvx_dtype_t dtype, void *data,
                                   kvx_memory_type_t memory) {
  memset(tensor, 0, sizeof(*tensor));
  tensor->size = sizeof(*tensor);
  tensor->dtype = dtype;
  tensor->layout = KVX_LAYOUT_BLOCK_CUSTOM;
  tensor->memory = memory;
  tensor->ndim = 3;
  tensor->shape[0] = num_tokens;
  tensor->shape[1] = num_heads;
  tensor->shape[2] = head_dim;
  tensor->stride[2] = 1;
  tensor->stride[1] = head_dim;
  tensor->stride[0] = (int64_t)num_heads * head_dim;
  tensor->data = data;
}

static void set_tensor_desc_seq(kvx_tensor_desc_t *tensor, uint32_t seq_count,
                                uint32_t max_seq_len, uint32_t num_heads,
                                uint32_t head_dim, kvx_dtype_t dtype,
                                void *data) {
  memset(tensor, 0, sizeof(*tensor));
  tensor->size = sizeof(*tensor);
  tensor->dtype = dtype;
  tensor->layout = KVX_LAYOUT_BLOCK_CUSTOM;
  tensor->memory = KVX_MEMORY_DEVICE;
  tensor->ndim = 4;
  tensor->shape[0] = seq_count;
  tensor->shape[1] = max_seq_len;
  tensor->shape[2] = num_heads;
  tensor->shape[3] = head_dim;
  tensor->stride[3] = 1;
  tensor->stride[2] = head_dim;
  tensor->stride[1] = (int64_t)num_heads * head_dim;
  tensor->stride[0] = (int64_t)max_seq_len * num_heads * head_dim;
  tensor->data = data;
}

template <typename T>
static kvx_status_t launch_write(const kvx_cache_desc_t *cache,
                                 const kvx_write_desc_t *write) {
  (void)sizeof(T);
  return kvx_launch_write_kv(cache, write, NULL);
}

template <typename T>
static kvx_status_t launch_write_prefill(const kvx_cache_desc_t *cache,
                                         const kvx_write_desc_t *write,
                                         int32_t tokens_per_seq) {
  (void)sizeof(T);
  return kvx_launch_write_kv_prefill(cache, write, tokens_per_seq, NULL);
}

template <typename T>
static kvx_status_t launch_gather(const kvx_cache_desc_t *cache,
                                  const kvx_gather_desc_t *gather) {
  (void)sizeof(T);
  return kvx_launch_gather_kv(cache, gather, NULL);
}

template <typename T>
static bool run_write_test_case(int32_t num_heads, int32_t head_dim,
                                int32_t tokens_per_seq) {
  const int32_t seq_count = 2;
  const int32_t seq_len = 4;
  const int32_t block_size = 4;
  const int32_t max_blocks_per_seq = (seq_len + block_size - 1) / block_size;
  const int32_t num_blocks = seq_count * max_blocks_per_seq;
  const int32_t num_tokens = seq_count * tokens_per_seq;
  const size_t token_elems = (size_t)num_tokens * num_heads * head_dim;
  const size_t cache_elems =
      (size_t)num_blocks * block_size * num_heads * head_dim;

  std::vector<T> h_key(token_elems);
  std::vector<T> h_value(token_elems);
  for (size_t i = 0; i < token_elems; ++i) {
    float val = 0.1f + 0.01f * (float)i;
    h_key[i] = DTypeTraits<T>::from_float(val);
    h_value[i] = DTypeTraits<T>::from_float(val + 0.5f);
  }

  std::vector<int64_t> h_slots(num_tokens, -1);
  int32_t start_pos = seq_len - tokens_per_seq;
  for (int32_t s = 0; s < seq_count; ++s) {
    for (int32_t t = 0; t < tokens_per_seq; ++t) {
      int32_t token_pos = start_pos + t;
      int32_t logical_block = token_pos / block_size;
      int32_t block_offset = token_pos - logical_block * block_size;
      int32_t block_id = s * max_blocks_per_seq + logical_block;
      int32_t token_idx = s * tokens_per_seq + t;
      h_slots[token_idx] =
          (int64_t)block_id * block_size + (int64_t)block_offset;
    }
  }

  T *d_key = nullptr;
  T *d_value = nullptr;
  T *d_key_cache = nullptr;
  T *d_value_cache = nullptr;
  int64_t *d_slots = nullptr;
  check_cuda(cudaMalloc(&d_key, token_elems * sizeof(T)), "cudaMalloc key");
  check_cuda(cudaMalloc(&d_value, token_elems * sizeof(T)), "cudaMalloc value");
  check_cuda(cudaMalloc(&d_key_cache, cache_elems * sizeof(T)),
             "cudaMalloc key_cache");
  check_cuda(cudaMalloc(&d_value_cache, cache_elems * sizeof(T)),
             "cudaMalloc value_cache");
  check_cuda(cudaMalloc(&d_slots, num_tokens * sizeof(int64_t)),
             "cudaMalloc slots");

  check_cuda(cudaMemcpy(d_key, h_key.data(), token_elems * sizeof(T),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy key");
  check_cuda(cudaMemcpy(d_value, h_value.data(), token_elems * sizeof(T),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy value");
  check_cuda(cudaMemcpy(d_slots, h_slots.data(), num_tokens * sizeof(int64_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy slots");

  kvx_cache_desc_t cache;
  memset(&cache, 0, sizeof(cache));
  cache.size = sizeof(cache);
  cache.num_blocks = num_blocks;
  cache.block_size = block_size;
  cache.num_kv_heads = num_heads;
  cache.head_dim = head_dim;
  set_tensor_desc_nhd(&cache.k, num_blocks, block_size, num_heads, head_dim,
                      DTypeTraits<T>::dtype, d_key_cache);
  set_tensor_desc_nhd(&cache.v, num_blocks, block_size, num_heads, head_dim,
                      DTypeTraits<T>::dtype, d_value_cache);

  kvx_write_desc_t write;
  memset(&write, 0, sizeof(write));
  write.size = sizeof(write);
  write.io.size = sizeof(write.io);
  write.io.num_tokens = num_tokens;
  write.io.num_kv_heads = num_heads;
  write.io.head_dim = head_dim;
  set_tensor_desc_tokens(&write.io.key, num_tokens, num_heads, head_dim,
                         DTypeTraits<T>::dtype, d_key, KVX_MEMORY_DEVICE);
  set_tensor_desc_tokens(&write.io.value, num_tokens, num_heads, head_dim,
                         DTypeTraits<T>::dtype, d_value, KVX_MEMORY_DEVICE);
  write.slots.size = sizeof(write.slots);
  write.slots.dtype = KVX_DTYPE_S64;
  write.slots.token_count = num_tokens;
  write.slots.invalid_slot = -1;
  write.slots.slots = d_slots;

  kvx_status_t st = launch_write<T>(&cache, &write);
  if (st != KVX_STATUS_OK) {
    fprintf(stderr, "write failed: %d\n", st);
    return false;
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize write");

  std::vector<T> h_key_cache(cache_elems);
  std::vector<T> h_value_cache(cache_elems);
  check_cuda(cudaMemcpy(h_key_cache.data(), d_key_cache,
                        cache_elems * sizeof(T), cudaMemcpyDeviceToHost),
             "cudaMemcpy key_cache");
  check_cuda(cudaMemcpy(h_value_cache.data(), d_value_cache,
                        cache_elems * sizeof(T), cudaMemcpyDeviceToHost),
             "cudaMemcpy value_cache");

  bool ok = true;
  float tol = DTypeTraits<T>::tol();
  for (int32_t token = 0; token < num_tokens; ++token) {
    int64_t slot = h_slots[token];
    int32_t block_idx = (int32_t)(slot / block_size);
    int32_t block_offset = (int32_t)(slot - (int64_t)block_idx * block_size);
    for (int32_t head = 0; head < num_heads; ++head) {
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        int64_t idx = (int64_t)token * num_heads * head_dim +
                      (int64_t)head * head_dim + dim;
        int64_t offset =
            (int64_t)block_idx * block_size * num_heads * head_dim +
            (int64_t)block_offset * num_heads * head_dim +
            (int64_t)head * head_dim + dim;
        float expected_k = DTypeTraits<T>::to_float(h_key[idx]);
        float expected_v = DTypeTraits<T>::to_float(h_value[idx]);
        float actual_k = DTypeTraits<T>::to_float(h_key_cache[offset]);
        float actual_v = DTypeTraits<T>::to_float(h_value_cache[offset]);
        if (!nearly_equal(actual_k, expected_k, tol) ||
            !nearly_equal(actual_v, expected_v, tol)) {
          fprintf(stderr,
                  "prefill mismatch: heads=%d head_dim=%d tokens_per_seq=%d "
                  "token=%d head=%d dim=%d expected_k=%f actual_k=%f "
                  "expected_v=%f actual_v=%f\n",
                  num_heads, head_dim, tokens_per_seq, token, head, dim,
                  expected_k, actual_k, expected_v, actual_v);
          ok = false;
          break;
        }
      }
      if (!ok) {
        break;
      }
    }
    if (!ok) {
      break;
    }
  }

  cudaFree(d_key);
  cudaFree(d_value);
  cudaFree(d_key_cache);
  cudaFree(d_value_cache);
  cudaFree(d_slots);
  return ok;
}

template <typename T>
static bool run_write_test_case_hnd(int32_t num_heads, int32_t head_dim,
                                    int32_t tokens_per_seq) {
  const int32_t seq_count = 2;
  const int32_t seq_len = 4;
  const int32_t block_size = 4;
  const int32_t max_blocks_per_seq = (seq_len + block_size - 1) / block_size;
  const int32_t num_blocks = seq_count * max_blocks_per_seq;
  const int32_t num_tokens = seq_count * tokens_per_seq;
  const size_t token_elems = (size_t)num_tokens * num_heads * head_dim;
  const size_t cache_elems =
      (size_t)num_blocks * block_size * num_heads * head_dim;

  std::vector<T> h_key(token_elems);
  std::vector<T> h_value(token_elems);
  for (size_t i = 0; i < token_elems; ++i) {
    float val = 0.1f + 0.01f * (float)i;
    h_key[i] = DTypeTraits<T>::from_float(val);
    h_value[i] = DTypeTraits<T>::from_float(val + 0.5f);
  }

  std::vector<int64_t> h_slots(num_tokens, -1);
  int32_t start_pos = seq_len - tokens_per_seq;
  for (int32_t s = 0; s < seq_count; ++s) {
    for (int32_t t = 0; t < tokens_per_seq; ++t) {
      int32_t token_pos = start_pos + t;
      int32_t logical_block = token_pos / block_size;
      int32_t block_offset = token_pos - logical_block * block_size;
      int32_t block_id = s * max_blocks_per_seq + logical_block;
      int32_t token_idx = s * tokens_per_seq + t;
      h_slots[token_idx] =
          (int64_t)block_id * block_size + (int64_t)block_offset;
    }
  }

  T *d_key = nullptr;
  T *d_value = nullptr;
  T *d_key_cache = nullptr;
  T *d_value_cache = nullptr;
  int64_t *d_slots = nullptr;
  check_cuda(cudaMalloc(&d_key, token_elems * sizeof(T)), "cudaMalloc key");
  check_cuda(cudaMalloc(&d_value, token_elems * sizeof(T)), "cudaMalloc value");
  check_cuda(cudaMalloc(&d_key_cache, cache_elems * sizeof(T)),
             "cudaMalloc key_cache");
  check_cuda(cudaMalloc(&d_value_cache, cache_elems * sizeof(T)),
             "cudaMalloc value_cache");
  check_cuda(cudaMalloc(&d_slots, num_tokens * sizeof(int64_t)),
             "cudaMalloc slots");

  check_cuda(cudaMemcpy(d_key, h_key.data(), token_elems * sizeof(T),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy key");
  check_cuda(cudaMemcpy(d_value, h_value.data(), token_elems * sizeof(T),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy value");
  check_cuda(cudaMemcpy(d_slots, h_slots.data(), num_tokens * sizeof(int64_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy slots");

  kvx_cache_desc_t cache;
  memset(&cache, 0, sizeof(cache));
  cache.size = sizeof(cache);
  cache.num_blocks = num_blocks;
  cache.block_size = block_size;
  cache.num_kv_heads = num_heads;
  cache.head_dim = head_dim;
  set_tensor_desc_hnd(&cache.k, num_blocks, block_size, num_heads, head_dim,
                      DTypeTraits<T>::dtype, d_key_cache);
  set_tensor_desc_hnd(&cache.v, num_blocks, block_size, num_heads, head_dim,
                      DTypeTraits<T>::dtype, d_value_cache);

  kvx_write_desc_t write;
  memset(&write, 0, sizeof(write));
  write.size = sizeof(write);
  write.io.size = sizeof(write.io);
  write.io.num_tokens = num_tokens;
  write.io.num_kv_heads = num_heads;
  write.io.head_dim = head_dim;
  set_tensor_desc_tokens(&write.io.key, num_tokens, num_heads, head_dim,
                         DTypeTraits<T>::dtype, d_key, KVX_MEMORY_DEVICE);
  set_tensor_desc_tokens(&write.io.value, num_tokens, num_heads, head_dim,
                         DTypeTraits<T>::dtype, d_value, KVX_MEMORY_DEVICE);
  write.slots.size = sizeof(write.slots);
  write.slots.dtype = KVX_DTYPE_S64;
  write.slots.token_count = num_tokens;
  write.slots.invalid_slot = -1;
  write.slots.slots = d_slots;

  kvx_status_t st = launch_write<T>(&cache, &write);
  if (st != KVX_STATUS_OK) {
    fprintf(stderr, "write HND failed: %d\n", st);
    return false;
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize write HND");

  std::vector<T> h_key_cache(cache_elems);
  std::vector<T> h_value_cache(cache_elems);
  check_cuda(cudaMemcpy(h_key_cache.data(), d_key_cache,
                        cache_elems * sizeof(T), cudaMemcpyDeviceToHost),
             "cudaMemcpy key_cache");
  check_cuda(cudaMemcpy(h_value_cache.data(), d_value_cache,
                        cache_elems * sizeof(T), cudaMemcpyDeviceToHost),
             "cudaMemcpy value_cache");

  bool ok = true;
  float tol = DTypeTraits<T>::tol();
  for (int32_t token = 0; token < num_tokens; ++token) {
    int64_t slot = h_slots[token];
    int32_t block_idx = (int32_t)(slot / block_size);
    int32_t block_offset = (int32_t)(slot - (int64_t)block_idx * block_size);
    for (int32_t head = 0; head < num_heads; ++head) {
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        int64_t idx = (int64_t)token * num_heads * head_dim +
                      (int64_t)head * head_dim + dim;
        int64_t offset =
            (int64_t)block_idx * num_heads * block_size * head_dim +
            (int64_t)head * block_size * head_dim +
            (int64_t)block_offset * head_dim + dim;
        float expected_k = DTypeTraits<T>::to_float(h_key[idx]);
        float expected_v = DTypeTraits<T>::to_float(h_value[idx]);
        float actual_k = DTypeTraits<T>::to_float(h_key_cache[offset]);
        float actual_v = DTypeTraits<T>::to_float(h_value_cache[offset]);
        if (!nearly_equal(actual_k, expected_k, tol) ||
            !nearly_equal(actual_v, expected_v, tol)) {
          ok = false;
          break;
        }
      }
      if (!ok) {
        break;
      }
    }
    if (!ok) {
      break;
    }
  }

  cudaFree(d_key);
  cudaFree(d_value);
  cudaFree(d_key_cache);
  cudaFree(d_value_cache);
  cudaFree(d_slots);
  return ok;
}

template <typename T>
static bool run_write_test() {
  bool ok = true;
  ok &= run_write_test_case<T>(2, 4, 1);
  ok &= run_write_test_case<T>(1, 7, 1);
  ok &= run_write_test_case<T>(2, 4, 2);
  // HND layout matches the vLLM flash cache layout when heads are strided.
  ok &= run_write_test_case_hnd<T>(2, 4, 1);
  ok &= run_write_test_case_hnd<T>(1, 7, 2);
  return ok;
}

template <typename T>
static bool run_write_prefill_test_case(int32_t num_heads, int32_t head_dim,
                                        int32_t tokens_per_seq) {
  const int32_t seq_count = 2;
  const int32_t seq_len = 4;
  const int32_t block_size = 4;
  const int32_t max_blocks_per_seq = (seq_len + block_size - 1) / block_size;
  const int32_t num_blocks = seq_count * max_blocks_per_seq;
  const int32_t num_tokens = seq_count * tokens_per_seq;
  const size_t token_elems = (size_t)num_tokens * num_heads * head_dim;
  const size_t cache_elems =
      (size_t)num_blocks * block_size * num_heads * head_dim;

  std::vector<T> h_key(token_elems);
  std::vector<T> h_value(token_elems);
  for (size_t i = 0; i < token_elems; ++i) {
    float val = 0.1f + 0.01f * (float)i;
    h_key[i] = DTypeTraits<T>::from_float(val);
    h_value[i] = DTypeTraits<T>::from_float(val + 0.5f);
  }

  std::vector<int64_t> h_slots(num_tokens, -1);
  int32_t start_pos = seq_len - tokens_per_seq;
  for (int32_t s = 0; s < seq_count; ++s) {
    for (int32_t t = 0; t < tokens_per_seq; ++t) {
      int32_t token_pos = start_pos + t;
      int32_t logical_block = token_pos / block_size;
      int32_t block_offset = token_pos - logical_block * block_size;
      int32_t block_id = s * max_blocks_per_seq + logical_block;
      int32_t token_idx = s * tokens_per_seq + t;
      h_slots[token_idx] =
          (int64_t)block_id * block_size + (int64_t)block_offset;
    }
  }

  T *d_key = nullptr;
  T *d_value = nullptr;
  T *d_key_cache = nullptr;
  T *d_value_cache = nullptr;
  int64_t *d_slots = nullptr;
  check_cuda(cudaMalloc(&d_key, token_elems * sizeof(T)), "cudaMalloc key");
  check_cuda(cudaMalloc(&d_value, token_elems * sizeof(T)), "cudaMalloc value");
  check_cuda(cudaMalloc(&d_key_cache, cache_elems * sizeof(T)),
             "cudaMalloc key_cache");
  check_cuda(cudaMalloc(&d_value_cache, cache_elems * sizeof(T)),
             "cudaMalloc value_cache");
  check_cuda(cudaMalloc(&d_slots, num_tokens * sizeof(int64_t)),
             "cudaMalloc slots");

  check_cuda(cudaMemcpy(d_key, h_key.data(), token_elems * sizeof(T),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy key");
  check_cuda(cudaMemcpy(d_value, h_value.data(), token_elems * sizeof(T),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy value");
  check_cuda(cudaMemcpy(d_slots, h_slots.data(), num_tokens * sizeof(int64_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy slots");

  kvx_cache_desc_t cache;
  memset(&cache, 0, sizeof(cache));
  cache.size = sizeof(cache);
  cache.num_blocks = num_blocks;
  cache.block_size = block_size;
  cache.num_kv_heads = num_heads;
  cache.head_dim = head_dim;
  set_tensor_desc_nhd(&cache.k, num_blocks, block_size, num_heads, head_dim,
                      DTypeTraits<T>::dtype, d_key_cache);
  set_tensor_desc_nhd(&cache.v, num_blocks, block_size, num_heads, head_dim,
                      DTypeTraits<T>::dtype, d_value_cache);

  kvx_write_desc_t write;
  memset(&write, 0, sizeof(write));
  write.size = sizeof(write);
  write.io.size = sizeof(write.io);
  write.io.num_tokens = num_tokens;
  write.io.num_kv_heads = num_heads;
  write.io.head_dim = head_dim;
  set_tensor_desc_tokens(&write.io.key, num_tokens, num_heads, head_dim,
                         DTypeTraits<T>::dtype, d_key, KVX_MEMORY_DEVICE);
  set_tensor_desc_tokens(&write.io.value, num_tokens, num_heads, head_dim,
                         DTypeTraits<T>::dtype, d_value, KVX_MEMORY_DEVICE);
  write.slots.size = sizeof(write.slots);
  write.slots.dtype = KVX_DTYPE_S64;
  write.slots.token_count = num_tokens;
  write.slots.invalid_slot = -1;
  write.slots.slots = d_slots;

  kvx_status_t st = launch_write_prefill<T>(&cache, &write, tokens_per_seq);
  if (st != KVX_STATUS_OK) {
    fprintf(stderr, "prefill write failed: %d\n", st);
    return false;
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize prefill write");

  std::vector<T> h_key_cache(cache_elems);
  std::vector<T> h_value_cache(cache_elems);
  check_cuda(cudaMemcpy(h_key_cache.data(), d_key_cache,
                        cache_elems * sizeof(T), cudaMemcpyDeviceToHost),
             "cudaMemcpy key_cache");
  check_cuda(cudaMemcpy(h_value_cache.data(), d_value_cache,
                        cache_elems * sizeof(T), cudaMemcpyDeviceToHost),
             "cudaMemcpy value_cache");

  bool ok = true;
  float tol = DTypeTraits<T>::tol();
  for (int32_t token = 0; token < num_tokens; ++token) {
    int64_t slot = h_slots[token];
    int32_t block_idx = (int32_t)(slot / block_size);
    int32_t block_offset = (int32_t)(slot - (int64_t)block_idx * block_size);
    for (int32_t head = 0; head < num_heads; ++head) {
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        int64_t idx = (int64_t)token * num_heads * head_dim +
                      (int64_t)head * head_dim + dim;
        int64_t offset =
            (int64_t)block_idx * block_size * num_heads * head_dim +
            (int64_t)block_offset * num_heads * head_dim +
            (int64_t)head * head_dim + dim;
        float expected_k = DTypeTraits<T>::to_float(h_key[idx]);
        float expected_v = DTypeTraits<T>::to_float(h_value[idx]);
        float actual_k = DTypeTraits<T>::to_float(h_key_cache[offset]);
        float actual_v = DTypeTraits<T>::to_float(h_value_cache[offset]);
        if (!nearly_equal(actual_k, expected_k, tol) ||
            !nearly_equal(actual_v, expected_v, tol)) {
          ok = false;
          break;
        }
      }
    }
  }

  cudaFree(d_key);
  cudaFree(d_value);
  cudaFree(d_key_cache);
  cudaFree(d_value_cache);
  cudaFree(d_slots);
  return ok;
}

template <typename T>
static bool run_write_prefill_invalid() {
  kvx_cache_desc_t cache = {0};
  kvx_write_desc_t write = {0};
  cache.k.dtype = DTypeTraits<T>::dtype;
  cache.v.dtype = DTypeTraits<T>::dtype;
  kvx_status_t st = launch_write_prefill<T>(&cache, &write, 0);
  if (st != KVX_STATUS_INVALID_ARGUMENT) {
    fprintf(stderr, "prefill invalid args expected=INVALID got=%d\n", st);
  }
  return st == KVX_STATUS_INVALID_ARGUMENT;
}

template <typename T>
static bool run_write_prefill_test() {
  bool ok = true;
  ok &= run_write_prefill_test_case<T>(2, 4, 2);
  ok &= run_write_prefill_test_case<T>(1, 7, 2);
  ok &= run_write_prefill_invalid<T>();
  return ok;
}

template <typename T>
static bool run_gather_test() {
  const int32_t seq_count = 2;
  const int32_t seq_len = 4;
  const int32_t block_size = 4;
  const int32_t num_heads = 2;
  const int32_t head_dim = 4;
  const int32_t max_blocks_per_seq = (seq_len + block_size - 1) / block_size;
  const int32_t num_blocks = seq_count * max_blocks_per_seq;
  const size_t cache_elems =
      (size_t)num_blocks * block_size * num_heads * head_dim;
  const size_t out_elems =
      (size_t)seq_count * seq_len * num_heads * head_dim;

  std::vector<T> h_key_cache(cache_elems);
  std::vector<T> h_value_cache(cache_elems);
  for (size_t i = 0; i < cache_elems; ++i) {
    float val = 0.05f + 0.01f * (float)i;
    h_key_cache[i] = DTypeTraits<T>::from_float(val);
    h_value_cache[i] = DTypeTraits<T>::from_float(val + 0.5f);
  }

  std::vector<int32_t> h_block_table((size_t)seq_count * max_blocks_per_seq);
  for (int32_t s = 0; s < seq_count; ++s) {
    for (int32_t b = 0; b < max_blocks_per_seq; ++b) {
      h_block_table[s * max_blocks_per_seq + b] = s * max_blocks_per_seq + b;
    }
  }
  std::vector<int32_t> h_seq_lens(seq_count, seq_len);

  T *d_key_cache = nullptr;
  T *d_value_cache = nullptr;
  T *d_key_out = nullptr;
  T *d_value_out = nullptr;
  int32_t *d_block_table = nullptr;
  int32_t *d_seq_lens = nullptr;
  check_cuda(cudaMalloc(&d_key_cache, cache_elems * sizeof(T)),
             "cudaMalloc key_cache");
  check_cuda(cudaMalloc(&d_value_cache, cache_elems * sizeof(T)),
             "cudaMalloc value_cache");
  check_cuda(cudaMalloc(&d_key_out, out_elems * sizeof(T)),
             "cudaMalloc key_out");
  check_cuda(cudaMalloc(&d_value_out, out_elems * sizeof(T)),
             "cudaMalloc value_out");
  check_cuda(cudaMalloc(&d_block_table,
                        (size_t)seq_count * max_blocks_per_seq *
                            sizeof(int32_t)),
             "cudaMalloc block_table");
  check_cuda(cudaMalloc(&d_seq_lens, seq_count * sizeof(int32_t)),
             "cudaMalloc seq_lens");

  check_cuda(cudaMemcpy(d_key_cache, h_key_cache.data(),
                        cache_elems * sizeof(T), cudaMemcpyHostToDevice),
             "cudaMemcpy key_cache");
  check_cuda(cudaMemcpy(d_value_cache, h_value_cache.data(),
                        cache_elems * sizeof(T), cudaMemcpyHostToDevice),
             "cudaMemcpy value_cache");
  check_cuda(cudaMemcpy(d_block_table, h_block_table.data(),
                        (size_t)seq_count * max_blocks_per_seq *
                            sizeof(int32_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy block_table");
  check_cuda(cudaMemcpy(d_seq_lens, h_seq_lens.data(),
                        seq_count * sizeof(int32_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy seq_lens");

  kvx_cache_desc_t cache;
  memset(&cache, 0, sizeof(cache));
  cache.size = sizeof(cache);
  cache.num_blocks = num_blocks;
  cache.block_size = block_size;
  cache.num_kv_heads = num_heads;
  cache.head_dim = head_dim;
  set_tensor_desc_nhd(&cache.k, num_blocks, block_size, num_heads, head_dim,
                      DTypeTraits<T>::dtype, d_key_cache);
  set_tensor_desc_nhd(&cache.v, num_blocks, block_size, num_heads, head_dim,
                      DTypeTraits<T>::dtype, d_value_cache);

  kvx_gather_desc_t gather;
  memset(&gather, 0, sizeof(gather));
  gather.size = sizeof(gather);
  gather.io.size = sizeof(gather.io);
  gather.io.num_tokens = seq_count * seq_len;
  gather.io.num_kv_heads = num_heads;
  gather.io.head_dim = head_dim;
  set_tensor_desc_seq(&gather.io.key, seq_count, seq_len, num_heads, head_dim,
                      DTypeTraits<T>::dtype, d_key_out);
  set_tensor_desc_seq(&gather.io.value, seq_count, seq_len, num_heads,
                      head_dim, DTypeTraits<T>::dtype, d_value_out);
  gather.block_table.size = sizeof(gather.block_table);
  gather.block_table.format = KVX_BLOCK_TABLE_PACKED;
  gather.block_table.index_dtype = KVX_DTYPE_S32;
  gather.block_table.seq_count = seq_count;
  gather.block_table.beam_width = 1;
  gather.block_table.max_blocks_per_seq = max_blocks_per_seq;
  gather.block_table.indices = d_block_table;
  gather.block_table.indices_count = seq_count * max_blocks_per_seq;
  gather.seq_lens.size = sizeof(gather.seq_lens);
  gather.seq_lens.dtype = KVX_DTYPE_S32;
  gather.seq_lens.seq_count = seq_count;
  gather.seq_lens.lengths = d_seq_lens;
  gather.max_seq_len = seq_len;

  kvx_status_t st = launch_gather<T>(&cache, &gather);
  if (st != KVX_STATUS_OK) {
    fprintf(stderr, "gather failed: %d\n", st);
    return false;
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize gather");

  std::vector<T> h_key_out(out_elems);
  std::vector<T> h_value_out(out_elems);
  check_cuda(cudaMemcpy(h_key_out.data(), d_key_out, out_elems * sizeof(T),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy key_out");
  check_cuda(cudaMemcpy(h_value_out.data(), d_value_out, out_elems * sizeof(T),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy value_out");

  bool ok = true;
  float tol = DTypeTraits<T>::tol();
  for (int32_t seq = 0; seq < seq_count; ++seq) {
    for (int32_t tok = 0; tok < seq_len; ++tok) {
      int32_t block_id =
          h_block_table[seq * max_blocks_per_seq + (tok / block_size)];
      int32_t block_offset = tok % block_size;
      for (int32_t head = 0; head < num_heads; ++head) {
        for (int32_t dim = 0; dim < head_dim; ++dim) {
          int64_t out_idx = ((int64_t)seq * seq_len + tok) * num_heads *
                                head_dim +
                            (int64_t)head * head_dim + dim;
          int64_t cache_idx =
              (int64_t)block_id * block_size * num_heads * head_dim +
              (int64_t)block_offset * num_heads * head_dim +
              (int64_t)head * head_dim + dim;
          float expected_k = DTypeTraits<T>::to_float(h_key_cache[cache_idx]);
          float expected_v = DTypeTraits<T>::to_float(h_value_cache[cache_idx]);
          float actual_k = DTypeTraits<T>::to_float(h_key_out[out_idx]);
          float actual_v = DTypeTraits<T>::to_float(h_value_out[out_idx]);
          if (!nearly_equal(actual_k, expected_k, tol) ||
              !nearly_equal(actual_v, expected_v, tol)) {
            ok = false;
            break;
          }
        }
      }
    }
  }

  cudaFree(d_key_cache);
  cudaFree(d_value_cache);
  cudaFree(d_key_out);
  cudaFree(d_value_out);
  cudaFree(d_block_table);
  cudaFree(d_seq_lens);
  return ok;
}

int main() {
  bool ok = true;
  auto record = [&](const char *name, bool result) {
    fprintf(stderr, "%s: %s\n", name, result ? "ok" : "fail");
    ok &= result;
  };

  record("write_f32", run_write_test<float>());
  record("write_f16", run_write_test<__half>());
  record("write_bf16", run_write_test<__nv_bfloat16>());

  record("prefill_f32", run_write_prefill_test<float>());
  record("prefill_f16", run_write_prefill_test<__half>());
  record("prefill_bf16", run_write_prefill_test<__nv_bfloat16>());

  record("gather_f32", run_gather_test<float>());
  record("gather_f16", run_gather_test<__half>());
  record("gather_bf16", run_gather_test<__nv_bfloat16>());

  if (!ok) {
    fprintf(stderr, "kvx_kernel_test failed\n");
    return 1;
  }
  printf("kvx_kernel_test passed\n");
  return 0;
}
