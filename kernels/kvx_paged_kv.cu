#include "kvx_paged_kv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef struct kvx_cache_params {
  kvx_layout_t layout;
  int32_t num_blocks;
  int32_t block_size;
  int32_t num_heads;
  int32_t head_dim;
  int32_t pack;
  int64_t stride[KVX_MAX_TENSOR_DIMS];
} kvx_cache_params_t;

static kvx_status_t kvx_fill_cache_params(const kvx_cache_desc_t *cache,
                                          const kvx_tensor_desc_t *tensor,
                                          kvx_cache_params_t *out) {
  if (cache == NULL || tensor == NULL || out == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  out->layout = tensor->layout;
  out->num_blocks = (int32_t)cache->num_blocks;
  out->block_size = (int32_t)cache->block_size;
  out->num_heads = (int32_t)cache->num_kv_heads;
  out->head_dim = (int32_t)cache->head_dim;
  out->pack = 1;
  for (int i = 0; i < KVX_MAX_TENSOR_DIMS; ++i) {
    out->stride[i] = tensor->stride[i];
  }

  switch (tensor->layout) {
    case KVX_LAYOUT_BLOCK_NHD:
    case KVX_LAYOUT_BLOCK_HND:
      return KVX_STATUS_OK;
    case KVX_LAYOUT_BLOCK_HND_PACKED: {
      int64_t pack = tensor->shape[4];
      if (pack <= 0) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if ((cache->head_dim % pack) != 0) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      out->pack = (int32_t)pack;
      return KVX_STATUS_OK;
    }
    default:
      return KVX_STATUS_UNSUPPORTED;
  }
}

__device__ __forceinline__ int64_t kvx_cache_offset(
    const kvx_cache_params_t &params, int64_t block_idx, int64_t block_offset,
    int64_t head, int64_t dim) {
  switch (params.layout) {
    case KVX_LAYOUT_BLOCK_NHD:
      return block_idx * params.stride[0] +
             block_offset * params.stride[1] + head * params.stride[2] +
             dim * params.stride[3];
    case KVX_LAYOUT_BLOCK_HND:
      return block_idx * params.stride[0] + head * params.stride[1] +
             block_offset * params.stride[2] + dim * params.stride[3];
    case KVX_LAYOUT_BLOCK_HND_PACKED: {
      int64_t packed_dim = dim / params.pack;
      int64_t pack_offset = dim - packed_dim * params.pack;
      return block_idx * params.stride[0] + head * params.stride[1] +
             packed_dim * params.stride[2] + block_offset * params.stride[3] +
             pack_offset * params.stride[4];
    }
    default:
      return 0;
  }
}

static bool kvx_is_nhd_contiguous(const kvx_cache_params_t &params) {
  return params.layout == KVX_LAYOUT_BLOCK_NHD && params.stride[3] == 1 &&
         params.stride[2] == params.head_dim &&
         params.stride[1] == (int64_t)params.num_heads * params.head_dim &&
         params.stride[0] ==
             (int64_t)params.block_size * params.num_heads * params.head_dim;
}

static bool kvx_is_aligned(const void *ptr, size_t alignment) {
  return ((uintptr_t)ptr % alignment) == 0u;
}

static bool kvx_is_aligned_16(const void *ptr) {
  return kvx_is_aligned(ptr, 16u);
}

template <typename T, typename SlotT>
__global__ void kvx_write_kv_kernel(
    const T *key, const T *value, T *key_cache, T *value_cache,
    const SlotT *slots, int32_t num_tokens, kvx_cache_params_t k_params,
    kvx_cache_params_t v_params) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)num_tokens * k_params.num_heads * k_params.head_dim;
  if (idx >= total) {
    return;
  }
  int64_t token = idx / (k_params.num_heads * k_params.head_dim);
  int64_t rem = idx - token * (k_params.num_heads * k_params.head_dim);
  int64_t head = rem / k_params.head_dim;
  int64_t dim = rem - head * k_params.head_dim;

  int64_t slot = (int64_t)slots[token];
  if (slot < 0) {
    return;
  }
  int64_t block_idx = slot / k_params.block_size;
  int64_t block_offset = slot - block_idx * k_params.block_size;
  if (block_idx < 0 || block_idx >= k_params.num_blocks) {
    return;
  }

  int64_t k_offset =
      kvx_cache_offset(k_params, block_idx, block_offset, head, dim);
  int64_t v_offset =
      kvx_cache_offset(v_params, block_idx, block_offset, head, dim);
  key_cache[k_offset] = key[idx];
  value_cache[v_offset] = value[idx];
}

template <typename T, typename SlotT, int VecElems>
__global__ void kvx_write_kv_kernel_nhd_vec(
    const T *key, const T *value, T *key_cache, T *value_cache,
    const SlotT *slots, int32_t num_tokens, int32_t block_size,
    int32_t num_blocks, int32_t num_heads, int32_t head_dim,
    int64_t cache_stride0, int64_t cache_stride1, int64_t key_stride,
    int64_t value_stride) {
  int32_t token = (int32_t)blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  int64_t slot = (int64_t)slots[token];
  if (slot < 0) {
    return;
  }
  int64_t block_idx = slot / block_size;
  int64_t block_offset = slot - block_idx * block_size;
  if (block_idx < 0 || block_idx >= num_blocks) {
    return;
  }

  int64_t n_elems = (int64_t)num_heads * head_dim;
  const T *key_src = key + (int64_t)token * key_stride;
  const T *value_src = value + (int64_t)token * value_stride;
  T *key_dst =
      key_cache + block_idx * cache_stride0 + block_offset * cache_stride1;
  T *value_dst =
      value_cache + block_idx * cache_stride0 + block_offset * cache_stride1;

  static_assert(sizeof(int4) == 16, "int4 must be 16 bytes");
  static_assert(16 % sizeof(T) == 0, "Vector size must align with element");
  const int64_t vec_count = n_elems / VecElems;
  const int4 *key_vec = reinterpret_cast<const int4 *>(key_src);
  const int4 *value_vec = reinterpret_cast<const int4 *>(value_src);
  int4 *key_dst_vec = reinterpret_cast<int4 *>(key_dst);
  int4 *value_dst_vec = reinterpret_cast<int4 *>(value_dst);
  for (int64_t i = threadIdx.x; i < vec_count; i += blockDim.x) {
    int4 kv = key_vec[i];
    int4 vv = value_vec[i];
    key_dst_vec[i] = kv;
    value_dst_vec[i] = vv;
  }
}

template <typename T, typename SlotT, int VecBytes>
__global__ void kvx_write_kv_kernel_prefill_nhd_vec_bytes(
    const T *key, const T *value, T *key_cache, T *value_cache,
    const SlotT *slots, int32_t num_tokens, int32_t block_size,
    int32_t num_blocks, int32_t num_heads, int32_t head_dim,
    int64_t cache_stride0, int64_t cache_stride1, int64_t key_stride,
    int64_t value_stride) {
  int32_t token = (int32_t)blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  int64_t slot = (int64_t)slots[token];
  if (slot < 0) {
    return;
  }
  int64_t block_idx = slot / block_size;
  int64_t block_offset = slot - block_idx * block_size;
  if (block_idx < 0 || block_idx >= num_blocks) {
    return;
  }

  constexpr int kVecBytes = VecBytes;
  constexpr int kVecElems = kVecBytes / (int)sizeof(T);
  constexpr int kVecSegments = kVecBytes / 16;
  static_assert(kVecBytes % 16 == 0, "VecBytes must be 16-byte aligned");
  static_assert(kVecBytes % sizeof(T) == 0, "VecBytes must align to dtype");
  static_assert(sizeof(int4) == 16, "int4 must be 16 bytes");

  int64_t n_elems = (int64_t)num_heads * head_dim;
  int64_t vec_count = n_elems / kVecElems;
  int64_t vec_idx = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
  if (vec_idx >= vec_count) {
    return;
  }

  const T *key_src = key + (int64_t)token * key_stride;
  const T *value_src = value + (int64_t)token * value_stride;
  T *key_dst =
      key_cache + block_idx * cache_stride0 + block_offset * cache_stride1;
  T *value_dst =
      value_cache + block_idx * cache_stride0 + block_offset * cache_stride1;

  const int4 *key_vec = reinterpret_cast<const int4 *>(key_src);
  const int4 *value_vec = reinterpret_cast<const int4 *>(value_src);
  int4 *key_dst_vec = reinterpret_cast<int4 *>(key_dst);
  int4 *value_dst_vec = reinterpret_cast<int4 *>(value_dst);
  int64_t seg_base = vec_idx * kVecSegments;
#pragma unroll
  for (int i = 0; i < kVecSegments; ++i) {
    int4 kv = key_vec[seg_base + i];
    int4 vv = value_vec[seg_base + i];
    key_dst_vec[seg_base + i] = kv;
    value_dst_vec[seg_base + i] = vv;
  }
}

template <typename T, typename IndexT, typename LenT>
__global__ void kvx_gather_kv_kernel(
    const T *key_cache, const T *value_cache, T *key_out, T *value_out,
    const IndexT *block_table, const LenT *seq_lens, int32_t seq_count,
    int32_t max_blocks_per_seq, int32_t max_seq_len,
    kvx_cache_params_t k_params, kvx_cache_params_t v_params) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total =
      (int64_t)seq_count * max_seq_len * k_params.num_heads * k_params.head_dim;
  if (idx >= total) {
    return;
  }

  int64_t token_idx = idx / (k_params.num_heads * k_params.head_dim);
  int64_t rem = idx - token_idx * (k_params.num_heads * k_params.head_dim);
  int64_t head = rem / k_params.head_dim;
  int64_t dim = rem - head * k_params.head_dim;

  int32_t seq = (int32_t)(token_idx / max_seq_len);
  int32_t tok = (int32_t)(token_idx - (int64_t)seq * max_seq_len);
  int64_t seq_len = (int64_t)seq_lens[seq];
  if (tok >= seq_len) {
    return;
  }
  int32_t logical_block = tok / k_params.block_size;
  if (logical_block >= max_blocks_per_seq) {
    return;
  }
  int64_t block_id =
      (int64_t)block_table[seq * max_blocks_per_seq + logical_block];
  if (block_id < 0) {
    return;
  }

  int64_t block_offset = tok - (int64_t)logical_block * k_params.block_size;
  int64_t k_offset =
      kvx_cache_offset(k_params, block_id, block_offset, head, dim);
  int64_t v_offset =
      kvx_cache_offset(v_params, block_id, block_offset, head, dim);
  key_out[idx] = key_cache[k_offset];
  value_out[idx] = value_cache[v_offset];
}

static kvx_status_t kvx_validate_io_tensor_3d(const kvx_tensor_desc_t *tensor,
                                              int32_t num_tokens,
                                              int32_t num_heads,
                                              int32_t head_dim) {
  if (tensor == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->ndim != 3) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->shape[0] != num_tokens || tensor->shape[1] != num_heads ||
      tensor->shape[2] != head_dim) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->stride[2] != 1 ||
      tensor->stride[1] != (int64_t)head_dim ||
      tensor->stride[0] != (int64_t)num_heads * head_dim) {
    return KVX_STATUS_UNSUPPORTED;
  }
  return KVX_STATUS_OK;
}

static kvx_status_t kvx_validate_io_tensor_4d(const kvx_tensor_desc_t *tensor,
                                              int32_t seq_count,
                                              int32_t max_seq_len,
                                              int32_t num_heads,
                                              int32_t head_dim) {
  if (tensor == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->ndim != 4) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->shape[0] != seq_count || tensor->shape[1] != max_seq_len ||
      tensor->shape[2] != num_heads || tensor->shape[3] != head_dim) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->stride[3] != 1 ||
      tensor->stride[2] != (int64_t)head_dim ||
      tensor->stride[1] != (int64_t)num_heads * head_dim ||
      tensor->stride[0] != (int64_t)max_seq_len * num_heads * head_dim) {
    return KVX_STATUS_UNSUPPORTED;
  }
  return KVX_STATUS_OK;
}

template <typename T>
static kvx_status_t kvx_launch_write_kv_t(const kvx_cache_desc_t *cache,
                                          const kvx_write_desc_t *write,
                                          void *stream,
                                          kvx_dtype_t expected_dtype) {
  if (cache == NULL || write == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (write->size < sizeof(kvx_write_desc_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (write->io.size < sizeof(kvx_kv_io_desc_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (write->slots.size < sizeof(kvx_slot_mapping_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (cache->k.dtype != expected_dtype || cache->v.dtype != expected_dtype) {
    return KVX_STATUS_UNSUPPORTED;
  }
  if (write->io.key.dtype != expected_dtype ||
      write->io.value.dtype != expected_dtype) {
    return KVX_STATUS_UNSUPPORTED;
  }
  if ((int32_t)write->io.num_kv_heads != (int32_t)cache->num_kv_heads ||
      (int32_t)write->io.head_dim != (int32_t)cache->head_dim) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }

  kvx_status_t status = kvx_validate_io_tensor_3d(
      &write->io.key, (int32_t)write->io.num_tokens,
      (int32_t)write->io.num_kv_heads, (int32_t)write->io.head_dim);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  status = kvx_validate_io_tensor_3d(&write->io.value,
                                     (int32_t)write->io.num_tokens,
                                     (int32_t)write->io.num_kv_heads,
                                     (int32_t)write->io.head_dim);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  if (write->slots.token_count != write->io.num_tokens) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }

  kvx_cache_params_t k_params;
  kvx_cache_params_t v_params;
  status = kvx_fill_cache_params(cache, &cache->k, &k_params);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  status = kvx_fill_cache_params(cache, &cache->v, &v_params);
  if (status != KVX_STATUS_OK) {
    return status;
  }

  int64_t total = (int64_t)write->io.num_tokens * k_params.num_heads *
                  k_params.head_dim;
  if (total == 0) {
    return KVX_STATUS_OK;
  }
  dim3 block(256);
  dim3 grid((unsigned int)((total + block.x - 1) / block.x));
  cudaStream_t cuda_stream = (stream == NULL)
                                 ? static_cast<cudaStream_t>(0)
                                 : reinterpret_cast<cudaStream_t>(stream);

  constexpr int kVecBytes = 16;
  constexpr int kVecElems = kVecBytes / sizeof(T);
  bool can_vectorize =
      kvx_is_nhd_contiguous(k_params) && kvx_is_nhd_contiguous(v_params) &&
      (k_params.num_heads * k_params.head_dim) % kVecElems == 0 &&
      (k_params.stride[0] * (int64_t)sizeof(T)) % kVecBytes == 0 &&
      (k_params.stride[1] * (int64_t)sizeof(T)) % kVecBytes == 0 &&
      (v_params.stride[0] * (int64_t)sizeof(T)) % kVecBytes == 0 &&
      (v_params.stride[1] * (int64_t)sizeof(T)) % kVecBytes == 0 &&
      (write->io.key.stride[0] * (int64_t)sizeof(T)) % kVecBytes == 0 &&
      (write->io.value.stride[0] * (int64_t)sizeof(T)) % kVecBytes == 0 &&
      kvx_is_aligned_16(write->io.key.data) &&
      kvx_is_aligned_16(write->io.value.data) &&
      kvx_is_aligned_16(cache->k.data) && kvx_is_aligned_16(cache->v.data);

  if (write->slots.dtype == KVX_DTYPE_S32) {
    if (can_vectorize) {
      kvx_write_kv_kernel_nhd_vec<T, int32_t, kVecElems><<<
          dim3(write->io.num_tokens), block, 0, cuda_stream>>>(
          reinterpret_cast<const T *>(write->io.key.data),
          reinterpret_cast<const T *>(write->io.value.data),
          reinterpret_cast<T *>(cache->k.data),
          reinterpret_cast<T *>(cache->v.data),
          reinterpret_cast<const int32_t *>(write->slots.slots),
          (int32_t)write->io.num_tokens, k_params.block_size,
          k_params.num_blocks, k_params.num_heads, k_params.head_dim,
          k_params.stride[0], k_params.stride[1], write->io.key.stride[0],
          write->io.value.stride[0]);
    } else {
      kvx_write_kv_kernel<T, int32_t><<<grid, block, 0, cuda_stream>>>(
          reinterpret_cast<const T *>(write->io.key.data),
          reinterpret_cast<const T *>(write->io.value.data),
          reinterpret_cast<T *>(cache->k.data),
          reinterpret_cast<T *>(cache->v.data),
          reinterpret_cast<const int32_t *>(write->slots.slots),
          (int32_t)write->io.num_tokens, k_params, v_params);
    }
  } else if (write->slots.dtype == KVX_DTYPE_S64) {
    if (can_vectorize) {
      kvx_write_kv_kernel_nhd_vec<T, int64_t, kVecElems><<<
          dim3(write->io.num_tokens), block, 0, cuda_stream>>>(
          reinterpret_cast<const T *>(write->io.key.data),
          reinterpret_cast<const T *>(write->io.value.data),
          reinterpret_cast<T *>(cache->k.data),
          reinterpret_cast<T *>(cache->v.data),
          reinterpret_cast<const int64_t *>(write->slots.slots),
          (int32_t)write->io.num_tokens, k_params.block_size,
          k_params.num_blocks, k_params.num_heads, k_params.head_dim,
          k_params.stride[0], k_params.stride[1], write->io.key.stride[0],
          write->io.value.stride[0]);
    } else {
      kvx_write_kv_kernel<T, int64_t><<<grid, block, 0, cuda_stream>>>(
          reinterpret_cast<const T *>(write->io.key.data),
          reinterpret_cast<const T *>(write->io.value.data),
          reinterpret_cast<T *>(cache->k.data),
          reinterpret_cast<T *>(cache->v.data),
          reinterpret_cast<const int64_t *>(write->slots.slots),
          (int32_t)write->io.num_tokens, k_params, v_params);
    }
  } else {
    return KVX_STATUS_INVALID_ARGUMENT;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return KVX_STATUS_INTERNAL_ERROR;
  }
  return KVX_STATUS_OK;
}

KVX_API kvx_status_t kvx_launch_write_kv_f32(const kvx_cache_desc_t *cache,
                                             const kvx_write_desc_t *write,
                                             void *stream) {
  return kvx_launch_write_kv_t<float>(cache, write, stream, KVX_DTYPE_F32);
}

KVX_API kvx_status_t kvx_launch_write_kv_f16(const kvx_cache_desc_t *cache,
                                             const kvx_write_desc_t *write,
                                             void *stream) {
  return kvx_launch_write_kv_t<__half>(cache, write, stream, KVX_DTYPE_F16);
}

KVX_API kvx_status_t kvx_launch_write_kv_bf16(const kvx_cache_desc_t *cache,
                                              const kvx_write_desc_t *write,
                                              void *stream) {
  return kvx_launch_write_kv_t<__nv_bfloat16>(cache, write, stream,
                                              KVX_DTYPE_BF16);
}

KVX_API kvx_status_t kvx_launch_write_kv(const kvx_cache_desc_t *cache,
                                         const kvx_write_desc_t *write,
                                         void *stream) {
  if (cache == NULL || write == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (cache->k.dtype != cache->v.dtype) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  switch (cache->k.dtype) {
    case KVX_DTYPE_F16:
      return kvx_launch_write_kv_f16(cache, write, stream);
    case KVX_DTYPE_BF16:
      return kvx_launch_write_kv_bf16(cache, write, stream);
    case KVX_DTYPE_F32:
      return kvx_launch_write_kv_f32(cache, write, stream);
    default:
      return KVX_STATUS_UNSUPPORTED;
  }
}

template <typename T>
static kvx_status_t kvx_launch_write_kv_prefill_t(
    const kvx_cache_desc_t *cache, const kvx_write_desc_t *write,
    int32_t tokens_per_seq, void *stream, kvx_dtype_t expected_dtype) {
  if (cache == NULL || write == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tokens_per_seq <= 0) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (write->size < sizeof(kvx_write_desc_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (write->io.size < sizeof(kvx_kv_io_desc_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (write->slots.size < sizeof(kvx_slot_mapping_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if ((write->io.num_tokens % tokens_per_seq) != 0) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (cache->k.dtype != expected_dtype || cache->v.dtype != expected_dtype) {
    return KVX_STATUS_UNSUPPORTED;
  }
  if (write->io.key.dtype != expected_dtype ||
      write->io.value.dtype != expected_dtype) {
    return KVX_STATUS_UNSUPPORTED;
  }
  if ((int32_t)write->io.num_kv_heads != (int32_t)cache->num_kv_heads ||
      (int32_t)write->io.head_dim != (int32_t)cache->head_dim) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }

  kvx_status_t status = kvx_validate_io_tensor_3d(
      &write->io.key, (int32_t)write->io.num_tokens,
      (int32_t)write->io.num_kv_heads, (int32_t)write->io.head_dim);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  status = kvx_validate_io_tensor_3d(&write->io.value,
                                     (int32_t)write->io.num_tokens,
                                     (int32_t)write->io.num_kv_heads,
                                     (int32_t)write->io.head_dim);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  if (write->slots.token_count != write->io.num_tokens) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }

  kvx_cache_params_t k_params;
  kvx_cache_params_t v_params;
  status = kvx_fill_cache_params(cache, &cache->k, &k_params);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  status = kvx_fill_cache_params(cache, &cache->v, &v_params);
  if (status != KVX_STATUS_OK) {
    return status;
  }

  int64_t total = (int64_t)write->io.num_tokens * k_params.num_heads *
                  k_params.head_dim;
  if (total == 0) {
    return KVX_STATUS_OK;
  }

  dim3 block(256);
  cudaStream_t cuda_stream = (stream == NULL)
                                 ? static_cast<cudaStream_t>(0)
                                 : reinterpret_cast<cudaStream_t>(stream);

  const int64_t elem_size = (int64_t)sizeof(T);
  auto can_vectorize_bytes = [&](int vec_bytes) {
    int vec_elems = vec_bytes / (int)elem_size;
    if ((k_params.num_heads * k_params.head_dim) % vec_elems != 0) {
      return false;
    }
    if (!kvx_is_nhd_contiguous(k_params) ||
        !kvx_is_nhd_contiguous(v_params)) {
      return false;
    }
    if ((k_params.stride[0] * elem_size) % vec_bytes != 0 ||
        (k_params.stride[1] * elem_size) % vec_bytes != 0 ||
        (v_params.stride[0] * elem_size) % vec_bytes != 0 ||
        (v_params.stride[1] * elem_size) % vec_bytes != 0 ||
        (write->io.key.stride[0] * elem_size) % vec_bytes != 0 ||
        (write->io.value.stride[0] * elem_size) % vec_bytes != 0) {
      return false;
    }
    return kvx_is_aligned(write->io.key.data, (size_t)vec_bytes) &&
           kvx_is_aligned(write->io.value.data, (size_t)vec_bytes) &&
           kvx_is_aligned(cache->k.data, (size_t)vec_bytes) &&
           kvx_is_aligned(cache->v.data, (size_t)vec_bytes);
  };

  int vec_bytes = 0;
  if (can_vectorize_bytes(64)) {
    vec_bytes = 64;
  } else if (can_vectorize_bytes(32)) {
    vec_bytes = 32;
  } else if (can_vectorize_bytes(16)) {
    vec_bytes = 16;
  }

  if (write->slots.dtype == KVX_DTYPE_S32) {
    if (vec_bytes != 0) {
      int vec_elems = vec_bytes / (int)elem_size;
      int64_t vec_count =
          (int64_t)k_params.num_heads * k_params.head_dim / vec_elems;
      dim3 grid((unsigned int)write->io.num_tokens,
                (unsigned int)((vec_count + block.x - 1) / block.x));
      if (vec_bytes == 64) {
        kvx_write_kv_kernel_prefill_nhd_vec_bytes<T, int32_t, 64><<<
            grid, block, 0, cuda_stream>>>(
            reinterpret_cast<const T *>(write->io.key.data),
            reinterpret_cast<const T *>(write->io.value.data),
            reinterpret_cast<T *>(cache->k.data),
            reinterpret_cast<T *>(cache->v.data),
            reinterpret_cast<const int32_t *>(write->slots.slots),
            (int32_t)write->io.num_tokens, k_params.block_size,
            k_params.num_blocks, k_params.num_heads, k_params.head_dim,
            k_params.stride[0], k_params.stride[1], write->io.key.stride[0],
            write->io.value.stride[0]);
      } else if (vec_bytes == 32) {
        kvx_write_kv_kernel_prefill_nhd_vec_bytes<T, int32_t, 32><<<
            grid, block, 0, cuda_stream>>>(
            reinterpret_cast<const T *>(write->io.key.data),
            reinterpret_cast<const T *>(write->io.value.data),
            reinterpret_cast<T *>(cache->k.data),
            reinterpret_cast<T *>(cache->v.data),
            reinterpret_cast<const int32_t *>(write->slots.slots),
            (int32_t)write->io.num_tokens, k_params.block_size,
            k_params.num_blocks, k_params.num_heads, k_params.head_dim,
            k_params.stride[0], k_params.stride[1], write->io.key.stride[0],
            write->io.value.stride[0]);
      } else {
        kvx_write_kv_kernel_prefill_nhd_vec_bytes<T, int32_t, 16><<<
            grid, block, 0, cuda_stream>>>(
            reinterpret_cast<const T *>(write->io.key.data),
            reinterpret_cast<const T *>(write->io.value.data),
            reinterpret_cast<T *>(cache->k.data),
            reinterpret_cast<T *>(cache->v.data),
            reinterpret_cast<const int32_t *>(write->slots.slots),
            (int32_t)write->io.num_tokens, k_params.block_size,
            k_params.num_blocks, k_params.num_heads, k_params.head_dim,
            k_params.stride[0], k_params.stride[1], write->io.key.stride[0],
            write->io.value.stride[0]);
      }
    } else {
      dim3 grid((unsigned int)((total + block.x - 1) / block.x));
      kvx_write_kv_kernel<T, int32_t><<<grid, block, 0, cuda_stream>>>(
          reinterpret_cast<const T *>(write->io.key.data),
          reinterpret_cast<const T *>(write->io.value.data),
          reinterpret_cast<T *>(cache->k.data),
          reinterpret_cast<T *>(cache->v.data),
          reinterpret_cast<const int32_t *>(write->slots.slots),
          (int32_t)write->io.num_tokens, k_params, v_params);
    }
  } else if (write->slots.dtype == KVX_DTYPE_S64) {
    if (vec_bytes != 0) {
      int vec_elems = vec_bytes / (int)elem_size;
      int64_t vec_count =
          (int64_t)k_params.num_heads * k_params.head_dim / vec_elems;
      dim3 grid((unsigned int)write->io.num_tokens,
                (unsigned int)((vec_count + block.x - 1) / block.x));
      if (vec_bytes == 64) {
        kvx_write_kv_kernel_prefill_nhd_vec_bytes<T, int64_t, 64><<<
            grid, block, 0, cuda_stream>>>(
            reinterpret_cast<const T *>(write->io.key.data),
            reinterpret_cast<const T *>(write->io.value.data),
            reinterpret_cast<T *>(cache->k.data),
            reinterpret_cast<T *>(cache->v.data),
            reinterpret_cast<const int64_t *>(write->slots.slots),
            (int32_t)write->io.num_tokens, k_params.block_size,
            k_params.num_blocks, k_params.num_heads, k_params.head_dim,
            k_params.stride[0], k_params.stride[1], write->io.key.stride[0],
            write->io.value.stride[0]);
      } else if (vec_bytes == 32) {
        kvx_write_kv_kernel_prefill_nhd_vec_bytes<T, int64_t, 32><<<
            grid, block, 0, cuda_stream>>>(
            reinterpret_cast<const T *>(write->io.key.data),
            reinterpret_cast<const T *>(write->io.value.data),
            reinterpret_cast<T *>(cache->k.data),
            reinterpret_cast<T *>(cache->v.data),
            reinterpret_cast<const int64_t *>(write->slots.slots),
            (int32_t)write->io.num_tokens, k_params.block_size,
            k_params.num_blocks, k_params.num_heads, k_params.head_dim,
            k_params.stride[0], k_params.stride[1], write->io.key.stride[0],
            write->io.value.stride[0]);
      } else {
        kvx_write_kv_kernel_prefill_nhd_vec_bytes<T, int64_t, 16><<<
            grid, block, 0, cuda_stream>>>(
            reinterpret_cast<const T *>(write->io.key.data),
            reinterpret_cast<const T *>(write->io.value.data),
            reinterpret_cast<T *>(cache->k.data),
            reinterpret_cast<T *>(cache->v.data),
            reinterpret_cast<const int64_t *>(write->slots.slots),
            (int32_t)write->io.num_tokens, k_params.block_size,
            k_params.num_blocks, k_params.num_heads, k_params.head_dim,
            k_params.stride[0], k_params.stride[1], write->io.key.stride[0],
            write->io.value.stride[0]);
      }
    } else {
      dim3 grid((unsigned int)((total + block.x - 1) / block.x));
      kvx_write_kv_kernel<T, int64_t><<<grid, block, 0, cuda_stream>>>(
          reinterpret_cast<const T *>(write->io.key.data),
          reinterpret_cast<const T *>(write->io.value.data),
          reinterpret_cast<T *>(cache->k.data),
          reinterpret_cast<T *>(cache->v.data),
          reinterpret_cast<const int64_t *>(write->slots.slots),
          (int32_t)write->io.num_tokens, k_params, v_params);
    }
  } else {
    return KVX_STATUS_INVALID_ARGUMENT;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return KVX_STATUS_INTERNAL_ERROR;
  }
  return KVX_STATUS_OK;
}

KVX_API kvx_status_t kvx_launch_write_kv_prefill_f32(
    const kvx_cache_desc_t *cache, const kvx_write_desc_t *write,
    int32_t tokens_per_seq, void *stream) {
  return kvx_launch_write_kv_prefill_t<float>(
      cache, write, tokens_per_seq, stream, KVX_DTYPE_F32);
}

KVX_API kvx_status_t kvx_launch_write_kv_prefill_f16(
    const kvx_cache_desc_t *cache, const kvx_write_desc_t *write,
    int32_t tokens_per_seq, void *stream) {
  return kvx_launch_write_kv_prefill_t<__half>(
      cache, write, tokens_per_seq, stream, KVX_DTYPE_F16);
}

KVX_API kvx_status_t kvx_launch_write_kv_prefill_bf16(
    const kvx_cache_desc_t *cache, const kvx_write_desc_t *write,
    int32_t tokens_per_seq, void *stream) {
  return kvx_launch_write_kv_prefill_t<__nv_bfloat16>(
      cache, write, tokens_per_seq, stream, KVX_DTYPE_BF16);
}

KVX_API kvx_status_t kvx_launch_write_kv_prefill(
    const kvx_cache_desc_t *cache, const kvx_write_desc_t *write,
    int32_t tokens_per_seq, void *stream) {
  if (cache == NULL || write == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (cache->k.dtype != cache->v.dtype) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  switch (cache->k.dtype) {
    case KVX_DTYPE_F16:
      return kvx_launch_write_kv_prefill_f16(cache, write, tokens_per_seq,
                                             stream);
    case KVX_DTYPE_BF16:
      return kvx_launch_write_kv_prefill_bf16(cache, write, tokens_per_seq,
                                              stream);
    case KVX_DTYPE_F32:
      return kvx_launch_write_kv_prefill_f32(cache, write, tokens_per_seq,
                                             stream);
    default:
      return KVX_STATUS_UNSUPPORTED;
  }
}

template <typename T>
static kvx_status_t kvx_launch_gather_kv_t(const kvx_cache_desc_t *cache,
                                           const kvx_gather_desc_t *gather,
                                           void *stream,
                                           kvx_dtype_t expected_dtype) {
  if (cache == NULL || gather == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (gather->size < sizeof(kvx_gather_desc_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (gather->io.size < sizeof(kvx_kv_io_desc_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (gather->block_table.size < sizeof(kvx_block_table_t) ||
      gather->seq_lens.size < sizeof(kvx_seq_lens_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (cache->k.dtype != expected_dtype || cache->v.dtype != expected_dtype) {
    return KVX_STATUS_UNSUPPORTED;
  }
  if (gather->io.key.dtype != expected_dtype ||
      gather->io.value.dtype != expected_dtype) {
    return KVX_STATUS_UNSUPPORTED;
  }
  if (gather->block_table.format != KVX_BLOCK_TABLE_PACKED) {
    return KVX_STATUS_UNSUPPORTED;
  }
  if ((int32_t)gather->io.num_kv_heads != (int32_t)cache->num_kv_heads ||
      (int32_t)gather->io.head_dim != (int32_t)cache->head_dim) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (gather->seq_lens.seq_count != gather->block_table.seq_count) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }

  kvx_status_t status = kvx_validate_io_tensor_4d(
      &gather->io.key, (int32_t)gather->block_table.seq_count,
      (int32_t)gather->max_seq_len, (int32_t)gather->io.num_kv_heads,
      (int32_t)gather->io.head_dim);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  status = kvx_validate_io_tensor_4d(
      &gather->io.value, (int32_t)gather->block_table.seq_count,
      (int32_t)gather->max_seq_len, (int32_t)gather->io.num_kv_heads,
      (int32_t)gather->io.head_dim);
  if (status != KVX_STATUS_OK) {
    return status;
  }

  kvx_cache_params_t k_params;
  kvx_cache_params_t v_params;
  status = kvx_fill_cache_params(cache, &cache->k, &k_params);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  status = kvx_fill_cache_params(cache, &cache->v, &v_params);
  if (status != KVX_STATUS_OK) {
    return status;
  }

  int64_t total = (int64_t)gather->block_table.seq_count *
                  gather->max_seq_len * k_params.num_heads * k_params.head_dim;
  if (total == 0) {
    return KVX_STATUS_OK;
  }

  dim3 block(256);
  dim3 grid((unsigned int)((total + block.x - 1) / block.x));
  cudaStream_t cuda_stream = (stream == NULL)
                                 ? static_cast<cudaStream_t>(0)
                                 : reinterpret_cast<cudaStream_t>(stream);

  if (gather->block_table.index_dtype == KVX_DTYPE_S32 &&
      gather->seq_lens.dtype == KVX_DTYPE_S32) {
    kvx_gather_kv_kernel<T, int32_t, int32_t><<<grid, block, 0, cuda_stream>>>(
        reinterpret_cast<const T *>(cache->k.data),
        reinterpret_cast<const T *>(cache->v.data),
        reinterpret_cast<T *>(gather->io.key.data),
        reinterpret_cast<T *>(gather->io.value.data),
        reinterpret_cast<const int32_t *>(gather->block_table.indices),
        reinterpret_cast<const int32_t *>(gather->seq_lens.lengths),
        (int32_t)gather->block_table.seq_count,
        (int32_t)gather->block_table.max_blocks_per_seq,
        (int32_t)gather->max_seq_len, k_params, v_params);
  } else if (gather->block_table.index_dtype == KVX_DTYPE_S64 &&
             gather->seq_lens.dtype == KVX_DTYPE_S64) {
    kvx_gather_kv_kernel<T, int64_t, int64_t><<<grid, block, 0, cuda_stream>>>(
        reinterpret_cast<const T *>(cache->k.data),
        reinterpret_cast<const T *>(cache->v.data),
        reinterpret_cast<T *>(gather->io.key.data),
        reinterpret_cast<T *>(gather->io.value.data),
        reinterpret_cast<const int64_t *>(gather->block_table.indices),
        reinterpret_cast<const int64_t *>(gather->seq_lens.lengths),
        (int32_t)gather->block_table.seq_count,
        (int32_t)gather->block_table.max_blocks_per_seq,
        (int32_t)gather->max_seq_len, k_params, v_params);
  } else {
    return KVX_STATUS_INVALID_ARGUMENT;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return KVX_STATUS_INTERNAL_ERROR;
  }
  return KVX_STATUS_OK;
}

KVX_API kvx_status_t kvx_launch_gather_kv_f32(const kvx_cache_desc_t *cache,
                                              const kvx_gather_desc_t *gather,
                                              void *stream) {
  return kvx_launch_gather_kv_t<float>(cache, gather, stream, KVX_DTYPE_F32);
}

KVX_API kvx_status_t kvx_launch_gather_kv_f16(const kvx_cache_desc_t *cache,
                                              const kvx_gather_desc_t *gather,
                                              void *stream) {
  return kvx_launch_gather_kv_t<__half>(cache, gather, stream, KVX_DTYPE_F16);
}

KVX_API kvx_status_t kvx_launch_gather_kv_bf16(const kvx_cache_desc_t *cache,
                                               const kvx_gather_desc_t *gather,
                                               void *stream) {
  return kvx_launch_gather_kv_t<__nv_bfloat16>(cache, gather, stream,
                                               KVX_DTYPE_BF16);
}

KVX_API kvx_status_t kvx_launch_gather_kv(const kvx_cache_desc_t *cache,
                                          const kvx_gather_desc_t *gather,
                                          void *stream) {
  if (cache == NULL || gather == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (cache->k.dtype != cache->v.dtype) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  switch (cache->k.dtype) {
    case KVX_DTYPE_F16:
      return kvx_launch_gather_kv_f16(cache, gather, stream);
    case KVX_DTYPE_BF16:
      return kvx_launch_gather_kv_bf16(cache, gather, stream);
    case KVX_DTYPE_F32:
      return kvx_launch_gather_kv_f32(cache, gather, stream);
    default:
      return KVX_STATUS_UNSUPPORTED;
  }
}
