#include "kvx_abi.h"

#include <string.h>

static int kvx_is_float_dtype(kvx_dtype_t dtype) {
  switch (dtype) {
    case KVX_DTYPE_F16:
    case KVX_DTYPE_BF16:
    case KVX_DTYPE_F32:
    case KVX_DTYPE_F8_E4M3:
    case KVX_DTYPE_F8_E5M2:
      return 1;
    default:
      return 0;
  }
}

static kvx_status_t kvx_validate_tensor_desc(const kvx_cache_desc_t *cache,
                                             const kvx_tensor_desc_t *tensor,
                                             kvx_layout_t layout) {
  if (tensor == NULL || cache == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->size < sizeof(kvx_tensor_desc_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (!kvx_is_float_dtype(tensor->dtype)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->data == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->layout != layout) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->ndim == 0 || tensor->ndim > KVX_MAX_TENSOR_DIMS) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }

  switch (layout) {
    case KVX_LAYOUT_BLOCK_NHD: {
      if (tensor->ndim != 4) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (tensor->shape[0] != cache->num_blocks ||
          tensor->shape[1] != cache->block_size ||
          tensor->shape[2] != cache->num_kv_heads ||
          tensor->shape[3] != cache->head_dim) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (tensor->stride[3] != 1 ||
          tensor->stride[2] != (int64_t)cache->head_dim ||
          tensor->stride[1] !=
              (int64_t)cache->num_kv_heads * cache->head_dim ||
          tensor->stride[0] != (int64_t)cache->block_size *
                                    cache->num_kv_heads * cache->head_dim) {
        return KVX_STATUS_UNSUPPORTED;
      }
      return KVX_STATUS_OK;
    }
    case KVX_LAYOUT_BLOCK_HND: {
      if (tensor->ndim != 4) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (tensor->shape[0] != cache->num_blocks ||
          tensor->shape[1] != cache->num_kv_heads ||
          tensor->shape[2] != cache->block_size ||
          tensor->shape[3] != cache->head_dim) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (tensor->stride[3] != 1 ||
          tensor->stride[2] != (int64_t)cache->head_dim ||
          tensor->stride[1] !=
              (int64_t)cache->block_size * cache->head_dim ||
          tensor->stride[0] != (int64_t)cache->num_kv_heads *
                                    cache->block_size * cache->head_dim) {
        return KVX_STATUS_UNSUPPORTED;
      }
      return KVX_STATUS_OK;
    }
    case KVX_LAYOUT_BLOCK_HND_PACKED: {
      int64_t pack = tensor->shape[4];
      if (tensor->ndim != 5) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (pack <= 0) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if ((cache->head_dim % pack) != 0) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (tensor->shape[0] != cache->num_blocks ||
          tensor->shape[1] != cache->num_kv_heads ||
          tensor->shape[2] != (int64_t)(cache->head_dim / pack) ||
          tensor->shape[3] != cache->block_size || tensor->shape[4] != pack) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (tensor->stride[4] != 1 || tensor->stride[3] != pack ||
          tensor->stride[2] != (int64_t)cache->block_size * pack ||
          tensor->stride[1] != (int64_t)(cache->head_dim / pack) *
                                    cache->block_size * pack ||
          tensor->stride[0] != (int64_t)cache->num_kv_heads *
                                    (cache->head_dim / pack) *
                                    cache->block_size * pack) {
        return KVX_STATUS_UNSUPPORTED;
      }
      return KVX_STATUS_OK;
    }
    case KVX_LAYOUT_BLOCK_CUSTOM:
    default:
      return KVX_STATUS_UNSUPPORTED;
  }
}

KVX_API kvx_status_t kvx_get_version(kvx_version_t *out_version) {
  if (out_version == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  out_version->size = sizeof(kvx_version_t);
  out_version->abi_major = KVX_ABI_VERSION_MAJOR;
  out_version->abi_minor = KVX_ABI_VERSION_MINOR;
  out_version->abi_patch = 0;
  return KVX_STATUS_OK;
}

KVX_API kvx_status_t kvx_validate_cache_desc(const kvx_cache_desc_t *cache) {
  kvx_status_t status;
  if (cache == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (cache->size < sizeof(kvx_cache_desc_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (cache->num_blocks == 0 || cache->block_size == 0 ||
      cache->num_kv_heads == 0 || cache->head_dim == 0) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (cache->k.layout != cache->v.layout) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (cache->k.dtype != cache->v.dtype) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }

  status = kvx_validate_tensor_desc(cache, &cache->k, cache->k.layout);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  status = kvx_validate_tensor_desc(cache, &cache->v, cache->v.layout);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  return KVX_STATUS_OK;
}

KVX_API kvx_status_t kvx_write_kv(const kvx_cache_desc_t *cache,
                                  const kvx_write_desc_t *write, void *stream) {
  (void)stream;
  if (cache == NULL || write == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (write->size < sizeof(kvx_write_desc_t) ||
      write->io.size < sizeof(kvx_kv_io_desc_t) ||
      write->slots.size < sizeof(kvx_slot_mapping_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  return kvx_validate_cache_desc(cache);
}

KVX_API kvx_status_t kvx_gather_kv(const kvx_cache_desc_t *cache,
                                   const kvx_gather_desc_t *gather,
                                   void *stream) {
  (void)stream;
  if (cache == NULL || gather == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (gather->size < sizeof(kvx_gather_desc_t) ||
      gather->io.size < sizeof(kvx_kv_io_desc_t) ||
      gather->block_table.size < sizeof(kvx_block_table_t) ||
      gather->seq_lens.size < sizeof(kvx_seq_lens_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  return kvx_validate_cache_desc(cache);
}
