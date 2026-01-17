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

static int kvx_is_index_dtype(kvx_dtype_t dtype) {
  return dtype == KVX_DTYPE_S32 || dtype == KVX_DTYPE_S64;
}

static int kvx_is_power_of_two(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static kvx_status_t kvx_validate_io_tensor_3d(const kvx_tensor_desc_t *tensor,
                                              uint32_t num_tokens,
                                              uint32_t num_heads,
                                              uint32_t head_dim) {
  if (tensor == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->size < sizeof(kvx_tensor_desc_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (!kvx_is_float_dtype(tensor->dtype)) {
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
  if (num_tokens > 0 && tensor->data == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  return KVX_STATUS_OK;
}

static kvx_status_t kvx_validate_io_tensor_4d(const kvx_tensor_desc_t *tensor,
                                              uint32_t seq_count,
                                              uint32_t max_seq_len,
                                              uint32_t num_heads,
                                              uint32_t head_dim) {
  if (tensor == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (tensor->size < sizeof(kvx_tensor_desc_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (!kvx_is_float_dtype(tensor->dtype)) {
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
  if (seq_count > 0 && max_seq_len > 0 && tensor->data == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  return KVX_STATUS_OK;
}

static kvx_status_t kvx_validate_slot_mapping(const kvx_slot_mapping_t *slots,
                                              uint32_t num_tokens) {
  if (slots == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (slots->size < sizeof(kvx_slot_mapping_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (!kvx_is_index_dtype(slots->dtype)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (slots->token_count != num_tokens) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (num_tokens > 0 && slots->slots == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  return KVX_STATUS_OK;
}

static kvx_status_t kvx_validate_block_table(
    const kvx_cache_desc_t *cache, const kvx_block_table_t *table) {
  if (table == NULL || cache == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (table->size < sizeof(kvx_block_table_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (table->seq_count == 0 || table->max_blocks_per_seq == 0) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }

  switch (table->format) {
    case KVX_BLOCK_TABLE_PACKED: {
      uint32_t expected = table->seq_count * table->max_blocks_per_seq;
      if (!kvx_is_index_dtype(table->index_dtype)) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->beam_width != 1) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->indices_count != expected) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->indptr != NULL || table->indptr_count != 0) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->indices_count > 0 && table->indices == NULL) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      return KVX_STATUS_OK;
    }
    case KVX_BLOCK_TABLE_RAGGED: {
      if (!kvx_is_index_dtype(table->index_dtype) ||
          !kvx_is_index_dtype(table->indptr_dtype)) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->beam_width != 1) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->indptr == NULL) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->indptr_count != table->seq_count + 1) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->indices_count > 0 && table->indices == NULL) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      return KVX_STATUS_OK;
    }
    case KVX_BLOCK_TABLE_KV_OFFSETS: {
      uint32_t expected =
          table->seq_count * table->beam_width * 2 * table->max_blocks_per_seq;
      if (table->index_dtype != KVX_DTYPE_S32) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if ((table->flags & KVX_BLOCK_TABLE_FLAG_KVCACHEINDEX) == 0) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->beam_width == 0) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (!kvx_is_power_of_two(cache->block_size)) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->indices_count != expected) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->indptr != NULL || table->indptr_count != 0) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (table->indices_count > 0 && table->indices == NULL) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      if (cache->pool.bytes_per_block == 0 || cache->pool.primary == NULL) {
        return KVX_STATUS_INVALID_ARGUMENT;
      }
      return KVX_STATUS_OK;
    }
    case KVX_BLOCK_TABLE_INVALID:
    default:
      return KVX_STATUS_INVALID_ARGUMENT;
  }
}

static kvx_status_t kvx_validate_seq_lens(const kvx_seq_lens_t *seq_lens,
                                          uint32_t seq_count) {
  if (seq_lens == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (seq_lens->size < sizeof(kvx_seq_lens_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (!kvx_is_index_dtype(seq_lens->dtype)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (seq_lens->seq_count != seq_count) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (seq_count > 0 && seq_lens->lengths == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  return KVX_STATUS_OK;
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
  kvx_status_t status;
  if (cache == NULL || write == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (write->size < sizeof(kvx_write_desc_t) ||
      write->io.size < sizeof(kvx_kv_io_desc_t) ||
      write->slots.size < sizeof(kvx_slot_mapping_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  status = kvx_validate_cache_desc(cache);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  if (write->io.num_kv_heads == 0 || write->io.head_dim == 0) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (write->io.num_kv_heads != cache->num_kv_heads ||
      write->io.head_dim != cache->head_dim) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (write->io.key.dtype != cache->k.dtype ||
      write->io.value.dtype != cache->v.dtype) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  status = kvx_validate_io_tensor_3d(&write->io.key, write->io.num_tokens,
                                     write->io.num_kv_heads,
                                     write->io.head_dim);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  status = kvx_validate_io_tensor_3d(&write->io.value, write->io.num_tokens,
                                     write->io.num_kv_heads,
                                     write->io.head_dim);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  status = kvx_validate_slot_mapping(&write->slots, write->io.num_tokens);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  return KVX_STATUS_OK;
}

KVX_API kvx_status_t kvx_gather_kv(const kvx_cache_desc_t *cache,
                                   const kvx_gather_desc_t *gather,
                                   void *stream) {
  (void)stream;
  kvx_status_t status;
  if (cache == NULL || gather == NULL) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (gather->size < sizeof(kvx_gather_desc_t) ||
      gather->io.size < sizeof(kvx_kv_io_desc_t) ||
      gather->block_table.size < sizeof(kvx_block_table_t) ||
      gather->seq_lens.size < sizeof(kvx_seq_lens_t)) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  status = kvx_validate_cache_desc(cache);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  if (gather->io.num_kv_heads == 0 || gather->io.head_dim == 0 ||
      gather->max_seq_len == 0) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (gather->io.num_kv_heads != cache->num_kv_heads ||
      gather->io.head_dim != cache->head_dim) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (gather->io.key.dtype != cache->k.dtype ||
      gather->io.value.dtype != cache->v.dtype) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  status = kvx_validate_block_table(cache, &gather->block_table);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  status =
      kvx_validate_seq_lens(&gather->seq_lens, gather->block_table.seq_count);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  if (gather->block_table.index_dtype != gather->seq_lens.dtype) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  if (gather->io.num_tokens !=
      gather->block_table.seq_count * gather->max_seq_len) {
    return KVX_STATUS_INVALID_ARGUMENT;
  }
  status = kvx_validate_io_tensor_4d(&gather->io.key,
                                     gather->block_table.seq_count,
                                     gather->max_seq_len,
                                     gather->io.num_kv_heads,
                                     gather->io.head_dim);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  status = kvx_validate_io_tensor_4d(&gather->io.value,
                                     gather->block_table.seq_count,
                                     gather->max_seq_len,
                                     gather->io.num_kv_heads,
                                     gather->io.head_dim);
  if (status != KVX_STATUS_OK) {
    return status;
  }
  if (gather->block_table.format != KVX_BLOCK_TABLE_PACKED) {
    return KVX_STATUS_UNSUPPORTED;
  }
  return KVX_STATUS_OK;
}
