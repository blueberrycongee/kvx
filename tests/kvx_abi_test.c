#include "kvx_abi.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

static void set_tensor_desc_nhd(kvx_tensor_desc_t *tensor, kvx_dtype_t dtype,
                                kvx_memory_type_t memory, uint32_t num_blocks,
                                uint32_t block_size, uint32_t num_heads,
                                uint32_t head_dim, void *data) {
  memset(tensor, 0, sizeof(*tensor));
  tensor->size = sizeof(*tensor);
  tensor->dtype = dtype;
  tensor->layout = KVX_LAYOUT_BLOCK_NHD;
  tensor->memory = memory;
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

static void set_tensor_desc_hnd_packed(kvx_tensor_desc_t *tensor,
                                       kvx_dtype_t dtype,
                                       kvx_memory_type_t memory,
                                       uint32_t num_blocks,
                                       uint32_t block_size, uint32_t num_heads,
                                       uint32_t head_dim, uint32_t pack,
                                       void *data) {
  memset(tensor, 0, sizeof(*tensor));
  tensor->size = sizeof(*tensor);
  tensor->dtype = dtype;
  tensor->layout = KVX_LAYOUT_BLOCK_HND_PACKED;
  tensor->memory = memory;
  tensor->ndim = 5;
  tensor->shape[0] = num_blocks;
  tensor->shape[1] = num_heads;
  tensor->shape[2] = (int64_t)(head_dim / pack);
  tensor->shape[3] = block_size;
  tensor->shape[4] = pack;
  tensor->stride[4] = 1;
  tensor->stride[3] = pack;
  tensor->stride[2] = (int64_t)block_size * pack;
  tensor->stride[1] = (int64_t)(head_dim / pack) * block_size * pack;
  tensor->stride[0] = (int64_t)num_heads * (head_dim / pack) * block_size * pack;
  tensor->data = data;
}

static void set_tensor_desc_tokens(kvx_tensor_desc_t *tensor, kvx_dtype_t dtype,
                                   kvx_memory_type_t memory,
                                   uint32_t num_tokens, uint32_t num_heads,
                                   uint32_t head_dim, void *data) {
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

static void set_tensor_desc_seq(kvx_tensor_desc_t *tensor, kvx_dtype_t dtype,
                                kvx_memory_type_t memory, uint32_t seq_count,
                                uint32_t max_seq_len, uint32_t num_heads,
                                uint32_t head_dim, void *data) {
  memset(tensor, 0, sizeof(*tensor));
  tensor->size = sizeof(*tensor);
  tensor->dtype = dtype;
  tensor->layout = KVX_LAYOUT_BLOCK_CUSTOM;
  tensor->memory = memory;
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

static kvx_cache_desc_t make_cache_nhd(void *k_data, void *v_data) {
  const uint32_t num_blocks = 2;
  const uint32_t block_size = 4;
  const uint32_t num_heads = 2;
  const uint32_t head_dim = 8;

  kvx_cache_desc_t cache;
  memset(&cache, 0, sizeof(cache));
  cache.size = sizeof(cache);
  cache.num_blocks = num_blocks;
  cache.block_size = block_size;
  cache.num_kv_heads = num_heads;
  cache.head_dim = head_dim;
  set_tensor_desc_nhd(&cache.k, KVX_DTYPE_F32, KVX_MEMORY_HOST, num_blocks,
                      block_size, num_heads, head_dim, k_data);
  set_tensor_desc_nhd(&cache.v, KVX_DTYPE_F32, KVX_MEMORY_HOST, num_blocks,
                      block_size, num_heads, head_dim, v_data);
  return cache;
}

static kvx_write_desc_t make_write_desc(uint32_t num_tokens, uint32_t num_heads,
                                        uint32_t head_dim, kvx_dtype_t dtype,
                                        void *key_data, void *value_data,
                                        kvx_dtype_t slots_dtype,
                                        void *slots_data) {
  kvx_write_desc_t write;
  memset(&write, 0, sizeof(write));
  write.size = sizeof(write);
  write.io.size = sizeof(write.io);
  write.io.num_tokens = num_tokens;
  write.io.num_kv_heads = num_heads;
  write.io.head_dim = head_dim;
  set_tensor_desc_tokens(&write.io.key, dtype, KVX_MEMORY_HOST, num_tokens,
                         num_heads, head_dim, key_data);
  set_tensor_desc_tokens(&write.io.value, dtype, KVX_MEMORY_HOST, num_tokens,
                         num_heads, head_dim, value_data);
  write.slots.size = sizeof(write.slots);
  write.slots.dtype = slots_dtype;
  write.slots.token_count = num_tokens;
  write.slots.invalid_slot = -1;
  write.slots.slots = slots_data;
  return write;
}

static kvx_block_table_t make_block_table_packed(uint32_t seq_count,
                                                 uint32_t max_blocks_per_seq,
                                                 kvx_dtype_t index_dtype,
                                                 void *indices) {
  kvx_block_table_t table;
  memset(&table, 0, sizeof(table));
  table.size = sizeof(table);
  table.format = KVX_BLOCK_TABLE_PACKED;
  table.index_dtype = index_dtype;
  table.seq_count = seq_count;
  table.beam_width = 1;
  table.max_blocks_per_seq = max_blocks_per_seq;
  table.indices = indices;
  table.indices_count = seq_count * max_blocks_per_seq;
  table.indptr = NULL;
  table.indptr_count = 0;
  table.flags = 0;
  return table;
}

static kvx_seq_lens_t make_seq_lens(uint32_t seq_count, kvx_dtype_t dtype,
                                    void *lengths) {
  kvx_seq_lens_t seq_lens;
  memset(&seq_lens, 0, sizeof(seq_lens));
  seq_lens.size = sizeof(seq_lens);
  seq_lens.dtype = dtype;
  seq_lens.seq_count = seq_count;
  seq_lens.lengths = lengths;
  return seq_lens;
}

static kvx_gather_desc_t make_gather_desc(
    uint32_t seq_count, uint32_t max_seq_len, uint32_t num_heads,
    uint32_t head_dim, kvx_dtype_t dtype, void *key_out, void *value_out,
    kvx_block_table_t table, kvx_seq_lens_t seq_lens) {
  kvx_gather_desc_t gather;
  memset(&gather, 0, sizeof(gather));
  gather.size = sizeof(gather);
  gather.io.size = sizeof(gather.io);
  gather.io.num_tokens = seq_count * max_seq_len;
  gather.io.num_kv_heads = num_heads;
  gather.io.head_dim = head_dim;
  set_tensor_desc_seq(&gather.io.key, dtype, KVX_MEMORY_HOST, seq_count,
                      max_seq_len, num_heads, head_dim, key_out);
  set_tensor_desc_seq(&gather.io.value, dtype, KVX_MEMORY_HOST, seq_count,
                      max_seq_len, num_heads, head_dim, value_out);
  gather.block_table = table;
  gather.seq_lens = seq_lens;
  gather.max_seq_len = max_seq_len;
  return gather;
}

int main(void) {
  kvx_version_t version = {0};
  kvx_status_t status = kvx_get_version(&version);

  assert(status == KVX_STATUS_OK);
  assert(version.abi_major == KVX_ABI_VERSION_MAJOR);
  assert(version.abi_minor == KVX_ABI_VERSION_MINOR);
  assert(version.size >= sizeof(kvx_version_t));

  kvx_block_table_t table = {0};
  kvx_scale_desc_t scale = {0};
  kvx_tensor_desc_t tensor = {0};

  table.format = KVX_BLOCK_TABLE_PACKED;
  table.indices_count = 0;
  table.indptr_count = 0;
  table.flags = 0;

  scale.granularity = KVX_SCALE_NONE;
  scale.dtype = KVX_DTYPE_F32;

  tensor.layout = KVX_LAYOUT_BLOCK_HND_PACKED;

  float k_buf[128] = {0};
  float v_buf[128] = {0};
  kvx_cache_desc_t cache = make_cache_nhd(k_buf, v_buf);

  status = kvx_validate_cache_desc(&cache);
  assert(status == KVX_STATUS_OK);

  cache.block_size = 0;
  status = kvx_validate_cache_desc(&cache);
  assert(status == KVX_STATUS_INVALID_ARGUMENT);

  cache = make_cache_nhd(k_buf, v_buf);
  cache.size = sizeof(cache) - 4;
  status = kvx_validate_cache_desc(&cache);
  assert(status == KVX_STATUS_INVALID_ARGUMENT);

  cache = make_cache_nhd(k_buf, v_buf);
  cache.k.stride[0] += 1;
  status = kvx_validate_cache_desc(&cache);
  assert(status == KVX_STATUS_UNSUPPORTED);

  cache = make_cache_nhd(k_buf, v_buf);
  cache.v.layout = KVX_LAYOUT_BLOCK_HND;
  status = kvx_validate_cache_desc(&cache);
  assert(status == KVX_STATUS_INVALID_ARGUMENT);

  kvx_cache_desc_t packed_cache;
  memset(&packed_cache, 0, sizeof(packed_cache));
  packed_cache.size = sizeof(packed_cache);
  packed_cache.num_blocks = 1;
  packed_cache.block_size = 4;
  packed_cache.num_kv_heads = 2;
  packed_cache.head_dim = 8;
  set_tensor_desc_hnd_packed(&packed_cache.k, KVX_DTYPE_F16, KVX_MEMORY_HOST, 1,
                             4, 2, 8, 3, k_buf);
  set_tensor_desc_hnd_packed(&packed_cache.v, KVX_DTYPE_F16, KVX_MEMORY_HOST, 1,
                             4, 2, 8, 3, v_buf);
  status = kvx_validate_cache_desc(&packed_cache);
  assert(status == KVX_STATUS_INVALID_ARGUMENT);

  cache = make_cache_nhd(k_buf, v_buf);
  float io_key[64] = {0};
  float io_value[64] = {0};
  int64_t slots[4] = {0, 1, 2, 3};
  kvx_write_desc_t write = make_write_desc(
      4, cache.num_kv_heads, cache.head_dim, KVX_DTYPE_F32, io_key, io_value,
      KVX_DTYPE_S64, slots);
  status = kvx_write_kv(&cache, &write, NULL);
  assert(status == KVX_STATUS_OK);

  write.slots.dtype = KVX_DTYPE_U32;
  status = kvx_write_kv(&cache, &write, NULL);
  assert(status == KVX_STATUS_INVALID_ARGUMENT);

  write = make_write_desc(4, cache.num_kv_heads, cache.head_dim, KVX_DTYPE_F32,
                          io_key, io_value, KVX_DTYPE_S64, slots);
  write.slots.token_count = 3;
  status = kvx_write_kv(&cache, &write, NULL);
  assert(status == KVX_STATUS_INVALID_ARGUMENT);

  write = make_write_desc(4, cache.num_kv_heads, cache.head_dim, KVX_DTYPE_F32,
                          io_key, io_value, KVX_DTYPE_S64, NULL);
  status = kvx_write_kv(&cache, &write, NULL);
  assert(status == KVX_STATUS_INVALID_ARGUMENT);

  write = make_write_desc(4, cache.num_kv_heads, cache.head_dim, KVX_DTYPE_F32,
                          io_key, io_value, KVX_DTYPE_S64, slots);
  write.io.key.size = sizeof(write.io.key) - 4;
  status = kvx_write_kv(&cache, &write, NULL);
  assert(status == KVX_STATUS_INVALID_ARGUMENT);

  int32_t block_table[2] = {0, 1};
  int32_t seq_lens_buf[2] = {4, 4};
  float gather_key[2 * 4 * 2 * 8] = {0};
  float gather_value[2 * 4 * 2 * 8] = {0};
  kvx_block_table_t gather_table =
      make_block_table_packed(2, 1, KVX_DTYPE_S32, block_table);
  kvx_seq_lens_t seq_lens = make_seq_lens(2, KVX_DTYPE_S32, seq_lens_buf);
  kvx_gather_desc_t gather = make_gather_desc(
      2, 4, cache.num_kv_heads, cache.head_dim, KVX_DTYPE_F32, gather_key,
      gather_value, gather_table, seq_lens);
  status = kvx_gather_kv(&cache, &gather, NULL);
  assert(status == KVX_STATUS_OK);

  gather.block_table.indices_count = 1;
  status = kvx_gather_kv(&cache, &gather, NULL);
  assert(status == KVX_STATUS_INVALID_ARGUMENT);

  gather = make_gather_desc(2, 4, cache.num_kv_heads, cache.head_dim,
                            KVX_DTYPE_F32, gather_key, gather_value,
                            gather_table, seq_lens);
  gather.seq_lens.seq_count = 1;
  status = kvx_gather_kv(&cache, &gather, NULL);
  assert(status == KVX_STATUS_INVALID_ARGUMENT);

  gather = make_gather_desc(2, 4, cache.num_kv_heads, cache.head_dim,
                            KVX_DTYPE_F32, gather_key, gather_value,
                            gather_table, seq_lens);
  gather.block_table.index_dtype = KVX_DTYPE_U32;
  status = kvx_gather_kv(&cache, &gather, NULL);
  assert(status == KVX_STATUS_INVALID_ARGUMENT);

  return 0;
}
