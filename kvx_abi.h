#ifndef KVX_ABI_H
#define KVX_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KVX_ABI_VERSION_MAJOR 1
#define KVX_ABI_VERSION_MINOR 1

#if defined(_WIN32)
#if defined(KVX_BUILD_DLL)
#define KVX_API __declspec(dllexport)
#elif defined(KVX_USE_DLL)
#define KVX_API __declspec(dllimport)
#else
#define KVX_API
#endif
#else
#define KVX_API
#endif

typedef enum kvx_status {
  KVX_STATUS_OK = 0,
  KVX_STATUS_INVALID_ARGUMENT = 1,
  KVX_STATUS_UNSUPPORTED = 2,
  KVX_STATUS_OUT_OF_RANGE = 3,
  KVX_STATUS_INCOMPATIBLE = 4,
  KVX_STATUS_INTERNAL_ERROR = 5
} kvx_status_t;

typedef struct kvx_version {
  uint32_t size;
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t abi_patch;
} kvx_version_t;

typedef enum kvx_dtype {
  KVX_DTYPE_INVALID = 0,
  KVX_DTYPE_F16 = 1,
  KVX_DTYPE_BF16 = 2,
  KVX_DTYPE_F32 = 3,
  KVX_DTYPE_F8_E4M3 = 4,
  KVX_DTYPE_F8_E5M2 = 5,
  KVX_DTYPE_S32 = 6,
  KVX_DTYPE_S64 = 7,
  KVX_DTYPE_U32 = 8,
  KVX_DTYPE_U64 = 9,
  KVX_DTYPE_U8 = 10
} kvx_dtype_t;

typedef enum kvx_memory_type {
  KVX_MEMORY_UNKNOWN = 0,
  KVX_MEMORY_HOST = 1,
  KVX_MEMORY_DEVICE = 2,
  KVX_MEMORY_UNIFIED = 3
} kvx_memory_type_t;

typedef enum kvx_layout {
  KVX_LAYOUT_INVALID = 0,
  KVX_LAYOUT_BLOCK_NHD = 1,
  KVX_LAYOUT_BLOCK_HND = 2,
  KVX_LAYOUT_BLOCK_HND_PACKED = 3,
  KVX_LAYOUT_BLOCK_CUSTOM = 15
} kvx_layout_t;

enum { KVX_MAX_TENSOR_DIMS = 5 };

typedef struct kvx_tensor_desc {
  uint32_t size;
  kvx_dtype_t dtype;
  kvx_layout_t layout;
  kvx_memory_type_t memory;
  uint32_t ndim;
  int64_t shape[KVX_MAX_TENSOR_DIMS];
  int64_t stride[KVX_MAX_TENSOR_DIMS];
  void *data;
} kvx_tensor_desc_t;

typedef struct kvx_pool_desc {
  uint32_t size;
  kvx_memory_type_t memory;
  uint32_t bytes_per_block;
  void *primary;
  void *secondary;
} kvx_pool_desc_t;

typedef struct kvx_cache_desc {
  uint32_t size;
  uint32_t num_blocks;
  uint32_t block_size;
  uint32_t num_kv_heads;
  uint32_t head_dim;
  kvx_tensor_desc_t k;
  kvx_tensor_desc_t v;
  kvx_pool_desc_t pool;
} kvx_cache_desc_t;

typedef enum kvx_block_table_format {
  KVX_BLOCK_TABLE_INVALID = 0,
  KVX_BLOCK_TABLE_PACKED = 1,
  KVX_BLOCK_TABLE_RAGGED = 2,
  KVX_BLOCK_TABLE_KV_OFFSETS = 3
} kvx_block_table_format_t;

typedef enum kvx_block_table_flags {
  KVX_BLOCK_TABLE_FLAG_NONE = 0,
  KVX_BLOCK_TABLE_FLAG_KVCACHEINDEX = 1 << 0
} kvx_block_table_flags_t;

typedef struct kvx_block_table {
  uint32_t size;
  kvx_block_table_format_t format;
  kvx_dtype_t index_dtype;
  kvx_dtype_t indptr_dtype;
  uint32_t seq_count;
  uint32_t beam_width;
  uint32_t max_blocks_per_seq;
  const void *indices;
  const void *indptr;
  uint32_t indices_count;
  uint32_t indptr_count;
  uint32_t flags;
} kvx_block_table_t;

typedef struct kvx_slot_mapping {
  uint32_t size;
  kvx_dtype_t dtype;
  uint32_t token_count;
  int64_t invalid_slot;
  const void *slots;
} kvx_slot_mapping_t;

typedef struct kvx_seq_lens {
  uint32_t size;
  kvx_dtype_t dtype;
  uint32_t seq_count;
  const void *lengths;
} kvx_seq_lens_t;

typedef struct kvx_kv_io_desc {
  uint32_t size;
  kvx_tensor_desc_t key;
  kvx_tensor_desc_t value;
  uint32_t num_tokens;
  uint32_t num_kv_heads;
  uint32_t head_dim;
} kvx_kv_io_desc_t;

typedef enum kvx_scale_granularity {
  KVX_SCALE_NONE = 0,
  KVX_SCALE_PER_TENSOR = 1,
  KVX_SCALE_PER_BLOCK = 2,
  KVX_SCALE_PER_HEAD = 3,
  KVX_SCALE_PER_BLOCK_HEAD = 4
} kvx_scale_granularity_t;

typedef struct kvx_scale_desc {
  uint32_t size;
  kvx_dtype_t dtype;
  kvx_scale_granularity_t granularity;
  uint32_t ndim;
  int64_t shape[KVX_MAX_TENSOR_DIMS];
  int64_t stride[KVX_MAX_TENSOR_DIMS];
  const void *data;
} kvx_scale_desc_t;

typedef struct kvx_write_desc {
  uint32_t size;
  kvx_kv_io_desc_t io;
  kvx_slot_mapping_t slots;
  const void *k_scale;
  const void *v_scale;
  kvx_scale_desc_t k_scale_desc;
  kvx_scale_desc_t v_scale_desc;
} kvx_write_desc_t;

typedef struct kvx_gather_desc {
  uint32_t size;
  kvx_kv_io_desc_t io;
  kvx_block_table_t block_table;
  kvx_seq_lens_t seq_lens;
  uint32_t max_seq_len;
} kvx_gather_desc_t;

KVX_API kvx_status_t kvx_get_version(kvx_version_t *out_version);

KVX_API kvx_status_t kvx_validate_cache_desc(const kvx_cache_desc_t *cache);

KVX_API kvx_status_t kvx_write_kv(const kvx_cache_desc_t *cache,
                                  const kvx_write_desc_t *write, void *stream);

KVX_API kvx_status_t kvx_gather_kv(const kvx_cache_desc_t *cache,
                                   const kvx_gather_desc_t *gather,
                                   void *stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // KVX_ABI_H
