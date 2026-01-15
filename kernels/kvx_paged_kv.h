#ifndef KVX_PAGED_KV_H
#define KVX_PAGED_KV_H

#include "kvx_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

KVX_API kvx_status_t kvx_launch_write_kv_f32(const kvx_cache_desc_t *cache,
                                             const kvx_write_desc_t *write,
                                             void *stream);

KVX_API kvx_status_t kvx_launch_write_kv_f16(const kvx_cache_desc_t *cache,
                                             const kvx_write_desc_t *write,
                                             void *stream);

KVX_API kvx_status_t kvx_launch_write_kv_bf16(const kvx_cache_desc_t *cache,
                                              const kvx_write_desc_t *write,
                                              void *stream);

KVX_API kvx_status_t kvx_launch_write_kv_prefill_f32(
    const kvx_cache_desc_t *cache, const kvx_write_desc_t *write,
    int32_t tokens_per_seq, void *stream);

KVX_API kvx_status_t kvx_launch_write_kv_prefill_f16(
    const kvx_cache_desc_t *cache, const kvx_write_desc_t *write,
    int32_t tokens_per_seq, void *stream);

KVX_API kvx_status_t kvx_launch_write_kv_prefill_bf16(
    const kvx_cache_desc_t *cache, const kvx_write_desc_t *write,
    int32_t tokens_per_seq, void *stream);

KVX_API kvx_status_t kvx_launch_gather_kv_f32(const kvx_cache_desc_t *cache,
                                              const kvx_gather_desc_t *gather,
                                              void *stream);

KVX_API kvx_status_t kvx_launch_gather_kv_f16(const kvx_cache_desc_t *cache,
                                              const kvx_gather_desc_t *gather,
                                              void *stream);

KVX_API kvx_status_t kvx_launch_gather_kv_bf16(const kvx_cache_desc_t *cache,
                                               const kvx_gather_desc_t *gather,
                                               void *stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // KVX_PAGED_KV_H
