# KVX v1 Specification (Draft)

KVX defines a GPU-first KV cache layout and metadata standard for high
performance inference. This draft targets paged KV caches and a minimal
C ABI for adapters and kernels.

## 1. Overview

### 1.1 Goals
- Standardize paged KV cache layout and metadata.
- Enable fast adapters between engines (vLLM, TensorRT-LLM, TGI).
- Provide a stable C ABI for kernels and runtime glue.

### 1.2 Non-goals
- Define scheduling, batching, or request management policies.
- Specify model architecture details.
- Provide a full runtime implementation.

## 2. Terminology
- Token: One time step of K/V data for all KV heads.
- Block (page): Fixed-size group of tokens in the KV cache.
- Block size: Tokens per block (page size).
- Slot: Global token slot index: `block_id * block_size + block_offset`.
- Logical block: Per-sequence block index (0..max_blocks_per_seq-1).
- Physical block: Global block id in the cache pool.
- Layout: Tensor shape order and stride scheme for K/V blocks.

## 3. Versioning and Negotiation
- ABI version is `(major, minor, patch)` in `kvx_version_t`.
- `kvx_get_version` MUST populate `kvx_version_t` and set `size`.
- A consumer MUST reject providers with mismatched major versions.
- A consumer MAY accept a newer minor version if all required structs are
  size-guarded and validated via `size` fields.
- A producer SHOULD return `KVX_STATUS_INCOMPATIBLE` when major versions differ.

## 4. Memory Model

### 4.1 Paged KV layout
- KV cache is paged and stored as separate K and V tensors.
- Each block holds `block_size` tokens for all KV heads and head dimensions.
- `kvx_cache_desc_t` defines `num_blocks`, `block_size`, `num_kv_heads`,
  and `head_dim` for the cache.
- Slot addressing:
  - `block_idx = slot_idx / block_size`
  - `block_offset = slot_idx % block_size`

### 4.2 Layout enum and strides
All strides are in elements (not bytes). `kvx_tensor_desc_t.ndim` MUST match
the layout, and `shape` MUST agree with the cache descriptor.

Standard layouts:
- `KVX_LAYOUT_BLOCK_NHD`: `[num_blocks, block_size, num_kv_heads, head_dim]`
- `KVX_LAYOUT_BLOCK_HND`: `[num_blocks, num_kv_heads, block_size, head_dim]`
- `KVX_LAYOUT_BLOCK_HND_PACKED`:
  `[num_blocks, num_kv_heads, head_dim / pack, block_size, pack]`
  where `pack = shape[4]` and `head_dim % pack == 0`.
- `KVX_LAYOUT_BLOCK_CUSTOM`: Any layout described by explicit shape and stride.

Canonical contiguous strides (row-major, last dim contiguous):
- NHD:
  - `stride[3] = 1`
  - `stride[2] = head_dim`
  - `stride[1] = num_kv_heads * head_dim`
  - `stride[0] = block_size * num_kv_heads * head_dim`
- HND:
  - `stride[3] = 1`
  - `stride[2] = head_dim`
  - `stride[1] = block_size * head_dim`
  - `stride[0] = num_kv_heads * block_size * head_dim`
- HND_PACKED:
  - `stride[4] = 1`
  - `stride[3] = pack`
  - `stride[2] = block_size * pack`
  - `stride[1] = (head_dim / pack) * block_size * pack`
  - `stride[0] = num_kv_heads * (head_dim / pack) * block_size * pack`

Implementations MUST honor provided strides. If a non-contiguous stride
pattern is unsupported, implementations SHOULD return `KVX_STATUS_UNSUPPORTED`.

### 4.3 Dtypes and memory
KVX v1 requires these dtypes:
- `F16`, `BF16`, `F32`
- `F8_E4M3`, `F8_E5M2` (optional support)
- Index dtypes: `S32`, `S64`

`kvx_tensor_desc_t.memory` describes buffer location. Device-side kernels
SHOULD support `KVX_MEMORY_DEVICE` and `KVX_MEMORY_UNIFIED`.

Baseline kernel implementations MUST support `F16`, `BF16`, and `F32` for
cache K/V buffers and IO tensors.

### 4.4 Alignment
Callers SHOULD align K/V buffers to at least 16 bytes for vectorized kernels.
Implementations MAY return `KVX_STATUS_UNSUPPORTED` if alignment requirements
are not met.

## 5. Page Table and Slot Mapping

### 5.1 Block table formats
`kvx_block_table_t` supports multiple encodings:

PACKED:
- 2D table `[seq_count, max_blocks_per_seq]` of physical block ids.
- `indices_count = seq_count * max_blocks_per_seq`.
- `indptr` is NULL and `indptr_count = 0`.
- `beam_width` MUST be 1.

RAGGED:
- `indices` length equals total cached tokens across all sequences.
- `indptr` length `seq_count + 1` with prefix sums of per-sequence lengths.
- For sequence `s`, token index within the sequence is
  `token_idx = i - indptr[s]` for `i` in `[indptr[s], indptr[s+1])`.
- Block id = `indices[i]`; block offset = `token_idx % block_size`.
- `beam_width` MUST be 1.

KV_OFFSETS:
- TensorRT-LLM style table `[seq_count, beam_width, 2, max_blocks_per_seq]`.
- `indices_count = seq_count * beam_width * 2 * max_blocks_per_seq`.
- `indices` values are `KVCacheIndex` (`int32` with secondary pool flag).
- `flags` MUST include `KVX_BLOCK_TABLE_FLAG_KVCACHEINDEX`.
- `index_dtype` MUST be `S32`; `indptr` is NULL.
- `block_size` MUST be a power of two.

### 5.2 Slot mapping
`kvx_slot_mapping_t` is a 1D array of slots for each token in the write batch.
- `slots` length is `token_count`.
- `dtype` MUST be `S32` or `S64`.
- Invalid or padded tokens use `invalid_slot` (default `-1`).
- Implementations MUST treat `invalid_slot` as a no-op write.

### 5.3 Sequence lengths
`kvx_seq_lens_t` provides cached lengths per sequence.
- `dtype` MUST be `S32` or `S64`.
- `seq_count` MUST match the block table's `seq_count`.

## 6. Metadata
All public structs include a `size` field and MUST be size-validated.

### 6.1 Cache descriptor (`kvx_cache_desc_t`)
| Field | Type | Required | Semantics |
| --- | --- | --- | --- |
| `num_blocks` | `uint32_t` | Yes | Total physical blocks in the cache |
| `block_size` | `uint32_t` | Yes | Tokens per block |
| `num_kv_heads` | `uint32_t` | Yes | Number of KV heads |
| `head_dim` | `uint32_t` | Yes | Dimension per KV head |
| `k`, `v` | `kvx_tensor_desc_t` | Yes | K/V buffers, layout, dtype, strides |
| `pool` | `kvx_pool_desc_t` | Conditional | Required for `KVX_BLOCK_TABLE_KV_OFFSETS` |

### 6.2 Tensor descriptor (`kvx_tensor_desc_t`)
| Field | Type | Required | Semantics |
| --- | --- | --- | --- |
| `dtype` | `kvx_dtype_t` | Yes | Element type |
| `layout` | `kvx_layout_t` | Yes | Layout enum |
| `memory` | `kvx_memory_type_t` | Yes | Memory location |
| `ndim` | `uint32_t` | Yes | Number of dimensions |
| `shape[]` | `int64_t[5]` | Yes | Shape in layout order |
| `stride[]` | `int64_t[5]` | Yes | Strides in elements |
| `data` | `void*` | Yes | Buffer pointer |

### 6.3 Block table (`kvx_block_table_t`)
| Field | Type | Required | Semantics |
| --- | --- | --- | --- |
| `format` | `kvx_block_table_format_t` | Yes | Encoding type |
| `index_dtype` | `kvx_dtype_t` | Yes | Indices dtype |
| `indptr_dtype` | `kvx_dtype_t` | Conditional | RAGGED only |
| `seq_count` | `uint32_t` | Yes | Number of sequences |
| `beam_width` | `uint32_t` | Conditional | KV_OFFSETS only |
| `max_blocks_per_seq` | `uint32_t` | Yes | Max blocks per sequence |
| `indices` | `void*` | Yes | Indices buffer |
| `indptr` | `void*` | Conditional | RAGGED only |
| `indices_count` | `uint32_t` | Yes | Total indices |
| `indptr_count` | `uint32_t` | Conditional | RAGGED only |
| `flags` | `uint32_t` | Conditional | KV_OFFSETS only |

### 6.4 Slot mapping (`kvx_slot_mapping_t`)
| Field | Type | Required | Semantics |
| --- | --- | --- | --- |
| `dtype` | `kvx_dtype_t` | Yes | `S32` or `S64` |
| `token_count` | `uint32_t` | Yes | Number of tokens |
| `invalid_slot` | `int64_t` | Yes | Sentinel slot |
| `slots` | `void*` | Yes | Slot buffer |

### 6.5 Sequence lengths (`kvx_seq_lens_t`)
| Field | Type | Required | Semantics |
| --- | --- | --- | --- |
| `dtype` | `kvx_dtype_t` | Yes | `S32` or `S64` |
| `seq_count` | `uint32_t` | Yes | Number of sequences |
| `lengths` | `void*` | Yes | Cached lengths |

### 6.6 Scale descriptor (`kvx_scale_desc_t`)
| Field | Type | Required | Semantics |
| --- | --- | --- | --- |
| `dtype` | `kvx_dtype_t` | Yes | Scale dtype |
| `granularity` | `kvx_scale_granularity_t` | Yes | Per-tensor/block/head |
| `ndim` | `uint32_t` | Conditional | For non-scalar scales |
| `shape[]` | `int64_t[5]` | Conditional | Scale shape |
| `stride[]` | `int64_t[5]` | Conditional | Scale strides |
| `data` | `void*` | Conditional | Scale data |

### 6.7 Pool descriptor (`kvx_pool_desc_t`)
| Field | Type | Required | Semantics |
| --- | --- | --- | --- |
| `memory` | `kvx_memory_type_t` | Yes | Memory location for cache pool |
| `bytes_per_block` | `uint32_t` | Conditional | Required for KV_OFFSETS |
| `primary` | `void*` | Conditional | Base pointer for primary pool |
| `secondary` | `void*` | Optional | Base pointer for secondary pool |

### 6.8 IO descriptor (`kvx_kv_io_desc_t`)
| Field | Type | Required | Semantics |
| --- | --- | --- | --- |
| `key` | `kvx_tensor_desc_t` | Yes | Input keys for `num_tokens` |
| `value` | `kvx_tensor_desc_t` | Yes | Input values for `num_tokens` |
| `num_tokens` | `uint32_t` | Yes | Tokens in the write or gather batch |
| `num_kv_heads` | `uint32_t` | Yes | KV heads for IO tensors |
| `head_dim` | `uint32_t` | Yes | Head dimension for IO tensors |

For IO tensors, implementations MUST accept a dense 3D row-major layout with
shape `[num_tokens, num_kv_heads, head_dim]` and strides
`[num_kv_heads * head_dim, head_dim, 1]`. The `layout` field MAY be set to
`KVX_LAYOUT_BLOCK_CUSTOM` and is otherwise ignored for IO tensors.

### 6.9 Write descriptor (`kvx_write_desc_t`)
| Field | Type | Required | Semantics |
| --- | --- | --- | --- |
| `io` | `kvx_kv_io_desc_t` | Yes | Input K/V tensors |
| `slots` | `kvx_slot_mapping_t` | Yes | Slot mapping for tokens |
| `k_scale` | `void*` | Optional | Per-tensor scale (F32) |
| `v_scale` | `void*` | Optional | Per-tensor scale (F32) |
| `k_scale_desc` | `kvx_scale_desc_t` | Optional | FP8 scale metadata |
| `v_scale_desc` | `kvx_scale_desc_t` | Optional | FP8 scale metadata |

### 6.10 Gather descriptor (`kvx_gather_desc_t`)
| Field | Type | Required | Semantics |
| --- | --- | --- | --- |
| `io` | `kvx_kv_io_desc_t` | Yes | Output K/V tensors |
| `block_table` | `kvx_block_table_t` | Yes | Logical->physical mapping |
| `seq_lens` | `kvx_seq_lens_t` | Yes | Cached lengths per sequence |
| `max_seq_len` | `uint32_t` | Yes | Max gather length per sequence |

## 7. API Semantics
See `kvx_abi.h` for the v1 ABI. Key expectations:
- `kvx_get_version` MUST return `KVX_STATUS_OK` and fill `kvx_version_t`.
- `kvx_validate_cache_desc` MUST validate layout, dtype, shape, stride, and
  required fields, returning `KVX_STATUS_INVALID_ARGUMENT` on failure.
- `kvx_write_kv` writes K/V for tokens using a slot mapping.
  - Invalid slots (`invalid_slot` or negative) MUST be treated as no-op.
  - Implementations MUST validate `num_tokens`, `num_kv_heads`, and `head_dim`
    against the cache descriptor.
  - If `k_scale_desc.data`/`v_scale_desc.data` is NULL, implementations MAY
    fall back to `k_scale`/`v_scale` as per-tensor `F32`.
- `kvx_gather_kv` gathers K/V for attention using a block table and `seq_lens`.
  - `seq_count` MUST match between `block_table` and `seq_lens`.
  - `max_seq_len` bounds the gather length per sequence.
- `stream` parameters are opaque (CUDA/HIP stream or NULL). NULL means the
  default stream for the implementation.

### 7.1 Validation rules (v1)
- `block_size`, `num_blocks`, `num_kv_heads`, `head_dim` MUST be non-zero.
- Layout shapes MUST match the cache descriptor.
- PACKED/RAGGED tables MUST use `S32` or `S64` indices.
- For KV_OFFSETS, `index_dtype` MUST be `S32`.
- For KV_OFFSETS, `block_size` MUST be power of two.
- `indices_count` MUST match the selected `format`.
- `indptr_count` MUST be `seq_count + 1` for RAGGED and `0` otherwise.
- For `KVX_LAYOUT_BLOCK_HND_PACKED`, `head_dim % pack == 0`.

### 7.2 Thread safety and lifetime
- The API is stateless aside from user-provided buffers; functions are
  thread-safe as long as callers avoid concurrent writes to overlapping cache
  regions.
- Callers own the lifetime of all buffer pointers passed via descriptors for
  the duration of the call; implementations MUST NOT retain them after return.

## 8. Error Codes
- `KVX_STATUS_OK`: Success.
- `KVX_STATUS_INVALID_ARGUMENT`: Malformed descriptor or unsupported field.
- `KVX_STATUS_UNSUPPORTED`: Feature or layout unsupported by implementation.
- `KVX_STATUS_OUT_OF_RANGE`: Slot or index out of bounds.
- `KVX_STATUS_INCOMPATIBLE`: ABI version mismatch.
- `KVX_STATUS_INTERNAL_ERROR`: Unexpected internal failure.

## 9. Conformance Tests
Minimum required tests:
- `kvx_get_version` returns matching major/minor and valid `size`.
- ABI size checks for all public structs.
- Layout shape/stride validation for NHD/HND/HND_PACKED.
- Block table counts and `indptr` validation for PACKED/RAGGED/KV_OFFSETS.
- Invalid slot handling (`-1` treated as no-op).

## 10. Optional Features
- FP8 cache: `KVX_DTYPE_F8_E4M3` / `KVX_DTYPE_F8_E5M2` with scale descriptors.
- Secondary pool: `kvx_pool_desc_t.secondary` and KV_OFFSETS flagging.
- Cyclic cache windowing: vendor-specific metadata in `kvx_pool_desc_t` or
  adapter-side state (not standardized in v1).

## 11. Compatibility and Migration
- vLLM: NHD/HND layouts map directly to KVX layouts; slot mapping uses `int64`
  slots with `-1` for padding or non-local tokens.
- TGI: flashinfer/flashdecoding layouts map to NHD; paged K layout maps to
  `KVX_LAYOUT_BLOCK_HND_PACKED` or requires conversion.
- TensorRT-LLM: `KV_OFFSETS` encodes `KVCacheIndex` with primary/secondary pool
  selection; adapters must carry pool pointers and enforce power-of-two
  `tokensPerBlock`.
