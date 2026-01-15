# KV Cache Layouts (vLLM / TensorRT-LLM / TGI)

This document summarizes KV cache layouts, page/block metadata, and adapter-relevant semantics across the three engines.

## vLLM

### Layouts (paged K/V)
- K and V are stored as separate paged tensors.
- vLLM exposes two cache write kernels with different layout expectations:
  - `reshape_and_cache_flash` (paged attention): NHD or HND.
    - NHD: `[num_blocks, block_size, num_heads, head_size]`
    - HND: `[num_blocks, num_heads, block_size, head_size]`
    - Layout selection is based on stride: contiguous heads use NHD (`head_stride == head_size`).
  - `reshape_and_cache` (non-flash paged attention): packed K + HND V.
    - K: `[num_blocks, num_heads, head_size / x, block_size, x]`
    - V: `[num_blocks, num_heads, head_size, block_size]`
    - `x` is a packing factor chosen by the kernel (requires `head_size % x == 0`).
- Addressing (element-based strides):
  - `block_idx = slot_idx / block_size`, `block_offset = slot_idx % block_size`
  - `dst = cache + block_idx * block_stride + block_offset * page_stride`
  - HND uses `head_stride` to index per-head segments.

Source: `vllm/csrc/cache_kernels.cu`
Source: `vllm/csrc/cache.h`

### Page table / slot mapping
- `block_table`: `int32` tensor with shape `[max_num_reqs, max_num_blocks_per_req]`.
  - Each row maps logical block index to physical block id for a request.
- `slot_mapping`: `int64` tensor with shape `[max_num_batched_tokens]`.
  - Global slot = `block_id * block_size + block_offset`.
  - Padded or non-local tokens use `-1`.
- Hybrid block sizes: manager `block_size` can be split into `kernel_block_size`.
  - Each manager block expands to `blocks_per_kv_block` kernel blocks.
- CP interleave: for `cp_world_size > 1`, slot mapping uses a
  `virtual_block_size = block_size * cp_world_size` and marks non-local slots as `-1`
  based on `cp_kv_cache_interleave_size`.

Source: `vllm/vllm/v1/worker/block_table.py`

### FlashInfer metadata (when used)
- `paged_kv_indptr`, `paged_kv_indices`, `paged_kv_last_page_len` computed from block tables.
  - `paged_kv_indptr` length is `seq_count + 1` (prefix sums).
  - `paged_kv_indices` length equals total cached blocks across the batch.
  - `paged_kv_last_page_len` provides tail block lengths.

Source: `vllm/vllm/v1/attention/backends/flashinfer.py`

## TGI (text-generation-inference)

### Layouts
- KV cache is allocated per layer; layout depends on attention backend and device.
- `flashinfer` / `flashdecoding` (CUDA):
  - K/V: `[num_blocks, BLOCK_SIZE, num_heads, head_size]`
- IPEX CPU:
  - K/V: `[num_blocks, num_heads, BLOCK_SIZE, head_size]`
- `paged` (kernels-community/paged-attention):
  - K: `[num_blocks, num_heads, head_size // x, BLOCK_SIZE, x]`
  - V: `[num_blocks, num_heads, head_size, BLOCK_SIZE]`
  - `x = BLOCK_SIZE / element_size` (for IPEX XPU, `x = 1`).
- KV cache dtype is passed via `kv_cache_dtype`; FP8 support depends on backend/device.

Source: `text-generation-inference/server/text_generation_server/layers/attention/kv_cache.py`

### Page tables / slots
- `block_tables`: per-request list of block ids.
- `slots`: flattened list of global slot ids per token.
- `block_tables_tensor`: padded `int32` tensor with shape `[num_reqs, max_blocks]`.
- `block_tables_ragged`: `int32` vector of per-token block ids;
  length equals total cached tokens across the batch.

Source: `text-generation-inference/server/text_generation_server/models/flash_causal_lm.py`

### Ragged block tables (FlashInfer)
- `block_tables_to_ragged` builds per-token block ids.
- For each sequence, `seq_len = cache_len + input_len` determines ragged length.

Source: `text-generation-inference/server/text_generation_server/models/metadata_kernels.py`

## TensorRT-LLM

### Layouts
- Paged KV cache exposed via `KVBlockArray`.
- Each block stores K/V as `[numHeads, tokensPerBlock, hiddenSizePerHead]`.
- K layout is remapped to match V layout in kernels.
- `tokensPerBlock` must be a power of two (`tokensPerBlockLog2` is used for indexing).

Source: `TensorRT-LLM/cpp/tensorrt_llm/kernels/kvCacheUtils.h`

### Page table / block offsets
- `block_offsets` table shape `[B, W, 2, M]`:
  - `B`: num sequences, `W`: beam width, `2`: K/V tables, `M`: max blocks per seq.
- Offsets are `KVCacheIndex` values (`int32` with top bit indicating secondary pool).
- Block pointer = pool pointer (primary/secondary) + `offset * bytes_per_block`.
  - `bytes_per_block = tokensPerBlock * bytesPerToken`.

Source: `TensorRT-LLM/cpp/tensorrt_llm/kernels/kvCacheUtils.h`
Source: `TensorRT-LLM/cpp/include/tensorrt_llm/kernels/kvCacheIndex.h`

### Additional metadata
- Optional secondary pool pointer.
- Windowing metadata: `mSinkTokens`, `mCyclicCacheLen`, `mBubbleLen`, `mMaxAttentionWindow`.

Source: `TensorRT-LLM/cpp/tensorrt_llm/kernels/kvCacheUtils.h`

## Common fields (KVX v1 candidates)
- `tokens_per_block` / `block_size` (page size)
- `block_table` / `block_offsets` (logical -> physical block id)
- `slot_mapping` or `paged_kv_indices + paged_kv_indptr`
- K/V block internal layout (standardized with layout enum + strides)
- Optional: FP8 scales, secondary pool flagging, cyclic cache parameters

## KVX adapter mapping checklist
- vLLM:
  - `reshape_and_cache_flash` maps to `KVX_LAYOUT_BLOCK_NHD` or `KVX_LAYOUT_BLOCK_HND`.
  - `reshape_and_cache` packed K maps to `KVX_LAYOUT_BLOCK_HND_PACKED`; V maps to `KVX_LAYOUT_BLOCK_HND`.
  - `slot_mapping` (int64 with `-1`) maps directly to `kvx_slot_mapping_t`.
  - `block_table` (int32 packed) maps to `kvx_block_table_t` PACKED.
- TGI:
  - `flashinfer`/`flashdecoding` maps to `KVX_LAYOUT_BLOCK_NHD`.
  - `paged` K layout maps to `KVX_LAYOUT_BLOCK_HND_PACKED` (pack = `x`).
  - `block_tables_tensor` -> PACKED; `block_tables_ragged` -> RAGGED.
  - `slots` -> `kvx_slot_mapping_t`.
- TensorRT-LLM:
  - `KVBlockArray` maps to `KVX_LAYOUT_BLOCK_HND`.
  - `block_offsets` maps to `KVX_BLOCK_TABLE_KV_OFFSETS` with `KVX_BLOCK_TABLE_FLAG_KVCACHEINDEX`.
  - `tokensPerBlock` power-of-two constraint must be enforced by adapters.
