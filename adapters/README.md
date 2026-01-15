# KVX Adapter Stubs

This folder contains adapter stubs that define expected inputs and outputs
for integrating KVX with other inference engines. These stubs are not
implemented yet; they document the integration contract and provide
typed placeholders for future work.

## TGI (text-generation-inference)

Expected inputs (normalized):
- K/V cache tensors
- `block_tables_tensor` (packed) or `block_tables_ragged` (ragged)
- `slots` (token slot mapping)
- `block_size`, `num_kv_heads`, `head_dim`, `kv_cache_dtype`, `layout`

Expected KVX outputs:
- `kvx_cache_desc_t` for K/V (layout + strides + dtype)
- `kvx_block_table_t` in `PACKED` or `RAGGED` form
- `kvx_slot_mapping_t` for token slots

Layout mapping (summary):
- flashinfer/flashdecoding -> `KVX_LAYOUT_BLOCK_NHD`
- paged K layout -> `KVX_LAYOUT_BLOCK_HND_PACKED`

## TensorRT-LLM

Expected inputs (normalized):
- `kv_block_offsets` (KVCacheIndex table)
- `pool_primary`, `pool_secondary`, `bytes_per_block`
- `tokens_per_block`, `num_kv_heads`, `head_dim`, `kv_cache_dtype`

Expected KVX outputs:
- `kvx_cache_desc_t` (with pool pointers)
- `kvx_block_table_t` in `KVX_BLOCK_TABLE_KV_OFFSETS` format

Integration constraints:
- `tokens_per_block` must be power of two.
- KV offsets must honor primary/secondary pool flagging.
