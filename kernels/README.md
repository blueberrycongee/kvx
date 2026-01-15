# KVX CUDA Kernel Skeletons

These kernels provide a correctness-first baseline for paged KV write/gather.
They currently support:
- Cache layouts: `KVX_LAYOUT_BLOCK_NHD`, `KVX_LAYOUT_BLOCK_HND`,
  `KVX_LAYOUT_BLOCK_HND_PACKED`
- Cache dtype: `F32`, `F16`, `BF16`
- Slot mapping: `S32` or `S64`
- Block table (gather): `KVX_BLOCK_TABLE_PACKED` with `S32` or `S64` indices
- NHD write path uses 16-byte vectorized copies when alignment and strides
  permit, with a scalar fallback for edge cases
- Prefill write path uses a token-major 2D grid with fused K/V vector copies
  (16/32/64-byte when alignment permits)

Input/output assumptions:
- Write IO tensors are contiguous `[num_tokens, num_heads, head_dim]`.
- Gather output tensors are contiguous
  `[seq_count, max_seq_len, num_heads, head_dim]`.

These are skeletons meant to be optimized and extended; they prioritize clear
indexing and layout handling over performance.

Preferred entry points for integrations:
- `kvx_launch_write_kv`
- `kvx_launch_write_kv_prefill`
- `kvx_launch_gather_kv`

The per-dtype functions remain available but are considered internal helpers.
