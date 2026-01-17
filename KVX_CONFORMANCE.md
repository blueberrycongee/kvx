# KVX v1 Conformance Matrix (Reference Implementation)

This document summarizes the current KVX reference implementation support
status. It does not change the KVX v1 specification; it clarifies what is
implemented today and what is planned.

## Status Legend
- Supported: Implemented and covered by tests.
- Partial: Implemented with limitations or missing edge-case validation.
- Planned: In spec, not implemented yet.
- Not supported: Explicitly not implemented in the reference code.

## Feature Matrix

| Feature | Spec Section | Status | Implementation Notes |
| --- | --- | --- | --- |
| ABI version reporting (`kvx_get_version`) | 3 | Supported | `kvx/kvx_abi.c` |
| Cache descriptor validation (layout/shape/stride) | 4, 7.1 | Supported | Only canonical contiguous strides accepted in `kvx/kvx_abi.c` |
| Layout: `KVX_LAYOUT_BLOCK_NHD` | 4.2 | Supported | Kernel + tests in `kvx/kernels/kvx_paged_kv.cu`, `kvx/tests/kvx_kernel_test.cu` |
| Layout: `KVX_LAYOUT_BLOCK_HND` | 4.2 | Supported | Kernel + tests in `kvx/kernels/kvx_paged_kv.cu`, `kvx/tests/kvx_kernel_test.cu` |
| Layout: `KVX_LAYOUT_BLOCK_HND_PACKED` | 4.2 | Partial | Kernel supports; no test coverage yet |
| Layout: `KVX_LAYOUT_BLOCK_CUSTOM` | 4.2 | Not supported | Returns `KVX_STATUS_UNSUPPORTED` |
| Cache dtypes: F16/BF16/F32 | 4.3 | Supported | Kernels implement these dtypes |
| Cache dtypes: FP8 (E4M3/E5M2) | 4.3, 10 | Planned | Not implemented in kernels |
| Slot mapping dtypes: S32/S64 | 5.2 | Supported | `kvx_launch_write_kv_*` supports S32/S64 |
| `invalid_slot` semantics | 5.2 | Partial | Kernel treats `slot < 0` as no-op; `invalid_slot` value not enforced |
| Block table: PACKED | 5.1 | Supported | Gather kernel supports PACKED only |
| Block table: RAGGED | 5.1 | Planned | Not implemented |
| Block table: KV_OFFSETS | 5.1 | Planned | Not implemented |
| Gather index dtype S32/S64 | 7.1 | Supported | Gather kernel supports S32/S64 |
| Gather bounds checking (block id range) | 7.1 | Partial | Negative block id skipped; upper bound not checked |
| IO tensor layout (write) 3D contiguous | 6.8 | Supported | `kvx_validate_io_tensor_3d` requires contiguous |
| IO tensor layout (gather) 4D contiguous | 6.10 | Supported | `kvx_validate_io_tensor_4d` requires contiguous |
| Write prefill fast path | 7 | Supported | Vectorized path in `kvx/kernels/kvx_paged_kv.cu` |
| Scale descriptors (`kvx_scale_desc_t`) | 6.6, 6.9 | Planned | Parsed in descriptors, not used by kernels |
| Pool descriptor + secondary pool | 6.7 | Planned | Only specified in ABI/spec |
| Memory types (device/unified) | 4.3 | Partial | Kernels assume device-accessible pointers; no explicit checks |

## Notes
- The reference implementation focuses on correctness and NHD/HND layouts.
- Spec-required validation may be stricter than the current implementation.
- See `kvx/KVX_SPEC.md` for normative language.
