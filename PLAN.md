# KVX Plan

Goal: Build a GPU-first KV cache standard layer (layout + metadata + C ABI)
with high-performance CUDA kernels and a benchmark suite that matches or beats
existing engines.

## Scope v1
- Paged KV layout + page table spec
- Read/write APIs
- Single-step decode kernel
- Reproducible benchmarks

## Deliverables (10)
1. [x] Layout comparison across vLLM, TensorRT-LLM, and TGI
   - `KV_CACHE_LAYOUTS.md`
2. [x] KVX v1 spec with layouts, metadata, and API semantics
   - `KVX_SPEC.md`
3. [x] Stable C ABI header with versioning and enums
   - `kvx_abi.h`
4. [x] ABI validation + smoke tests
   - `kvx_abi.c`, `tests/kvx_abi_test.c`
5. [x] CUDA kernel skeletons for write/gather
   - `kernels/kvx_paged_kv.cu`, `kernels/kvx_paged_kv.h`
6. [x] Kernel correctness tests for F32/F16/BF16
   - `tests/kvx_kernel_test.cu`
7. [x] Benchmark harness with dtype support
   - `benchmarks/kvx_bench.cu`
8. [x] Fair vLLM vs KVX kernel benchmark
   - `benchmarks/fair_bench.py`
9. [x] Results summary with environment capture
   - `benchmarks/results/README.md`, `benchmarks/results/fair_bench_wsl_env.txt`
10. [x] Reproducible runbooks for benchmarks and tests
    - `benchmarks/README.md`, `tests/README.md`
