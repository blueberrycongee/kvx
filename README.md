# KVX

English | [中文](README.zh-CN.md)

KVX is a GPU-first KV cache standard and kernel library for LLM inference. The
project defines a stable KV cache layout, metadata, and C ABI, and ships CUDA
kernels plus reproducible benchmarks for vLLM-style paged attention.

## Scope
- Standard KV cache layout + metadata (paged KV).
- Stable C ABI for adapters and engines.
- CUDA write/gather kernel skeletons.
- Fair, reproducible benchmarks (kernel-level + E2E).

## Repo Layout
- `KVX_SPEC.md`: KVX v1 spec (layout, metadata, API semantics).
- `KV_CACHE_LAYOUTS.md`: Layout comparison across vLLM / TRT-LLM / TGI.
- `kvx_abi.h`, `kvx_abi.c`: KVX C ABI and helpers.
- `kernels/`: CUDA kernels and headers.
- `benchmarks/`: Benchmark harness and runbooks.
- `tests/`: ABI and kernel correctness tests.

## Integration (vLLM)
This repo does not vendor vLLM. To integrate KVX with vLLM, apply the KVX
patch under `patches/vllm/` to a local vLLM checkout and build with
`VLLM_ENABLE_KVX=ON`. Runtime selection is controlled via an environment flag
(e.g., `VLLM_USE_KVX_CACHE_WRITE=1`). See `patches/vllm/README.md` for the
patch workflow.

## E2E Benchmark (WSL RTX 3060 Laptop)
Reference runs compare vLLM baseline vs KVX cache write under identical
settings. We ran 3 alternating A/B runs to reduce variance.

### Environment
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (WSL)
- torch: 2.7.0.dev20250310+cu124
- CUDA: 12.4
- vLLM upstream: v0.14.0rc0-508-g0346396e9 (commit 0346396e94106edcbce13e083da172c119d0aa17; local build reports 0.0.0+kvx via VLLM_VERSION_OVERRIDE)

### Config
- model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- dtype: float16
- batch_size: 8
- input_len / output_len: 128 / 128
- max_model_len: 264
- warmup / iters: 5 / 20
- gpu_memory_utilization: 0.80
- tensor_parallel_size: 1

### Summary (mean +/- stdev over 3 runs)
- tokens/s: baseline 527.94 +/- 129.76, KVX 676.83 +/- 12.71 (1.28x faster)
- latency_ms: baseline 3881 +/- 938, KVX 2795 +/- 52 (1.39x lower)

See `benchmarks/results/README.md` for kernel-level fair benchmark results.

## Repro
- Kernel-level benchmark: `benchmarks/README.md`
- E2E benchmark runbook: `benchmarks/E2E_RUNBOOK.md`
- Tests: `tests/README.md`
