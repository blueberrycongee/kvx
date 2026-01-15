# KVX Benchmark Harness

English | [中文](README.zh-CN.md)

This harness measures decode-time write/gather throughput for KVX baseline
CUDA kernels on a single GPU.

## Build (WSL)

Example build from WSL (adjust paths as needed):

```bash
nvcc -std=c++17 -O3 -arch=sm_86 \
  kvx_abi.c \
  kernels/kvx_paged_kv.cu \
  benchmarks/kvx_bench.cu \
  -o benchmarks/kvx_bench
```

## Run

```bash
benchmarks/kvx_bench \
  --seq-count=8 --seq-len=128 --block-size=16 --num-heads=8 --head-dim=128 \
  --warmup=10 --iters=50 --dtype=float32 --tokens-per-seq=1
```

`--tokens-per-seq > 1` triggers the prefill write path in the KVX kernel.

## Metrics

The program prints:
- tokens/s for write and gather
- GB/s for approximate K/V reads + writes
- per-iteration latency

It also emits a CSV-like summary line per kernel for easy logging.

## Fair vLLM vs KVX (WSL)

Use the fair benchmark to compare vLLM `reshape_and_cache_flash` (NHD layout)
with KVX write kernels under the same WSL environment.

```bash
python benchmarks/fair_bench.py \
  --dtype=float32 --iters=50 --warmup=10 --repeats=10 --tokens-per-seq=1
```

Use `--bin-path` to point at your KVX binary and `--output-csv` to select an
output file name.

The script writes a CSV summary to:
`benchmarks/results/fair_bench_wsl_*.csv`.

See `benchmarks/results/README.md` for a summary of the most recent run.

## Environment capture

Use the shared capture script to record GPU/driver/CUDA/torch/vLLM metadata:

```bash
python benchmarks/capture_env.py \
  --output benchmarks/results/fair_bench_wsl_env.json
```

## A100 / Linux runbook

For an A100 Linux run (Autodl, bare metal, or cloud), use:
`benchmarks/A100_RUNBOOK.md`.

## H800 / Linux runbook

For a minimal H800 run, use:
`benchmarks/H800_RUNBOOK.md`.

## End-to-end (vLLM baseline)

This script measures end-to-end generation throughput and latency using vLLM.

```bash
python benchmarks/e2e_bench.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype float16 --batch-size 4 --input-len 128 --output-len 128 \
  --iters 10 --warmup 3 \
  --output-json benchmarks/results/e2e_3060_f16.json
```

For a full runbook (3060 + A100), see `benchmarks/E2E_RUNBOOK.md`.
 
