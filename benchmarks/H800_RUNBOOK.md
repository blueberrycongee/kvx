# H800 Fair Bench Runbook (Linux)

This runbook prepares a minimal H800 run of the KVX vs vLLM kernel-level
fair benchmark. It keeps steps short for rental instances.

## Quick start (copy/paste)

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install vllm

nvcc -std=c++17 -O3 -arch=sm_90 -I. \
  kvx_abi.c \
  kernels/kvx_paged_kv.cu \
  benchmarks/kvx_bench.cu \
  -o benchmarks/kvx_bench_h800

mkdir -p benchmarks/results
python benchmarks/capture_env.py \
  --output benchmarks/results/fair_bench_h800_env.json

python benchmarks/fair_bench.py \
  --dtype float16 --iters 50 --warmup 10 --repeats 10 \
  --bin-path ./benchmarks/kvx_bench_h800 \
  --output-csv benchmarks/results/fair_bench_h800_f16.csv
```

## Optional: additional dtypes

```bash
python benchmarks/fair_bench.py \
  --dtype float32 --iters 50 --warmup 10 --repeats 10 \
  --bin-path ./benchmarks/kvx_bench_h800 \
  --output-csv benchmarks/results/fair_bench_h800_f32.csv

python benchmarks/fair_bench.py \
  --dtype bfloat16 --iters 50 --warmup 10 --repeats 10 \
  --bin-path ./benchmarks/kvx_bench_h800 \
  --output-csv benchmarks/results/fair_bench_h800_bf16.csv
```

## Optional: E2E A/B (baseline vs KVX)

This requires vLLM built with the KVX patch (see `patches/vllm/README.md`).

```bash
MODEL=Qwen/Qwen2-7B-Instruct
python benchmarks/e2e_bench.py \
  --model ${MODEL} --dtype float16 --batch-size 8 \
  --input-len 256 --output-len 256 --iters 20 --warmup 5 \
  --output-json benchmarks/results/e2e_h800_baseline.json

VLLM_USE_KVX_CACHE_WRITE=1 python benchmarks/e2e_bench.py \
  --model ${MODEL} --dtype float16 --batch-size 8 \
  --input-len 256 --output-len 256 --iters 20 --warmup 5 \
  --output-json benchmarks/results/e2e_h800_kvx.json
```

## Notes
- If `vllm` is gated by your environment, install it from a wheel or
  a local build. The fair benchmark will fail without `vllm`.
- Use `HF_TOKEN` if your chosen model is gated.
