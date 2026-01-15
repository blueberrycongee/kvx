# A100 Fair Bench Runbook

This runbook prepares a clean, reproducible A100 run of the KVX vs vLLM
kernel-level fair benchmark on Linux.

## Prereqs
- CUDA driver + toolkit installed (`nvidia-smi`, `nvcc` available).
- Python environment with `torch` + `vllm`.
- GPU visible (`torch.cuda.is_available()` returns `True`).

## Sanity check (Python)
```bash
python - <<'PY'
import torch
import vllm
print("torch_version=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("device=", torch.cuda.get_device_name(0))
print("vllm_version=", getattr(vllm, "__version__", "unknown"))
PY
```

## Build KVX bench (A100 sm_80)
```bash
nvcc -std=c++17 -O3 -arch=sm_80 \
  kvx_abi.c \
  kernels/kvx_paged_kv.cu \
  benchmarks/kvx_bench.cu \
  -o benchmarks/kvx_bench
```

## Capture environment
```bash
mkdir -p benchmarks/results
python benchmarks/capture_env.py \
  --output benchmarks/results/fair_bench_a100_env.json
```

## Run fair benchmark (KVX vs vLLM)
```bash
python benchmarks/fair_bench.py \
  --dtype=float32 --iters=50 --warmup=10 --repeats=10 \
  --bin-path ./benchmarks/kvx_bench \
  --output-csv benchmarks/results/fair_bench_a100_f32.csv

python benchmarks/fair_bench.py \
  --dtype=float16 --iters=50 --warmup=10 --repeats=10 \
  --bin-path ./benchmarks/kvx_bench \
  --output-csv benchmarks/results/fair_bench_a100_f16.csv

python benchmarks/fair_bench.py \
  --dtype=bfloat16 --iters=50 --warmup=10 --repeats=10 \
  --bin-path ./benchmarks/kvx_bench \
  --output-csv benchmarks/results/fair_bench_a100_bf16.csv
```

## Notes
- `fair_bench.py` includes both decode-style and prefill-style cases by default.
- Use `--tokens-per-seq` to override the default prefill length if needed.
- Store all results under `benchmarks/results/` for later summary.

## Edge cases
- `nvcc` missing: install CUDA toolkit or use an image with CUDA already set up.
- `vllm` import fails: install `vllm` in the active environment.
- `torch.cuda.is_available()` is false: verify drivers and `CUDA_VISIBLE_DEVICES`.
