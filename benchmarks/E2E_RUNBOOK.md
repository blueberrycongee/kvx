# E2E Benchmark Runbook (vLLM baseline + KVX)

This runbook measures end-to-end generation throughput and latency using vLLM.
It can be used for baseline and KVX A/B comparisons once vLLM is built with KVX.

## Prereqs
- CUDA driver + toolkit installed (`nvidia-smi` available).
- Python environment with `torch` + `vllm`.
- GPU visible (`torch.cuda.is_available()` returns `True`).

## Build vLLM with KVX
From the KVX repo root inside WSL:
```bash
export KVX_ROOT="$(pwd)"
cd /path/to/vllm
CMAKE_ARGS="-DVLLM_ENABLE_KVX=ON -DVLLM_KVX_ROOT=${KVX_ROOT}" \
  pip install -e .
```

## Sanity check
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

## Local 3060 (small model)
```bash
python benchmarks/e2e_bench.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype float16 --batch-size 4 --input-len 128 --output-len 128 \
  --iters 10 --warmup 3 \
  --output-json benchmarks/results/e2e_3060_f16.json
```

## Fair A/B (baseline vs KVX)
Use identical settings for both runs.
```bash
VLLM_USE_KVX_CACHE_WRITE=0 python benchmarks/e2e_bench.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype float16 --batch-size 4 --input-len 128 --output-len 128 \
  --iters 10 --warmup 3 \
  --output-json benchmarks/results/e2e_3060_f16_baseline.json

VLLM_USE_KVX_CACHE_WRITE=1 python benchmarks/e2e_bench.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype float16 --batch-size 4 --input-len 128 --output-len 128 \
  --iters 10 --warmup 3 \
  --output-json benchmarks/results/e2e_3060_f16_kvx.json
```

## A100 (larger model)
```bash
python benchmarks/e2e_bench.py \
  --model meta-llama/Llama-2-7b-hf \
  --dtype float16 --batch-size 8 --input-len 256 --output-len 256 \
  --iters 20 --warmup 5 \
  --output-json benchmarks/results/e2e_a100_f16.json
```

## Capture environment
```bash
mkdir -p benchmarks/results
{
  nvidia-smi -L
  nvidia-smi
  python - <<'PY'
import torch
import vllm
print("torch_version=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("device=", torch.cuda.get_device_name(0))
print("vllm_version=", getattr(vllm, "__version__", "unknown"))
PY
} | tee benchmarks/results/e2e_env.txt
```

## Notes
- Reduce `--batch-size` or `--input-len` if you hit OOM.
- Use `--dry-run` to validate the config without loading a model.
- The JSON output includes per-iteration latency and tokens/s.
- KVX cache write only applies when `kv_cache_dtype=auto`.
