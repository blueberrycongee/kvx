# H800 公平基准运行手册（Linux）

中文 | [English](H800_RUNBOOK.md)

本手册用于在 H800 上以最少步骤跑通 KVX vs vLLM 内核级公平基准。

## 快速开始（直接复制）

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

## 可选：其他 dtype

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

## 可选：端到端 A/B（基线 vs KVX）

需要先按 `patches/vllm/README.md` 构建带 KVX 的 vLLM。

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

## 备注
- 若环境不允许安装 `vllm`，请改用本地构建或轮子包；公平基准依赖 `vllm`。
- 如模型受限，请配置 `HF_TOKEN`。
