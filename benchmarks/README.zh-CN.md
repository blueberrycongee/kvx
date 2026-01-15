# KVX 基准测试

中文 | [English](README.md)

本基准用于在单卡上测量 KVX 基线 CUDA 内核的解码阶段写入/聚合吞吐。

## 构建（WSL）

WSL 中示例构建（按需调整路径）：

```bash
nvcc -std=c++17 -O3 -arch=sm_86 \
  kvx_abi.c \
  kernels/kvx_paged_kv.cu \
  benchmarks/kvx_bench.cu \
  -o benchmarks/kvx_bench
```

## 运行

```bash
benchmarks/kvx_bench \
  --seq-count=8 --seq-len=128 --block-size=16 --num-heads=8 --head-dim=128 \
  --warmup=10 --iters=50 --dtype=float32 --tokens-per-seq=1
```

`--tokens-per-seq > 1` 会触发 KVX 内核的 prefill 写入路径。

## 指标

程序会输出：
- write / gather 的 tokens/s
- 近似的 K/V 读写 GB/s
- 单次迭代延迟

同时会打印一行 CSV 风格的汇总，便于记录。

## 公平对比（vLLM vs KVX, WSL）

使用公平基准，在同一 WSL 环境下对比 vLLM 的
`reshape_and_cache_flash`（NHD 布局）与 KVX 写入内核。

```bash
python benchmarks/fair_bench.py \
  --dtype=float32 --iters=50 --warmup=10 --repeats=10 --tokens-per-seq=1
```

用 `--bin-path` 指定 KVX 二进制路径，`--output-csv` 指定输出文件名。

脚本会输出：
`benchmarks/results/fair_bench_wsl_*.csv`。

最新汇总见 `benchmarks/results/README.md`。

## 环境采集

使用统一脚本记录 GPU/driver/CUDA/torch/vLLM 元数据：

```bash
python benchmarks/capture_env.py \
  --output benchmarks/results/fair_bench_wsl_env.json
```

## A100 / Linux 运行手册

A100（Autodl、裸机或云端）请用：
`benchmarks/A100_RUNBOOK.md`。

## H800 / Linux 运行手册

最简 H800 流程请用：
`benchmarks/H800_RUNBOOK.zh-CN.md`。

## 端到端（vLLM 基线）

该脚本用于测量 vLLM 的端到端生成吞吐与延迟：

```bash
python benchmarks/e2e_bench.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype float16 --batch-size 4 --input-len 128 --output-len 128 \
  --iters 10 --warmup 3 \
  --output-json benchmarks/results/e2e_3060_f16.json
```

完整流程（3060 + A100）见 `benchmarks/E2E_RUNBOOK.md`。
