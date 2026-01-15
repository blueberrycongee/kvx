# KVX

中文 | [English](README.md)

KVX 是面向 LLM 推理的 GPU 优先 KV 缓存标准与内核库。项目定义稳定的 KV
缓存布局、元数据和 C ABI，并提供用于 vLLM 风格分页注意力的 CUDA 内核与
可复现基准。

## 范围
- 标准 KV 缓存布局 + 元数据（分页 KV）。
- 面向适配器与引擎的稳定 C ABI。
- CUDA 写入/聚合内核骨架。
- 公平、可复现的基准（内核级 + 端到端）。

## 仓库结构
- `KVX_SPEC.md`：KVX v1 规范（布局、元数据、API 语义）。
- `KV_CACHE_LAYOUTS.md`：vLLM / TRT-LLM / TGI 布局对比。
- `kvx_abi.h`, `kvx_abi.c`：KVX C ABI 及辅助实现。
- `kernels/`：CUDA 内核与头文件。
- `benchmarks/`：基准测试与运行手册。
- `tests/`：ABI 与内核正确性测试。

## 集成（vLLM）
本仓库不内置 vLLM。要将 KVX 集成到 vLLM，请应用 `patches/vllm/` 下的补丁，
并以 `VLLM_ENABLE_KVX=ON` 编译。运行时通过环境变量切换（如
`VLLM_USE_KVX_CACHE_WRITE=1`）。补丁流程见 `patches/vllm/README.md`。

## E2E 基准（WSL RTX 3060 Laptop）
参考运行在相同设置下对比 vLLM 基线与 KVX cache 写入。为降低波动，我们做了
3 轮交替 A/B 运行。

### 环境
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU（WSL）
- torch：2.7.0.dev20250310+cu124
- CUDA：12.4
- vLLM 上游版本：v0.14.0rc0-508-g0346396e9（commit 0346396e94106edcbce13e083da172c119d0aa17；本地构建因 VLLM_VERSION_OVERRIDE 显示为 0.0.0+kvx）

### 配置
- model：TinyLlama/TinyLlama-1.1B-Chat-v1.0
- dtype：float16
- batch_size：8
- input_len / output_len：128 / 128
- max_model_len：264
- warmup / iters：5 / 20
- gpu_memory_utilization：0.80
- tensor_parallel_size：1

### 汇总（3 轮均值 +/- 标准差）
- tokens/s：baseline 527.94 +/- 129.76，KVX 676.83 +/- 12.71（1.28x 更快）
- latency_ms：baseline 3881 +/- 938，KVX 2795 +/- 52（降低 1.39x）

内核级公平基准结果见 `benchmarks/results/README.md`。

## 复现
- 内核基准：`benchmarks/README.md`
- E2E 运行手册：`benchmarks/E2E_RUNBOOK.md`
- 测试：`tests/README.md`
