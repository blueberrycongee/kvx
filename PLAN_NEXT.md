# KVX 里程碑计划（闭环可验）

目的：把 KVX 从“规范+原型+基准”推进到“跨引擎可用、可验证、可维护”的标准层，
每个里程碑都有本地可闭环的验收方式。

## 约束与原则
- 所有新文件落在 `D:\`（WSL: `/mnt/d`），避免写入 C 盘。
- 每个里程碑必须给出可执行的本地闭环校验命令与期望结果。
- 无外部实验数据也可推进；需要 GPU 的里程碑标注为“本机 GPU”。

## 里程碑

### M1 规范-实现一致性基线（无外部实验）
范围：
- 明确“已支持/未支持/计划支持”的特性矩阵。
- 对齐 `KVX_SPEC` 与当前实现能力。
交付物：
- 新增 `kvx/KVX_CONFORMANCE.md`（特性矩阵）。
- 更新 `kvx/KVX_SPEC.md` 中与实现不一致的段落（标注 v1 现状）。
闭环校验：
- `test -f kvx/KVX_CONFORMANCE.md`
- `rg -n "Conformance|特性矩阵" kvx/KVX_CONFORMANCE.md kvx/KVX_SPEC.md`
- 期望：文件存在且矩阵条目与实现一致。

### M2 ABI 描述符校验完备化（无外部实验）
范围：
- 完善 write/gather/slot/block table/seq_lens 的参数校验与错误语义。
交付物：
- 更新 `kvx/kvx_abi.c`（新增校验逻辑）。
- 扩展 `kvx/tests/kvx_abi_test.c`（覆盖边界与错误码）。
闭环校验：
- `cc -std=c11 kvx/kvx_abi.c kvx/tests/kvx_abi_test.c -o /tmp/kvx_abi_test`
- `/tmp/kvx_abi_test`
- 期望：退出码 0。

### M3 适配层最小实现 + 单测（无外部实验）
范围：
- 为 TGI / TensorRT-LLM 适配层补齐最小描述符构建逻辑。
交付物：
- 实现 `kvx/adapters/tgi_adapter.py`、`kvx/adapters/trt_llm_adapter.py`。
- 新增 `kvx/adapters/tests/tgi_adapter_test.py`、
  `kvx/adapters/tests/trt_llm_adapter_test.py`（纯 CPU 形状/字段校验）。
闭环校验：
- `python -m py_compile kvx/adapters/tgi_adapter.py kvx/adapters/trt_llm_adapter.py`
- `python kvx/adapters/tests/tgi_adapter_test.py`
- `python kvx/adapters/tests/trt_llm_adapter_test.py`
- 期望：全部退出码 0。

### M4 Kernel 布局覆盖与鲁棒性（本机 GPU）
范围：
- 覆盖 NHD/HND/HND_PACKED 的写入与回退路径。
- 验证非连续输入的行为与回退正确性。
交付物：
- 扩展 `kvx/tests/kvx_kernel_test.cu`（新增布局与非连续用例）。
- 必要时完善 `kvx/kernels/kvx_paged_kv.cu` 的错误处理。
闭环校验：
- `nvcc -O2 -arch=sm_86 kvx/tests/kvx_kernel_test.cu kvx/kernels/kvx_paged_kv.cu -o /tmp/kvx_kernel_test`
- `/tmp/kvx_kernel_test`
- 期望：输出 `kvx_kernel_test passed`。

### M5 Gather 格式扩展（本机 GPU）
范围：
- 实现 RAGGED 与 KV_OFFSETS gather 支持。
交付物：
- 更新 `kvx/kernels/kvx_paged_kv.cu` 与相关校验。
- 扩展 `kvx/tests/kvx_kernel_test.cu` 覆盖 RAGGED/KV_OFFSETS。
闭环校验：
- `nvcc -O2 -arch=sm_86 kvx/tests/kvx_kernel_test.cu kvx/kernels/kvx_paged_kv.cu -o /tmp/kvx_kernel_test`
- `/tmp/kvx_kernel_test`
- 期望：全部 gather 用例通过。

### M6 vLLM 集成回归（本机 GPU + WSL）
范围：
- 验证 KVX write 与 vLLM 默认 write 行为一致。
交付物：
- 更新 `kvx/patches/vllm/kvx.patch` 的相关测试覆盖（如有必要）。
闭环校验：
- `kvx/patches/vllm/apply_vllm_patch.sh --check /mnt/d/Desktop/kvx-workspace/third_party/vllm`
- 在 vLLM 仓库执行：`pytest tests/kernels/attention/test_cache.py -k kvx`
- 期望：KVX 测试通过，且与默认路径数值一致。

### M7 基准与环境捕获稳定性（本机 GPU）
范围：
- 固定基准输出字段与环境捕获格式。
交付物：
- 更新 `kvx/benchmarks/kvx_bench.cu` 与 `kvx/benchmarks/capture_env.py`
  的字段稳定性。
闭环校验：
- `python kvx/benchmarks/tests/kvx_bench_cli_test.py`
- `python kvx/benchmarks/tests/env_capture_test.py`
- 期望：全部退出码 0。

### M8 3060 本机性能闭环（本机 GPU）
范围：
- 用固定用例记录 KVX vs vLLM 的对比结果。
交付物：
- 固定 `fair_bench.py` 用例集与结果输出文件。
闭环校验：
- `python kvx/benchmarks/fair_bench.py --dtype float16 --repeats 5`
- 期望：输出包含 `ratio tokens_per_s` 且记录到 CSV。

## 里程碑完成判定
- 交付物全部落盘且闭环校验通过。
- 如需外部依赖（vLLM/TGI/TRT‑LLM），在文档中注明版本与获取方式。
