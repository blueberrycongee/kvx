# vLLM KVX Patch

This directory contains the minimal patch needed to enable KVX cache write
integration in vLLM.

## Prereqs
- vLLM checkout at the pinned commit used for KVX benchmarking:
  - tag: v0.14.0rc0-508-g0346396e9
  - commit: 0346396e94106edcbce13e083da172c119d0aa17

## Apply
```bash
./apply_vllm_patch.sh --check /path/to/vllm
./apply_vllm_patch.sh /path/to/vllm
```

## Build
From the vLLM repo root:
```bash
CMAKE_ARGS="-DVLLM_ENABLE_KVX=ON -DVLLM_KVX_ROOT=/path/to/kvx" \
  pip install -e .
```

## Runtime
Enable the KVX cache write path:
```bash
export VLLM_USE_KVX_CACHE_WRITE=1
```

## Notes
- The patch adds a CUDA-only KVX write kernel binding and a runtime switch.
- If `VLLM_USE_KVX_CACHE_WRITE=0`, vLLM uses the default kernel.
