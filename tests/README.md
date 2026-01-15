# KVX Tests (WSL)

## Build

```bash
nvcc -std=c++17 -O3 -arch=sm_86 -I. \
  kvx_abi.c \
  kernels/kvx_paged_kv.cu \
  tests/kvx_kernel_test.cu \
  -o tests/kvx_kernel_test
```

## Run

```bash
tests/kvx_kernel_test
```
