#!/usr/bin/env bash
set -eo pipefail

echo "[vllm_build] $(date) starting"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KVX_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_ROOT="${VLLM_ROOT:-${1:-}}"

if [ -z "$VLLM_ROOT" ]; then
  echo "[vllm_build] missing VLLM_ROOT (set env var or pass as arg)"
  echo "[vllm_build] usage: VLLM_ROOT=/path/to/vllm $0"
  exit 1
fi

PYTHON="${PYTHON:-$(command -v python)}"

if [ ! -x "$PYTHON" ]; then
  echo "[vllm_build] missing python at $PYTHON"
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "[vllm_build] missing cmake in PATH"
  exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
  echo "[vllm_build] missing ninja in PATH"
  exit 1
fi

if [ ! -d "$VLLM_ROOT" ]; then
  echo "[vllm_build] VLLM_ROOT not found: $VLLM_ROOT"
  exit 1
fi

"$PYTHON" -m pip --version
cd "$VLLM_ROOT"

MAX_JOBS=1 NVCC_THREADS=1 VLLM_VERSION_OVERRIDE=0.0.0+kvx PIP_NO_BUILD_ISOLATION=1 \
  CMAKE_ARGS="-DVLLM_ENABLE_KVX=ON -DVLLM_KVX_ROOT=$KVX_ROOT -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler" \
  "$PYTHON" -m pip install -e . --no-deps --no-build-isolation
