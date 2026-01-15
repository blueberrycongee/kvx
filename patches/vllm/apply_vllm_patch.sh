#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/kvx.patch"

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
  shift
fi

VLLM_ROOT="${VLLM_ROOT:-${1:-}}"
if [[ -z "$VLLM_ROOT" ]]; then
  echo "usage: $0 [--check] /path/to/vllm" >&2
  echo "       VLLM_ROOT=/path/to/vllm $0 [--check]" >&2
  exit 2
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "patch not found: $PATCH_FILE" >&2
  exit 1
fi

cd "$VLLM_ROOT"

if [[ $CHECK_ONLY -eq 1 ]]; then
  git apply --check "$PATCH_FILE"
  echo "patch check: OK"
  exit 0
fi

git apply "$PATCH_FILE"
echo "patch applied: $PATCH_FILE"
