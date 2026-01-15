# Fair Benchmark Results (WSL RTX 3060 Laptop)

This summary compares vLLM `reshape_and_cache_flash` (NHD layout) with the
KVX write kernel under the same shapes in WSL. It is a kernel-level write
benchmark, not an end-to-end inference comparison.

## Environment
- See `benchmarks/results/fair_bench_wsl_env.txt` for GPU/driver/CUDA/WSL.
- Benchmark defaults: `warmup=10`, `iters=50`, `repeats=10`.

## Summary (ratio = KVX / vLLM)
Aggregates across all cases in the CSV (decode + prefill).

### float32
- tokens/s: mean 4.90x, median 5.49x, min 1.14x, max 6.03x
- GB/s: mean 4.56x, median 5.11x, min 1.06x, max 5.62x

### float16
- tokens/s: mean 5.01x, median 5.23x, min 2.04x, max 6.48x
- GB/s: mean 4.67x, median 4.87x, min 1.90x, max 6.03x

### bfloat16
- tokens/s: mean 4.98x, median 5.35x, min 2.25x, max 6.19x
- GB/s: mean 4.64x, median 4.98x, min 2.10x, max 5.77x

## Fairness notes
- Same layouts: vLLM uses NHD cache layout; KVX uses NHD layout in the write
  kernel.
- Same shapes and slot mapping: per-case `num_heads`, `seq_count`, `seq_len`,
  `block_size`, `head_dim` with last-token slot mapping per sequence.
- Same environment: both run inside the same WSL distro and GPU.

## Notes
- Decode-style mean tokens/s ratios: float32 5.47x, float16 5.26x,
  bfloat16 5.61x.
- Prefill-style mean tokens/s ratios: float32 3.20x, float16 4.26x,
  bfloat16 3.08x.

## Reproduce
```bash
source /path/to/venv/bin/activate
python benchmarks/fair_bench.py \
  --dtype=float16 --iters=50 --warmup=10 --repeats=10
```

Results are written to `benchmarks/results/fair_bench_wsl_*.csv`.

To include multi-token (prefill-style) writes, pass `--tokens-per-seq` or use
the cases in `fair_bench.py` that specify `tokens_per_seq`.
