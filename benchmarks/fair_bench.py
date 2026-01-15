import argparse
import csv
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch
from vllm import _custom_ops as ops
from vllm.model_executor import set_random_seed
from vllm.utils import create_kv_caches_with_random_flash


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_kvx_like_slot_mapping(seq_count, seq_len, block_size,
                                tokens_per_seq, device):
    max_blocks_per_seq = (seq_len + block_size - 1) // block_size
    start_pos = seq_len - tokens_per_seq
    if start_pos < 0:
        raise ValueError("tokens_per_seq cannot exceed seq_len")
    slots = []
    for seq in range(seq_count):
        for t in range(tokens_per_seq):
            token_pos = start_pos + t
            logical_block = token_pos // block_size
            block_offset = token_pos - logical_block * block_size
            block_id = seq * max_blocks_per_seq + logical_block
            slots.append(block_id * block_size + block_offset)
    return torch.tensor(slots, dtype=torch.long, device=device)


def run_vllm_once(seq_count, num_tokens, num_heads, head_dim, block_size,
                  num_blocks, seq_len, tokens_per_seq, dtype, kv_cache_dtype,
                  iters, warmup, device="cuda"):
    if kv_cache_dtype == "fp8" and head_dim % 16:
        raise ValueError("fp8 kv-cache requires head_dim to be a multiple of 16")

    set_random_seed(42)
    torch.set_default_device(device)

    key = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn_like(key)

    num_slots = block_size * num_blocks
    if num_tokens > num_slots:
        raise ValueError("num_tokens cannot exceed the total number of cache slots")

    slot_mapping = build_kvx_like_slot_mapping(
        seq_count, seq_len, block_size, tokens_per_seq, device
    )
    if slot_mapping.numel() != num_tokens:
        raise ValueError("slot_mapping length does not match num_tokens")
    if slot_mapping.max().item() >= num_blocks * block_size:
        raise ValueError("slot_mapping contains out-of-range slots")

    key_caches, value_caches = create_kv_caches_with_random_flash(
        num_blocks,
        block_size,
        1,  # num_layers
        num_heads,
        head_dim,
        kv_cache_dtype,
        dtype,
        device=device,
        cache_layout="NHD",
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    del key_caches, value_caches

    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    def function_under_test():
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    def run_cuda_benchmark(n_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            function_under_test()
            torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - start) / n_iters

    run_cuda_benchmark(warmup)
    lat = run_cuda_benchmark(iters)

    del key, value, key_cache, value_cache, slot_mapping
    torch.cuda.empty_cache()

    return lat


def mean_std(values):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stdev


def run_kvx_once(bin_path, seq_count, seq_len, tokens_per_seq, block_size,
                 num_heads, head_dim, iters, warmup, dtype):
    cmd = [
        bin_path,
        "--seq-count", str(seq_count),
        "--seq-len", str(seq_len),
        "--tokens-per-seq", str(tokens_per_seq),
        "--block-size", str(block_size),
        "--num-heads", str(num_heads),
        "--head-dim", str(head_dim),
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--dtype", dtype,
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    write_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("write,"):
            write_line = line
            break
    if write_line is None:
        raise RuntimeError("KVX output missing write line")
    parts = write_line.split(",")
    tokens_per_s = float(parts[10])
    gb_per_s = float(parts[11])
    ms_per = float(parts[9])
    return tokens_per_s, gb_per_s, ms_per


def parse_args():
    bench_root = Path(__file__).resolve().parent
    default_results = bench_root / "results" / "fair_bench_wsl.csv"
    default_bin = bench_root / "kvx_bench_wsl"
    parser = argparse.ArgumentParser(
        description="Fair KVX vs vLLM write benchmark (WSL)."
    )
    parser.add_argument("--output-csv", default=str(default_results))
    parser.add_argument("--dtype", default="float32",
                        choices=sorted(DTYPE_MAP.keys()))
    parser.add_argument("--kv-cache-dtype", default="auto",
                        choices=["auto", "fp8"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tokens-per-seq", type=int, default=1)
    parser.add_argument("--bin-path", default=str(default_bin))
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    kv_cache_dtype = args.kv_cache_dtype

    cases = [
        {"num_heads": 8, "seq_count": 1, "seq_len": 128},
        {"num_heads": 8, "seq_count": 8, "seq_len": 512},
        {"num_heads": 8, "seq_count": 32, "seq_len": 128},
        {"num_heads": 32, "seq_count": 1, "seq_len": 128},
        {"num_heads": 32, "seq_count": 8, "seq_len": 512},
        {"num_heads": 32, "seq_count": 16, "seq_len": 1024},
        {"num_heads": 8, "seq_count": 8, "seq_len": 512, "tokens_per_seq": 16},
        {"num_heads": 32, "seq_count": 8, "seq_len": 1024, "tokens_per_seq": 32},
    ]

    print("env")
    print(f"torch_version={torch.__version__}")
    print("vllm_backend=reshape_and_cache_flash layout=NHD")
    print(f"dtype={args.dtype} kv_cache_dtype={kv_cache_dtype} iters={args.iters} "
          f"warmup={args.warmup} repeats={args.repeats} "
          f"tokens_per_seq_default={args.tokens_per_seq}")
    print("cases=num_heads,seq_count,seq_len,block_size,head_dim,tokens_per_seq")
    print()

    rows = []
    for case in cases:
        num_heads = case["num_heads"]
        seq_count = case["seq_count"]
        seq_len = case["seq_len"]
        block_size = args.block_size
        head_dim = args.head_dim
        tokens_per_seq = case.get("tokens_per_seq", args.tokens_per_seq)
        num_tokens = seq_count * tokens_per_seq
        num_blocks = seq_count * math.ceil(seq_len / block_size)

        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        bytes_per_call = num_tokens * num_heads * head_dim * bytes_per_elem * 4

        vllm_latencies = []
        vllm_tokens = []
        vllm_gbs = []
        for _ in range(args.repeats):
            lat = run_vllm_once(
                seq_count=seq_count,
                num_tokens=num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                block_size=block_size,
                num_blocks=num_blocks,
                seq_len=seq_len,
                tokens_per_seq=tokens_per_seq,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                iters=args.iters,
                warmup=args.warmup,
            )
            vllm_latencies.append(lat)
            vllm_tokens.append(num_tokens / lat)
            vllm_gbs.append(bytes_per_call / lat / 1e9)

        vllm_tokens_mean, vllm_tokens_std = mean_std(vllm_tokens)
        vllm_gbs_mean, vllm_gbs_std = mean_std(vllm_gbs)
        vllm_lat_mean, vllm_lat_std = mean_std(vllm_latencies)

        kvx_tokens = []
        kvx_gbs = []
        kvx_ms = []
        for _ in range(args.repeats):
            tokens_per_s, gb_per_s, ms_per = run_kvx_once(
                args.bin_path,
                seq_count,
                seq_len,
                tokens_per_seq,
                block_size,
                num_heads,
                head_dim,
                args.iters,
                args.warmup,
                args.dtype,
            )
            kvx_tokens.append(tokens_per_s)
            kvx_gbs.append(gb_per_s)
            kvx_ms.append(ms_per)

        kvx_tokens_mean, kvx_tokens_std = mean_std(kvx_tokens)
        kvx_gbs_mean, kvx_gbs_std = mean_std(kvx_gbs)
        kvx_ms_mean, kvx_ms_std = mean_std(kvx_ms)

        ratio_tokens = kvx_tokens_mean / vllm_tokens_mean if vllm_tokens_mean > 0 else float("nan")
        ratio_gbs = kvx_gbs_mean / vllm_gbs_mean if vllm_gbs_mean > 0 else float("nan")

        print("case")
        print(f"num_heads={num_heads} seq_count={seq_count} seq_len={seq_len} "
              f"block_size={block_size} head_dim={head_dim} "
              f"tokens_per_seq={tokens_per_seq} num_blocks={num_blocks}")
        print(f"vllm latency_ms_mean={(vllm_lat_mean * 1e3):.3f} "
              f"latency_ms_std={(vllm_lat_std * 1e3):.3f}")
        print(f"vllm tokens_per_s_mean={vllm_tokens_mean:.2f} "
              f"tokens_per_s_std={vllm_tokens_std:.2f}")
        print(f"vllm gb_per_s_mean={vllm_gbs_mean:.2f} "
              f"gb_per_s_std={vllm_gbs_std:.2f}")
        print(f"kvx ms_per_mean={kvx_ms_mean:.3f} ms_per_std={kvx_ms_std:.3f}")
        print(f"kvx tokens_per_s_mean={kvx_tokens_mean:.2f} "
              f"tokens_per_s_std={kvx_tokens_std:.2f}")
        print(f"kvx gb_per_s_mean={kvx_gbs_mean:.2f} "
              f"gb_per_s_std={kvx_gbs_std:.2f}")
        print(f"ratio tokens_per_s={ratio_tokens:.2f}x gb_per_s={ratio_gbs:.2f}x")
        print()

        rows.append({
            "dtype": args.dtype,
            "num_heads": num_heads,
            "seq_count": seq_count,
            "seq_len": seq_len,
            "tokens_per_seq": tokens_per_seq,
            "block_size": block_size,
            "head_dim": head_dim,
            "num_blocks": num_blocks,
            "vllm_latency_ms_mean": vllm_lat_mean * 1e3,
            "vllm_latency_ms_std": vllm_lat_std * 1e3,
            "vllm_tokens_per_s_mean": vllm_tokens_mean,
            "vllm_tokens_per_s_std": vllm_tokens_std,
            "vllm_gb_per_s_mean": vllm_gbs_mean,
            "vllm_gb_per_s_std": vllm_gbs_std,
            "kvx_ms_per_mean": kvx_ms_mean,
            "kvx_ms_per_std": kvx_ms_std,
            "kvx_tokens_per_s_mean": kvx_tokens_mean,
            "kvx_tokens_per_s_std": kvx_tokens_std,
            "kvx_gb_per_s_mean": kvx_gbs_mean,
            "kvx_gb_per_s_std": kvx_gbs_std,
            "ratio_tokens_per_s": ratio_tokens,
            "ratio_gb_per_s": ratio_gbs,
        })

    output_csv = args.output_csv
    output_dir = output_csv.rsplit("/", 1)[0]
    subprocess.run(["mkdir", "-p", output_dir], check=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
