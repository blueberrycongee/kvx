import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    bin_path = repo_root / "benchmarks" / "kvx_bench_wsl"
    cmd = [
        str(bin_path),
        "--seq-count", "2",
        "--seq-len", "8",
        "--tokens-per-seq", "2",
        "--block-size", "4",
        "--num-heads", "2",
        "--head-dim", "4",
        "--iters", "1",
        "--warmup", "1",
        "--dtype=float32",
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    write_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("write,"):
            write_line = line.strip()
            break
    if write_line is None:
        print("kvx_bench_cli_test failed: missing write line")
        return 1
    parts = write_line.split(",")
    if len(parts) < 13:
        print("kvx_bench_cli_test failed: unexpected write line format")
        return 1
    seq_count = int(parts[1])
    seq_len = int(parts[2])
    block_size = int(parts[3])
    tokens_per_seq = int(parts[12])
    if seq_count != 2 or seq_len != 8 or block_size != 4 or tokens_per_seq != 2:
        print("kvx_bench_cli_test failed: args not reflected in output")
        return 1
    print("kvx_bench_cli_test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
