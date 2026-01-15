import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cmd(cmd):
    return subprocess.run(cmd, text=True, capture_output=True)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "benchmarks" / "e2e_bench.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "out.json")
        cmd = [
            sys.executable,
            str(script_path),
            "--dry-run",
            "--model", "dummy-model",
            "--dtype", "float16",
            "--batch-size", "4",
            "--input-len", "16",
            "--output-len", "8",
            "--iters", "2",
            "--warmup", "1",
            "--output-json", out_json,
        ]
        proc = run_cmd(cmd)
        if proc.returncode != 0:
            print("kvx_e2e_bench_cli_test failed: dry-run exit code != 0")
            print(proc.stdout)
            print(proc.stderr)
            return 1
        with open(out_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config = payload.get("config", {})
        if config.get("model") != "dummy-model":
            print("kvx_e2e_bench_cli_test failed: model not reflected")
            return 1
        if config.get("dtype") != "float16":
            print("kvx_e2e_bench_cli_test failed: dtype not reflected")
            return 1
        if config.get("batch_size") != 4:
            print("kvx_e2e_bench_cli_test failed: batch_size not reflected")
            return 1
        if payload.get("dry_run") is not True:
            print("kvx_e2e_bench_cli_test failed: dry_run missing")
            return 1

    bad_cmd = [
        sys.executable,
        str(script_path),
        "--dry-run",
        "--model", "dummy-model",
        "--input-len", "0",
    ]
    proc = run_cmd(bad_cmd)
    if proc.returncode == 0:
        print("kvx_e2e_bench_cli_test failed: invalid args did not error")
        return 1

    print("kvx_e2e_bench_cli_test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
