import json
import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "benchmarks" / "capture_env.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)

    if payload.get("schema_version") != 1:
        print("env_capture_test failed: schema_version != 1")
        return 1

    required_top = ["timestamp_utc", "gpu", "cuda", "torch", "vllm"]
    missing = [key for key in required_top if key not in payload]
    if missing:
        print(f"env_capture_test failed: missing keys {missing}")
        return 1

    gpu = payload["gpu"]
    cuda = payload["cuda"]
    torch_info = payload["torch"]
    vllm_info = payload["vllm"]

    for key in ["names", "count", "driver_version"]:
        if key not in gpu:
            print(f"env_capture_test failed: gpu missing {key}")
            return 1
    if not isinstance(gpu["names"], list):
        print("env_capture_test failed: gpu.names not a list")
        return 1
    if gpu["count"] != len(gpu["names"]):
        print("env_capture_test failed: gpu.count mismatch")
        return 1

    for key in ["nvcc", "torch_cuda"]:
        if key not in cuda:
            print(f"env_capture_test failed: cuda missing {key}")
            return 1

    for key in ["version", "cuda_available"]:
        if key not in torch_info:
            print(f"env_capture_test failed: torch missing {key}")
            return 1

    for key in ["version", "commit", "source"]:
        if key not in vllm_info:
            print(f"env_capture_test failed: vllm missing {key}")
            return 1

    print("env_capture_test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
