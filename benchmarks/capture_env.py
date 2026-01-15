import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(args):
    try:
        proc = subprocess.run(
            args,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def find_git_root(path):
    current = Path(path).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return None


def git_rev_parse(path):
    output = run_cmd(["git", "-C", str(path), "rev-parse", "HEAD"])
    if output:
        return output.splitlines()[0].strip()
    return "unknown"


def get_vllm_info():
    version = "unknown"
    commit = os.environ.get("VLLM_COMMIT") or os.environ.get("VLLM_GIT_SHA")
    source = "unknown"
    try:
        import vllm  # type: ignore

        version = getattr(vllm, "__version__", "unknown")
        source = str(Path(vllm.__file__).resolve())
        if commit is None:
            repo_root = find_git_root(source)
            if repo_root:
                commit = git_rev_parse(repo_root)
    except Exception:
        pass
    if commit is None:
        commit = "unknown"
    return {"version": version, "commit": commit, "source": source}


def get_torch_info():
    version = "unknown"
    cuda_version = "unknown"
    cuda_available = False
    try:
        import torch  # type: ignore

        version = getattr(torch, "__version__", "unknown")
        cuda_version = getattr(torch.version, "cuda", "unknown")
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pass
    return {"version": version, "cuda_available": cuda_available, "cuda": cuda_version}


def get_gpu_info():
    names = []
    driver_version = "unknown"
    smi_names = run_cmd(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if smi_names:
        names = [line.strip() for line in smi_names.splitlines() if line.strip()]
    smi_driver = run_cmd(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if smi_driver:
        driver_version = smi_driver.splitlines()[0].strip()
    return {
        "names": names,
        "count": len(names),
        "driver_version": driver_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def get_cuda_info():
    nvcc = run_cmd(["nvcc", "--version"]) or "unknown"
    return {"nvcc": nvcc, "torch_cuda": get_torch_info()["cuda"]}


def get_platform_info():
    is_wsl = "microsoft" in platform.release().lower()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "wsl": {
            "enabled": is_wsl,
            "distro": os.environ.get("WSL_DISTRO_NAME", "unknown") if is_wsl else "unknown",
        },
    }


def build_payload():
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": get_platform_info(),
        "gpu": get_gpu_info(),
        "cuda": get_cuda_info(),
        "torch": get_torch_info(),
        "vllm": get_vllm_info(),
    }


def main():
    parser = argparse.ArgumentParser(description="Capture KVX benchmark environment.")
    parser.add_argument("--output", help="Write JSON output to this path.")
    args = parser.parse_args()

    payload = build_payload()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
