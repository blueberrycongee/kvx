import argparse
import json
import os
import statistics
import time


def mean_std(values):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stdev


def build_parser():
    parser = argparse.ArgumentParser(
        description="End-to-end vLLM generation benchmark for KVX baselines."
    )
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args, parser):
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.input_len <= 0:
        parser.error("--input-len must be > 0")
    if args.output_len <= 0:
        parser.error("--output-len must be > 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be > 0")
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    max_len = args.max_model_len
    if max_len is not None and max_len < args.input_len + args.output_len:
        parser.error("--max-model-len must cover input_len + output_len")


def default_output_path():
    return os.path.join(
        os.path.dirname(__file__),
        "results",
        "e2e_bench.json",
    )


def write_json(path, payload):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_prompt(tokenizer, prompt, target_len):
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not base_ids:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        base_ids = [eos_id if eos_id is not None else 0]
    ids = []
    while len(ids) < target_len:
        ids.extend(base_ids)
    ids = ids[:target_len]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    final_ids = tokenizer.encode(text, add_special_tokens=False)
    return text, len(final_ids)


def run_benchmark(args, output_path):
    import torch
    from vllm import LLM, SamplingParams
    try:
        from vllm.model_executor import set_random_seed
    except Exception:
        set_random_seed = None

    if set_random_seed is not None:
        set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    max_len = args.max_model_len
    if max_len is None:
        max_len = args.input_len + args.output_len + 8

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_len,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
    )
    tokenizer = llm.get_tokenizer()
    prompt_text, prompt_len = build_prompt(tokenizer, args.prompt, args.input_len)
    prompts = [prompt_text] * args.batch_size

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.output_len,
    )

    def run_once():
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        end = time.perf_counter()
        output_tokens = 0
        for out in outputs:
            if not out.outputs:
                continue
            seq = out.outputs[0]
            token_ids = getattr(seq, "token_ids", None)
            if token_ids is not None:
                output_tokens += len(token_ids)
        return end - start, output_tokens

    for _ in range(args.warmup):
        run_once()

    latencies = []
    output_tokens_list = []
    for _ in range(args.iters):
        lat, out_tokens = run_once()
        latencies.append(lat)
        output_tokens_list.append(out_tokens)

    total_prompt_tokens = prompt_len * args.batch_size
    per_iter = []
    tokens_per_s = []
    prompt_tokens_per_s = []
    output_tokens_per_s = []
    latency_ms = []
    for lat, out_tokens in zip(latencies, output_tokens_list):
        total_tokens = total_prompt_tokens + out_tokens
        tokens_per_s.append(total_tokens / lat)
        prompt_tokens_per_s.append(total_prompt_tokens / lat)
        output_tokens_per_s.append(out_tokens / lat if out_tokens else 0.0)
        latency_ms.append(lat * 1e3)
        per_iter.append({
            "latency_ms": lat * 1e3,
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": out_tokens,
            "tokens_per_s": total_tokens / lat,
        })

    tokens_mean, tokens_std = mean_std(tokens_per_s)
    prompt_mean, prompt_std = mean_std(prompt_tokens_per_s)
    output_mean, output_std = mean_std(output_tokens_per_s)
    lat_mean, lat_std = mean_std(latency_ms)

    env = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }
    try:
        import vllm
        env["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except Exception:
        env["vllm_version"] = "unknown"

    payload = {
        "dry_run": False,
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "iters": args.iters,
            "warmup": args.warmup,
            "seed": args.seed,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": max_len,
            "prompt_len": prompt_len,
            "prompt": args.prompt,
        },
        "env": env,
        "metrics": {
            "latency_ms_mean": lat_mean,
            "latency_ms_std": lat_std,
            "tokens_per_s_mean": tokens_mean,
            "tokens_per_s_std": tokens_std,
            "prompt_tokens_per_s_mean": prompt_mean,
            "prompt_tokens_per_s_std": prompt_std,
            "output_tokens_per_s_mean": output_mean,
            "output_tokens_per_s_std": output_std,
        },
        "per_iter": per_iter,
    }

    write_json(output_path, payload)
    print(
        "e2e,model={model},dtype={dtype},batch={batch},input_len={in_len},"
        "output_len={out_len},latency_ms_mean={lat:.3f},tokens_per_s_mean={tps:.2f}"
        .format(
            model=args.model,
            dtype=args.dtype,
            batch=args.batch_size,
            in_len=args.input_len,
            out_len=args.output_len,
            lat=lat_mean,
            tps=tokens_mean,
        )
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    output_path = args.output_json or default_output_path()
    if args.dry_run:
        payload = {
            "dry_run": True,
            "config": {
                "model": args.model,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "iters": args.iters,
                "warmup": args.warmup,
                "seed": args.seed,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "prompt": args.prompt,
            },
        }
        write_json(output_path, payload)
        print("e2e_bench dry-run: wrote", output_path)
        return

    run_benchmark(args, output_path)


if __name__ == "__main__":
    main()
