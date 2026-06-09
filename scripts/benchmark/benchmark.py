#!/usr/bin/env python3
"""ccInfer Benchmark Harness.

Measures TTFT, TPOT, E2E latency, tokens/s, and peak GPU memory across
configurable prompt lengths, output lengths, and concurrency levels.

Usage:
    python scripts/benchmark/benchmark.py \
        --model-path models/qwen3-0.6B \
        --binary build/src/ccinfer-server \
        --output-dir docs/benchmarks/$(date +%Y-%m-%d)

    # Quick smoke test:
    python scripts/benchmark/benchmark.py --quick
"""

import argparse
import csv
import json
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import requests

from harness import ServerHarness
from metrics import RequestMetrics, BenchmarkResult, poll_gpu_stats
from workload import full_workload_matrix, generate_prompt, get_tokenizer, quick_workload_matrix


BENCHMARK_CONFIGS = {
    "baseline": {"server_args": [], "admission": "continuous"},
    "static_admission": {"server_args": ["--prefill-chunk-size", "0"], "admission": "static"},
    "no_prefill_chunk": {"server_args": ["--prefill-chunk-size", "0"], "admission": "continuous"},
    "chunk_128": {"server_args": ["--prefill-chunk-size", "128"], "admission": "continuous"},
    "chunk_256": {"server_args": ["--prefill-chunk-size", "256"], "admission": "continuous"},
    "chunk_512": {"server_args": ["--prefill-chunk-size", "512"], "admission": "continuous"},
    "chunk_1024": {"server_args": ["--prefill-chunk-size", "1024"], "admission": "continuous"},
}


# ---------------------------------------------------------------------------
# SSE-based chat completion with timing
# ---------------------------------------------------------------------------

def chat_completion_timed(base_url: str, prompt: str, input_tokens: int, max_tokens: int,
                          temperature: float = 0.0, timeout: float = 120.0) -> RequestMetrics:
    """Send a chat completion request and measure TTFT/TPOT/E2E from SSE stream."""
    metrics = RequestMetrics()
    metrics.input_tokens = input_tokens
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    t_start = time.perf_counter()
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=timeout,
        )
        response.raise_for_status()

        t_first = None
        t_prev = t_start
        token_count = 0

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            now = time.perf_counter()
            if t_first is None:
                t_first = now
                metrics.ttft = now - t_start
                metrics.first_token = data.get("token", "")
                t_prev = now
            else:
                metrics.tpot.append(now - t_prev)
                t_prev = now
            if "token_id" in data:
                token_count += 1

            if data.get("done"):
                break

        metrics.e2e = time.perf_counter() - t_start
        metrics.output_tokens = token_count
    except Exception as e:
        metrics.error = str(e)

    return metrics


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_workload(harness: ServerHarness, prompts: list[str], input_tokens: int, max_tokens: int,
                 concurrency: int, warmup: int = 3, measured: int = 5,
                 admission: str = "continuous") -> BenchmarkResult:
    """Run one workload.

    `concurrency` is the number of simultaneously in-flight requests.
    `warmup` and `measured` are batch counts, so measured request count is
    measured * concurrency.
    """
    config = {
        "prompt": prompts[0][:80] + ("..." if len(prompts[0]) > 80 else ""),
        "prompt_len": input_tokens,
        "output_len": max_tokens,
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "admission": admission,
        "max_context_len": 2048,
        "estimated_live_tokens": concurrency * (input_tokens + max_tokens),
        "no_page_reserved_tokens": concurrency * 2048,
        "no_page_token_utilization_estimate": round(
            (input_tokens + max_tokens) / 2048, 4
        ),
    }
    all_metrics: list[RequestMetrics] = []

    # GPU memory polling
    stop_poll = threading.Event()
    mem_thread = threading.Thread(target=lambda: None)  # placeholder type hint
    mem_readings: list[float] = []
    mem_total = 0.0
    gpu_utilization: list[float] = []
    server_metrics: list[dict] = []

    def _poll():
        nonlocal mem_readings, mem_total, gpu_utilization
        mem_readings, mem_total, gpu_utilization = poll_gpu_stats(stop_poll)

    mem_thread = threading.Thread(target=_poll, daemon=True)
    mem_thread.start()

    def _poll_server_metrics():
        while not stop_poll.is_set():
            try:
                r = requests.get(f"{harness.base_url}/metrics", timeout=2)
                if r.status_code == 200:
                    server_metrics.append(r.json())
            except (requests.ConnectionError, requests.Timeout, ValueError):
                pass
            stop_poll.wait(0.1)

    server_metrics_thread = threading.Thread(target=_poll_server_metrics, daemon=True)
    server_metrics_thread.start()

    def submit_group(executor: ThreadPoolExecutor, count: int):
        batch_prompts = [next_prompt() for _ in range(count)]
        return [
            executor.submit(chat_completion_timed, harness.base_url, batch_prompts[i], input_tokens,
                            max_tokens)
            for i in range(count)
        ]

    prompt_cursor = 0

    def next_prompt() -> str:
        nonlocal prompt_cursor
        if prompt_cursor >= len(prompts):
            raise RuntimeError("not enough prompt variants for workload")
        prompt = prompts[prompt_cursor]
        prompt_cursor += 1
        return prompt

    def run_one_batch():
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            if admission == "static":
                futures = submit_group(executor, concurrency)
                for f in as_completed(futures):
                    all_metrics.append(f.result())
            else:
                futures = submit_group(executor, concurrency)
                for f in as_completed(futures):
                    all_metrics.append(f.result())

    def run_batches(batch_count: int):
        for _ in range(batch_count):
            run_one_batch()

    # Warmup
    if warmup > 0:
        run_batches(warmup)
        all_metrics.clear()  # discard warmup metrics

    # Measured
    run_batches(measured)

    stop_poll.set()
    mem_thread.join(timeout=2)
    server_metrics_thread.join(timeout=2)

    return BenchmarkResult(config=config, requests=all_metrics, gpu_memory_mb=mem_readings,
                           gpu_memory_total_mb=mem_total, gpu_utilization_pct=gpu_utilization,
                           server_metrics=server_metrics)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "config",
        "prompt_len",
        "output_len",
        "concurrency",
        "admission",
        "warmup_batches",
        "measured_batches",
        "num_requests",
        "num_errors",
        "output_tokens_mean",
        "output_tokens_min",
        "ttft_mean_ms",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "tpot_mean_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "e2e_mean_s",
        "tokens_per_second",
        "peak_gpu_memory_mb",
        "peak_gpu_memory_utilization_pct",
        "mean_gpu_utilization_pct",
        "kv_block_active_peak",
        "kv_block_cached_idle_peak",
        "kv_pool_utilization_peak",
        "prefix_lookup_hits",
        "prefix_lookup_misses",
        "prefix_evictions",
        "estimated_live_tokens",
        "no_page_reserved_tokens",
        "no_page_token_utilization_estimate",
        "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_result_files(output_dir: Path, config_name: str, results: list[BenchmarkResult],
                       all_summaries: list[dict]) -> None:
    results_json = output_dir / f"{config_name}.json"
    with open(results_json, "w") as f:
        json.dump([{"config": r.config, "summary": r.summary()} for r in results],
                  f, indent=2)

    csv_path = output_dir / f"{config_name}.csv"
    write_summary_csv(csv_path, [{"config": config_name, **r.config, **r.summary()}
                                 for r in results])

    summary_csv = output_dir / "summary.csv"
    write_summary_csv(summary_csv, all_summaries)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ccInfer Benchmark Harness")
    parser.add_argument("--model-path", default="models/qwen3-0.6B")
    parser.add_argument("--binary", default="build/src/ccinfer-server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test with reduced workloads")
    parser.add_argument("--config", choices=[*BENCHMARK_CONFIGS.keys(), "all"], default="all",
                        help="Server config to benchmark")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--restart-each-workload", action="store_true",
                        help="Restart server before every workload to isolate KV/prefix cache state")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    binary = Path(args.binary).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"docs/benchmarks/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for exact prompt lengths
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(str(model_path))

    workloads = quick_workload_matrix() if args.quick else full_workload_matrix()
    config_names = list(BENCHMARK_CONFIGS) if args.config == "all" else [args.config]

    print(f"Model: {model_path}")
    print(f"Binary: {binary}")
    print(f"Output: {output_dir}")
    print(f"Configs: {', '.join(config_names)}")
    print(f"Workloads: {len(workloads)}")
    print()

    all_summaries: list[dict] = []

    for config_name in config_names:
        config_spec = BENCHMARK_CONFIGS[config_name]
        server_args = config_spec["server_args"]
        admission = config_spec["admission"]
        config_results: list[BenchmarkResult] = []
        shared_server: Optional[ServerHarness] = None
        if not args.restart_each_workload:
            print(f"Starting server config={config_name} args={server_args} admission={admission}")
            shared_server = ServerHarness(str(binary), str(model_path), port=args.port,
                                          extra_args=server_args)
            shared_server.start()
            print(f"Server ready at {shared_server.base_url}\n")

        try:
            for i, wl in enumerate(workloads):
                concurrency = wl["concurrency"]
                prompt_count = (args.warmup + args.measured) * concurrency
                prompts = [
                    generate_prompt(wl["prompt_len"], tokenizer,
                                    variant=i * prompt_count + j)
                    for j in range(prompt_count)
                ]
                token_count = len(tokenizer.encode(prompts[0], add_special_tokens=False))

                if any(len(tokenizer.encode(p, add_special_tokens=False)) != wl["prompt_len"]
                       for p in prompts):
                    raise RuntimeError(
                        f"prompt generator returned non-{wl['prompt_len']} token prompt"
                    )

                print(f"[{config_name} {i+1}/{len(workloads)}] prompt_len={wl['prompt_len']}"
                      f", output_len={wl['output_len']}, concurrency={concurrency}")

                server = shared_server
                if server is None:
                    print(f"  Starting isolated server args={server_args} admission={admission}")
                    server = ServerHarness(str(binary), str(model_path), port=args.port,
                                           extra_args=server_args)
                    server.start()

                try:
                    result = run_workload(
                        server, prompts, token_count, wl["output_len"], concurrency,
                        warmup=args.warmup, measured=args.measured, admission=admission,
                    )
                    result.config["config"] = config_name
                    result.config["warmup_batches"] = args.warmup
                    result.config["measured_batches"] = args.measured

                    summary = result.summary()
                    summary_with_config = {"config": config_name, **result.config, **summary}
                    all_summaries.append(summary_with_config)

                    if "error" in summary:
                        print(f"  ERROR: {summary['error']}")
                    else:
                        print(f"  TTFT: mean={summary['ttft_mean_ms']}ms"
                              f" p50={summary['ttft_p50_ms']}ms"
                              f" p95={summary['ttft_p95_ms']}ms")
                        print(f"  TPOT: mean={summary['tpot_mean_ms']}ms"
                              f" p50={summary['tpot_p50_ms']}ms"
                              f" p95={summary['tpot_p95_ms']}ms")
                        print(f"  tokens/s: {summary['tokens_per_second']}")
                        print(f"  output tokens: mean={summary['output_tokens_mean']}"
                              f" min={summary['output_tokens_min']}")
                        print(f"  GPU memory peak: {summary['peak_gpu_memory_mb']:.0f}MB"
                              f" ({summary['peak_gpu_memory_utilization_pct']}%)")
                        print(f"  GPU utilization mean: {summary['mean_gpu_utilization_pct']}%")
                        print(f"  KV blocks: active_peak={summary['kv_block_active_peak']}"
                              f" cached_idle_peak={summary['kv_block_cached_idle_peak']}"
                              f" pool_peak={summary['kv_pool_utilization_peak']}")
                        print(f"  Prefix cache: hits={summary['prefix_lookup_hits']}"
                              f" misses={summary['prefix_lookup_misses']}"
                              f" evictions={summary['prefix_evictions']}")
                    print()

                    config_results.append(result)
                    write_result_files(output_dir, config_name, config_results, all_summaries)
                finally:
                    if shared_server is None:
                        print("  Stopping isolated server...")
                        server.stop()
        finally:
            if shared_server is not None:
                print(f"Stopping server config={config_name}...")
                shared_server.stop()

        write_result_files(output_dir, config_name, config_results, all_summaries)
        print(f"Results written to {output_dir / f'{config_name}.json'}")
        print(f"CSV written to {output_dir / f'{config_name}.csv'}\n")

    summary_csv = output_dir / "summary.csv"
    write_summary_csv(summary_csv, all_summaries)
    print(f"Combined CSV written to {summary_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
