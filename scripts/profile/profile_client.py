#!/usr/bin/env python3
"""Small HTTP/SSE load generator for profiling ccInfer.

This is intentionally lighter than scripts/benchmark/benchmark.py. Profilers
wrap the server process; this client only drives a fixed amount of traffic so
the captured trace stays short and readable.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def make_prompt(prompt_len: int, variant: int) -> str:
    # Qwen-style tokenizers usually encode " a" as one stable token. The
    # server-side tokenizer is simplified, so this is a profiling workload,
    # not a correctness/benchmark length guarantee.
    markers = " " + chr(ord("a") + (variant % 26))
    return (markers + " a" * max(prompt_len - 1, 1)).strip()


def run_one(base_url: str, prompt_len: int, max_tokens: int, variant: int, timeout: float) -> dict:
    payload = {
        "messages": [{"role": "user", "content": make_prompt(prompt_len, variant)}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    first = None
    tokens = 0
    with requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            now = time.perf_counter()
            if first is None and "token_id" in data:
                first = now
            if "token_id" in data:
                tokens += 1
            if data.get("done"):
                break
    t1 = time.perf_counter()
    return {
        "variant": variant,
        "tokens": tokens,
        "ttft_ms": None if first is None else round((first - t0) * 1000, 2),
        "e2e_ms": round((t1 - t0) * 1000, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Drive a small ccInfer profiling workload")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    total = max(args.requests, args.concurrency)
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_one,
                args.base_url,
                args.prompt_len,
                args.max_tokens,
                i,
                args.timeout,
            )
            for i in range(total)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r["variant"])
    print(json.dumps({"config": vars(args), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
