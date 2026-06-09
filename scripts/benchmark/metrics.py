#!/usr/bin/env python3
"""Metrics collection for LLM inference benchmarking.

Captures:
  - TTFT (Time To First Token): wall-clock from request sent to first token received
  - TPOT (Time Per Output Token): average time between consecutive output tokens
  - E2E latency: total wall-clock from request sent to final token
  - tokens/s: total tokens / E2E time
  - Peak GPU memory: via nvidia-smi polling
"""

import time
import json
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RequestMetrics:
    """Per-request timing metrics."""
    ttft: float = 0.0          # time to first token (seconds)
    tpot: list[float] = field(default_factory=list)  # inter-token latencies
    e2e: float = 0.0           # end-to-end latency
    input_tokens: int = 0
    output_tokens: int = 0
    first_token: str = ""
    error: Optional[str] = None

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.e2e if self.e2e > 0 else 0.0

    @property
    def mean_tpot(self) -> float:
        """Mean TPOT from fine-grained timing, or estimated from (E2E - TTFT) / (N-1)."""
        non_zero = [x for x in self.tpot if x > 1e-6]
        if non_zero:
            return float(np.mean(non_zero))
        # Fallback: estimate from E2E and TTFT
        if self.output_tokens > 1:
            return (self.e2e - self.ttft) / (self.output_tokens - 1)
        return 0.0

    def to_dict(self) -> dict:
        non_zero = [x for x in self.tpot if x > 1e-6]
        tpot_data = non_zero if non_zero else (
            [(self.e2e - self.ttft) / max(self.output_tokens - 1, 1)] if self.output_tokens > 1 else [0.0]
        )
        return {
            "ttft_s": round(self.ttft, 4),
            "tpot_mean_ms": round(self.mean_tpot * 1000, 2),
            "tpot_p50_ms": round(float(np.percentile(tpot_data, 50)) * 1000, 2),
            "tpot_p95_ms": round(float(np.percentile(tpot_data, 95)) * 1000, 2),
            "e2e_s": round(self.e2e, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }


@dataclass
class BenchmarkResult:
    """Aggregated results for a single workload configuration."""
    config: dict
    requests: list[RequestMetrics]
    gpu_memory_mb: list[float] = field(default_factory=list)
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_pct: list[float] = field(default_factory=list)
    server_metrics: list[dict] = field(default_factory=list)

    @property
    def valid_requests(self) -> list[RequestMetrics]:
        return [r for r in self.requests if r.error is None]

    @property
    def ttft_mean(self) -> float:
        return float(np.mean([r.ttft for r in self.valid_requests]))

    @property
    def ttft_p50(self) -> float:
        return float(np.percentile([r.ttft for r in self.valid_requests], 50))

    @property
    def ttft_p95(self) -> float:
        return float(np.percentile([r.ttft for r in self.valid_requests], 95))

    @property
    def tpot_mean(self) -> float:
        # Use per-request mean_tpot (which includes fallback estimation)
        vals = [r.mean_tpot for r in self.valid_requests]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def tpot_p50(self) -> float:
        vals = [r.mean_tpot for r in self.valid_requests]
        return float(np.percentile(vals, 50)) if vals else 0.0

    @property
    def tpot_p95(self) -> float:
        vals = [r.mean_tpot for r in self.valid_requests]
        return float(np.percentile(vals, 95)) if vals else 0.0

    @property
    def e2e_mean(self) -> float:
        return float(np.mean([r.e2e for r in self.valid_requests]))

    @property
    def tokens_per_second_total(self) -> float:
        total_out = sum(r.output_tokens for r in self.valid_requests)
        total_time = sum(r.e2e for r in self.valid_requests)
        return total_out / total_time if total_time > 0 else 0.0

    @property
    def peak_gpu_memory_mb(self) -> float:
        return max(self.gpu_memory_mb) if self.gpu_memory_mb else 0.0

    @property
    def peak_gpu_memory_utilization_pct(self) -> float:
        if self.gpu_memory_total_mb <= 0:
            return 0.0
        return self.peak_gpu_memory_mb / self.gpu_memory_total_mb * 100.0

    @property
    def mean_gpu_utilization_pct(self) -> float:
        return float(np.mean(self.gpu_utilization_pct)) if self.gpu_utilization_pct else 0.0

    @property
    def kv_block_active_peak(self) -> int:
        return max((m.get("kv_cache", {}).get("block_active", 0) for m in self.server_metrics),
                   default=0)

    @property
    def kv_block_cached_idle_peak(self) -> int:
        return max((m.get("kv_cache", {}).get("block_cached_idle", 0)
                    for m in self.server_metrics), default=0)

    @property
    def kv_pool_utilization_peak(self) -> float:
        return max((m.get("kv_cache", {}).get("pool_utilization", 0.0)
                    for m in self.server_metrics), default=0.0)

    @property
    def prefix_lookup_hits(self) -> int:
        return max((m.get("prefix_cache", {}).get("lookup_hits", 0)
                    for m in self.server_metrics), default=0)

    @property
    def prefix_lookup_misses(self) -> int:
        return max((m.get("prefix_cache", {}).get("lookup_misses", 0)
                    for m in self.server_metrics), default=0)

    @property
    def prefix_evictions(self) -> int:
        return max((m.get("prefix_cache", {}).get("evictions", 0)
                    for m in self.server_metrics), default=0)

    def summary(self) -> dict:
        v = self.valid_requests
        if not v:
            return {
                "error": "no valid requests",
                "request_errors": [r.error for r in self.requests if r.error],
            }
        output_tokens = [r.output_tokens for r in v]
        return {
            "num_requests": len(v),
            "num_errors": len(self.requests) - len(v),
            "output_tokens_mean": round(float(np.mean(output_tokens)), 2),
            "output_tokens_min": min(output_tokens),
            "ttft_mean_ms": round(self.ttft_mean * 1000, 2),
            "ttft_p50_ms": round(self.ttft_p50 * 1000, 2),
            "ttft_p95_ms": round(self.ttft_p95 * 1000, 2),
            "tpot_mean_ms": round(self.tpot_mean * 1000, 2),
            "tpot_p50_ms": round(self.tpot_p50 * 1000, 2),
            "tpot_p95_ms": round(self.tpot_p95 * 1000, 2),
            "e2e_mean_s": round(self.e2e_mean, 4),
            "tokens_per_second": round(self.tokens_per_second_total, 2),
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 0),
            "peak_gpu_memory_utilization_pct": round(self.peak_gpu_memory_utilization_pct, 2),
            "mean_gpu_utilization_pct": round(self.mean_gpu_utilization_pct, 2),
            "kv_block_active_peak": self.kv_block_active_peak,
            "kv_block_cached_idle_peak": self.kv_block_cached_idle_peak,
            "kv_pool_utilization_peak": round(self.kv_pool_utilization_peak, 4),
            "prefix_lookup_hits": self.prefix_lookup_hits,
            "prefix_lookup_misses": self.prefix_lookup_misses,
            "prefix_evictions": self.prefix_evictions,
            "per_request": [r.to_dict() for r in v],
            "request_errors": [r.error for r in self.requests if r.error],
        }


def poll_gpu_memory(stop_event: threading.Event, device_id: int = 0,
                    interval: float = 0.1) -> list[float]:
    """Poll GPU memory usage until stop_event is set. Returns list of MB readings."""
    readings: list[float] = []
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", f"--id={device_id}"],
                text=True, timeout=5,
            )
            readings.append(float(out.strip()))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError,
                FileNotFoundError):
            pass
        stop_event.wait(interval)
    return readings


def poll_gpu_stats(stop_event: threading.Event, device_id: int = 0,
                   interval: float = 0.1) -> tuple[list[float], float, list[float]]:
    """Poll memory used/total and GPU utilization until stop_event is set."""
    memory_used: list[float] = []
    utilization: list[float] = []
    memory_total = 0.0
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits", f"--id={device_id}"],
                text=True, timeout=5,
            )
            parts = [p.strip() for p in out.strip().split(",")]
            if len(parts) >= 3:
                memory_used.append(float(parts[0]))
                memory_total = float(parts[1])
                utilization.append(float(parts[2]))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError,
                FileNotFoundError):
            pass
        stop_event.wait(interval)
    return memory_used, memory_total, utilization
