"""
Performance metrics for latency and throughput evaluation.

Implements metrics following VectorDBBench and Qdrant benchmark methodologies.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.types import PerformanceMetrics


def compute_latency_percentiles(
    latencies_ms: List[float],
) -> Dict[str, float]:
    """
    Compute latency percentiles from a list of latencies.
    """
    if not latencies_ms:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "mean": 0, "std": 0, "min": 0, "max": 0}
        
    latencies = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(latencies, 50)),
        "p90": float(np.percentile(latencies, 90)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
    }


def compute_qps(
    latencies_ms: List[float],
    num_threads: int = 1,
) -> float:
    """
    Compute queries per second from latencies.
    """
    if not latencies_ms:
        return 0.0

    mean_latency_ms = np.mean(latencies_ms)
    return 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

def compute_throughput(
    num_items: int,
    total_time_sec: float,
) -> float:
    """
    Compute throughput (items per second).
    """
    return num_items / total_time_sec if total_time_sec > 0 else 0.0


def measure_coldstart_latency(
    search_fn: Callable[[NDArray[np.float32], int], Any],
    query: NDArray[np.float32],
    k: int,
) -> float:
    """
    Measure cold start latency - the time for the very first query.
    Assumes the index is already loaded but no queries have been run.
    """
    start = time.perf_counter()
    search_fn(query, k)
    return (time.perf_counter() - start) * 1000


def measure_warmup_time(
    search_fn: Callable[[NDArray[np.float32], int], Any],
    queries: NDArray[np.float32],
    k: int,
    stabilization_threshold: float = 0.05,
    window_size: int = 10,
    max_warmup_queries: int = 1000,
) -> Tuple[float, int]:
    """
    Measure the time and number of queries to reach a stable latency.
    """
    latencies = []
    start_time = time.perf_counter()

    for i in range(max_warmup_queries):
        query = queries[i % len(queries)]
        q_start = time.perf_counter()
        search_fn(query, k)
        latency = (time.perf_counter() - q_start) * 1000
        latencies.append(latency)

        # Check for stabilization
        if len(latencies) > window_size * 2:
            recent_window = latencies[-window_size:]
            previous_window = latencies[-window_size*2:-window_size]
            
            recent_avg = np.mean(recent_window)
            previous_avg = np.mean(previous_window)

            if abs(recent_avg - previous_avg) / previous_avg < stabilization_threshold:
                break
    
    warmup_time_ms = (time.perf_counter() - start_time) * 1000
    return warmup_time_ms, len(latencies)


def compute_all_performance_metrics(
    latencies_ms: List[float],
    coldstart_latency_ms: Optional[float] = None,
    warmup_time_ms: Optional[float] = None,
) -> PerformanceMetrics:
    """
    Compute all performance metrics.
    """
    percentiles = compute_latency_percentiles(latencies_ms)

    return PerformanceMetrics(
        latency_p50=percentiles["p50"],
        latency_p90=percentiles["p90"],
        latency_p95=percentiles["p95"],
        latency_p99=percentiles["p99"],
        latency_mean=percentiles["mean"],
        latency_std=percentiles["std"],
        latency_min=percentiles["min"],
        latency_max=percentiles["max"],
        qps_single_thread=compute_qps(latencies_ms, 1),
        coldstart_latency_ms=coldstart_latency_ms or 0.0,
        warmup_time_ms=warmup_time_ms or 0.0,
        latencies_ms=latencies_ms,
    )
