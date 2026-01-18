"""
Metrics computation module for vector database benchmarking.

Implements all evaluation metrics following SOTA methodologies:
    - Quality metrics (Recall, Precision, MRR, NDCG, MAP, F1)
    - Performance metrics (Latency percentiles, QPS, throughput)
    - Resource metrics (Memory, disk, CPU, build time)
    - Operational metrics (Insert, update, delete latencies)
"""

from src.metrics.quality import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_map_at_k,
    compute_hit_rate_at_k,
    compute_f1_at_k,
    compute_all_quality_metrics,
)

from src.metrics.performance import (
    compute_latency_percentiles,
    compute_qps,
    compute_throughput,
    measure_coldstart_latency,
    measure_warmup_time,
    compute_all_performance_metrics,
)

from src.metrics.resource import (
    measure_memory_usage,
    measure_disk_usage,
    compute_all_resource_metrics,
)

__all__ = [
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_map_at_k",
    "compute_hit_rate_at_k",
    "compute_f1_at_k",
    "compute_all_quality_metrics",
    "compute_latency_percentiles",
    "compute_qps",
    "compute_throughput",
    "measure_coldstart_latency",
    "measure_warmup_time",
    "compute_all_performance_metrics",
    "measure_memory_usage",
    "measure_disk_usage",
    "compute_all_resource_metrics",
]
