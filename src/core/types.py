"""
Type definitions for the Vector Database Benchmarking Framework.

This module defines all data structures used throughout the benchmarking framework,
following best practices from ANN-Benchmarks and VectorDBBench.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Enums
# =============================================================================


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity search."""

    L2 = "l2"  # Euclidean distance
    COSINE = "cosine"  # Cosine similarity
    IP = "ip"  # Inner product (dot product)

    @property
    def is_similarity(self) -> bool:
        """Return True if higher values indicate more similarity."""
        return self in (DistanceMetric.COSINE, DistanceMetric.IP)


class IndexType(str, Enum):
    """Common index types across vector databases."""

    FLAT = "flat"  # Exact brute-force search
    IVF = "ivf"  # Inverted file index
    IVFPQ = "ivfpq"  # IVF with product quantization
    IVFSQ = "ivfsq"  # IVF with scalar quantization
    HNSW = "hnsw"  # Hierarchical Navigable Small World
    DISKANN = "diskann"  # Disk-based ANN
    ANNOY = "annoy"  # Approximate Nearest Neighbors Oh Yeah
    SCANN = "scann"  # Scalable Nearest Neighbors


class WorkloadType(str, Enum):
    """Types of benchmark workloads."""

    STATIC = "static"  # Build index, then query
    STREAMING = "streaming"  # Continuous inserts during queries
    MIXED = "mixed"  # Concurrent reads and writes
    BATCH = "batch"  # Batch operations


class DatabaseType(str, Enum):
    """Types of vector database implementations."""

    LIBRARY = "library"  # In-process library (e.g., FAISS)
    DATABASE = "database"  # Standalone database server
    EXTENSION = "extension"  # Database extension (e.g., pgvector)
    CLOUD = "cloud"  # Cloud-managed service


# =============================================================================
# Configuration Types
# =============================================================================


@dataclass
class IndexConfig:
    """Configuration for a specific index type."""

    name: str
    type: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    search_params: Dict[str, Any] = field(default_factory=dict)
    quantization: Optional[Dict[str, Any]] = None
    requires_gpu: bool = False
    requires_training: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Index configuration must have a name")


@dataclass
class DatabaseInfo:
    """Information about a vector database."""

    name: str
    display_name: str
    version: str
    type: DatabaseType
    language: str
    license: str
    supported_metrics: List[DistanceMetric] = field(default_factory=list)
    supported_index_types: List[IndexType] = field(default_factory=list)
    supports_filtering: bool = False
    supports_hybrid_search: bool = False
    supports_gpu: bool = False
    is_distributed: bool = False


@dataclass
class DatasetInfo:
    """Information about a benchmark dataset."""

    name: str
    display_name: str
    description: str
    num_vectors: int
    num_queries: int
    dimensions: int
    data_type: str = "float32"
    distance_metric: DistanceMetric = DistanceMetric.L2
    ground_truth_k: int = 100
    source_url: Optional[str] = None
    size_mb: Optional[float] = None


# =============================================================================
# Search Result Types
# =============================================================================


@dataclass
class SearchResult:
    """Result of a single vector search query."""

    query_id: int
    indices: NDArray[np.int64]  # Indices of retrieved vectors
    distances: NDArray[np.float32]  # Distances to retrieved vectors
    latency_ms: float  # Query latency in milliseconds

    @property
    def k(self) -> int:
        """Number of results returned."""
        return len(self.indices)


@dataclass
class BatchSearchResult:
    """Result of a batch of vector search queries."""

    results: List[SearchResult]
    total_latency_ms: float
    batch_size: int

    @property
    def mean_latency_ms(self) -> float:
        """Mean latency per query."""
        return self.total_latency_ms / self.batch_size if self.batch_size > 0 else 0.0


# =============================================================================
# Metrics Types
# =============================================================================


@dataclass
class QualityMetrics:
    """Quality metrics for retrieval accuracy."""

    # Recall metrics
    recall_at_1: float = 0.0
    recall_at_10: float = 0.0
    recall_at_50: float = 0.0
    recall_at_100: float = 0.0

    # Precision metrics
    precision_at_1: float = 0.0
    precision_at_10: float = 0.0
    precision_at_50: float = 0.0
    precision_at_100: float = 0.0

    # Ranking metrics
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_10: float = 0.0  # Normalized DCG @10
    ndcg_at_100: float = 0.0  # Normalized DCG @100
    map_at_10: float = 0.0  # Mean Average Precision @10
    map_at_100: float = 0.0  # Mean Average Precision @100

    # Success metrics
    hit_rate_at_1: float = 0.0
    hit_rate_at_10: float = 0.0

    # F1 score
    f1_at_10: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "recall@1": self.recall_at_1,
            "recall@10": self.recall_at_10,
            "recall@50": self.recall_at_50,
            "recall@100": self.recall_at_100,
            "precision@1": self.precision_at_1,
            "precision@10": self.precision_at_10,
            "precision@50": self.precision_at_50,
            "precision@100": self.precision_at_100,
            "mrr": self.mrr,
            "ndcg@10": self.ndcg_at_10,
            "ndcg@100": self.ndcg_at_100,
            "map@10": self.map_at_10,
            "map@100": self.map_at_100,
            "hit_rate@1": self.hit_rate_at_1,
            "hit_rate@10": self.hit_rate_at_10,
            "f1@10": self.f1_at_10,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for query speed."""

    # Latency percentiles (milliseconds)
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_mean: float = 0.0
    latency_std: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0

    # Throughput
    qps_single_thread: float = 0.0  # Queries per second (single thread)
    qps_max: float = 0.0  # Maximum QPS (multi-threaded)
    qps_at_recall_90: float = 0.0  # QPS at 90% recall
    qps_at_recall_95: float = 0.0  # QPS at 95% recall
    qps_at_recall_99: float = 0.0  # QPS at 99% recall

    # Cold start
    coldstart_latency_ms: float = 0.0
    warmup_time_ms: float = 0.0

    # Raw latencies for distribution analysis
    latencies_ms: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (excluding raw latencies)."""
        return {
            "latency_p50_ms": self.latency_p50,
            "latency_p90_ms": self.latency_p90,
            "latency_p95_ms": self.latency_p95,
            "latency_p99_ms": self.latency_p99,
            "latency_mean_ms": self.latency_mean,
            "latency_std_ms": self.latency_std,
            "latency_min_ms": self.latency_min,
            "latency_max_ms": self.latency_max,
            "qps_single_thread": self.qps_single_thread,
            "qps_max": self.qps_max,
            "qps_at_recall_90": self.qps_at_recall_90,
            "qps_at_recall_95": self.qps_at_recall_95,
            "qps_at_recall_99": self.qps_at_recall_99,
            "coldstart_latency_ms": self.coldstart_latency_ms,
            "warmup_time_ms": self.warmup_time_ms,
        }


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    # Build metrics
    index_build_time_sec: float = 0.0
    index_size_bytes: int = 0
    disk_bytes: int = 0

    # Memory metrics
    ram_bytes_peak: int = 0
    ram_bytes_steady: int = 0
    ram_bytes_index: int = 0

    # Efficiency metrics
    bytes_per_vector: float = 0.0

    # CPU metrics
    cpu_utilization_percent: float = 0.0
    cpu_time_user_sec: float = 0.0
    cpu_time_system_sec: float = 0.0

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert to dictionary."""
        return {
            "index_build_time_sec": self.index_build_time_sec,
            "index_size_bytes": self.index_size_bytes,
            "index_size_mb": self.index_size_bytes / (1024 * 1024) if self.index_size_bytes else 0,
            "disk_bytes": self.disk_bytes,
            "disk_mb": self.disk_bytes / (1024 * 1024) if self.disk_bytes else 0,
            "ram_bytes_peak": self.ram_bytes_peak,
            "ram_mb_peak": self.ram_bytes_peak / (1024 * 1024) if self.ram_bytes_peak else 0,
            "ram_bytes_steady": self.ram_bytes_steady,
            "ram_mb_steady": self.ram_bytes_steady / (1024 * 1024) if self.ram_bytes_steady else 0,
            "bytes_per_vector": self.bytes_per_vector,
            "cpu_utilization_percent": self.cpu_utilization_percent,
        }


@dataclass
class OperationalMetrics:
    """Operational metrics for production readiness."""

    # Insert operations
    insert_latency_single_ms: float = 0.0
    insert_throughput_batch: float = 0.0  # vectors per second

    # Update operations
    update_latency_ms: float = 0.0
    update_throughput: float = 0.0

    # Delete operations
    delete_latency_ms: float = 0.0
    delete_throughput: float = 0.0

    # Maintenance
    compaction_time_sec: float = 0.0
    recovery_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "insert_latency_single_ms": self.insert_latency_single_ms,
            "insert_throughput_batch": self.insert_throughput_batch,
            "update_latency_ms": self.update_latency_ms,
            "update_throughput": self.update_throughput,
            "delete_latency_ms": self.delete_latency_ms,
            "delete_throughput": self.delete_throughput,
            "compaction_time_sec": self.compaction_time_sec,
            "recovery_time_sec": self.recovery_time_sec,
        }


@dataclass
class ScalabilityMetrics:
    """Scalability metrics across different dimensions."""

    # QPS vs threads
    qps_by_threads: Dict[int, float] = field(default_factory=dict)

    # Metrics vs dataset size
    latency_by_size: Dict[int, float] = field(default_factory=dict)
    memory_by_size: Dict[int, int] = field(default_factory=dict)
    build_time_by_size: Dict[int, float] = field(default_factory=dict)

    # Metrics vs dimensions
    latency_by_dims: Dict[int, float] = field(default_factory=dict)
    memory_by_dims: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "qps_by_threads": self.qps_by_threads,
            "latency_by_size": self.latency_by_size,
            "memory_by_size": self.memory_by_size,
            "build_time_by_size": self.build_time_by_size,
            "latency_by_dims": self.latency_by_dims,
            "memory_by_dims": self.memory_by_dims,
        }


@dataclass
class MetricsResult:
    """Aggregated metrics from a benchmark run."""

    quality: QualityMetrics = field(default_factory=QualityMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    resource: ResourceMetrics = field(default_factory=ResourceMetrics)
    operational: OperationalMetrics = field(default_factory=OperationalMetrics)
    scalability: ScalabilityMetrics = field(default_factory=ScalabilityMetrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to a flat dictionary."""
        result = {}
        result.update(self.quality.to_dict())
        result.update(self.performance.to_dict())
        result.update(self.resource.to_dict())
        result.update(self.operational.to_dict())
        result["scalability"] = self.scalability.to_dict()
        return result


# =============================================================================
# Benchmark Result Types
# =============================================================================


@dataclass
class RunConfig:
    """Configuration for a single benchmark run."""

    database: str
    dataset: str
    index_config: IndexConfig
    distance_metric: DistanceMetric
    k: int
    num_queries: int
    num_threads: int = 1
    search_params: Dict[str, Any] = field(default_factory=dict)
    run_id: int = 0


@dataclass
class BenchmarkRun:
    """Result of a single benchmark run."""

    config: RunConfig
    metrics: MetricsResult
    run_id: int
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None
    duration_sec: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result including all runs."""

    # Identification
    experiment_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    retrieval_method: str = "vector_search"

    # Configuration
    database_info: Optional[DatabaseInfo] = None
    dataset_info: Optional[DatasetInfo] = None
    index_config: Optional[IndexConfig] = None

    # Runs
    runs: List[BenchmarkRun] = field(default_factory=list)
    num_runs: int = 0

    # Aggregated metrics (mean across runs)
    mean_metrics: Optional[MetricsResult] = None
    std_metrics: Optional[MetricsResult] = None

    # Hardware info
    hardware_info: Dict[str, Any] = field(default_factory=dict)

    # Status
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        flat_metrics = self.mean_metrics.to_dict() if self.mean_metrics else {}
        
        # Base dictionary
        data = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp.isoformat(),
            "retrieval_method": self.retrieval_method,
            "database": self.database_info.name if self.database_info else None,
            "dataset": self.dataset_info.name if self.dataset_info else None,
            "index_config": self.index_config.name if self.index_config else None,
            "num_runs": self.num_runs,
            "hardware_info": self.hardware_info,
            "success": self.success,
            "error_message": self.error_message,
        }
        
        # Merge metrics
        data.update(flat_metrics)
        
        return data


# =============================================================================
# Filtering Types
# =============================================================================


@dataclass
class FilterCondition:
    """A single filter condition for metadata filtering."""

    field: str
    operator: str  # eq, ne, gt, gte, lt, lte, in, contains
    value: Any


@dataclass
class FilterConfig:
    """Configuration for a filtering benchmark."""

    name: str
    selectivity: float  # Fraction of data that matches (0.0-1.0)
    conditions: List[FilterCondition] = field(default_factory=list)
    description: str = ""


# =============================================================================
# Type Aliases
# =============================================================================

# Vector types
Vector = NDArray[np.float32]
VectorBatch = NDArray[np.float32]  # Shape: (n, d)

# ID types
VectorID = Union[int, str]
VectorIDs = List[VectorID]

# Ground truth type: list of (query_id, list of true neighbor ids)
GroundTruth = List[Tuple[int, List[int]]]

# Parameter grid for hyperparameter search
ParamGrid = Dict[str, List[Any]]
