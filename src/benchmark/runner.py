"""
Main benchmark runner orchestrating all benchmark operations.
"""

import uuid
import json
import logging
import time
import random
import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from src.core.base import VectorDBInterface
from src.core.config import Config, detect_hardware, load_config
from src.core.types import (
    BenchmarkResult,
    BenchmarkRun,
    IndexConfig,
    MetricsResult,
    RunConfig,
    DistanceMetric,
    ResourceMetrics,
)
from src.databases import get_database
from src.datasets import get_dataset
from src.datasets.base import DatasetLoader
from src.metrics import compute_all_quality_metrics, compute_all_performance_metrics
from src.metrics.resource import ResourceMonitor, compute_all_resource_metrics

logger = logging.getLogger(__name__)
console = Console()


class BenchmarkRunner:
    """
    Main benchmark orchestrator.

    Coordinates database initialization, dataset loading, benchmark execution,
    and results collection following SOTA methodologies.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            config: Configuration object
            config_path: Path to configuration file
        """
        self.config = config or load_config(config_path)
        self.results: List[BenchmarkResult] = []
        self.hardware_info = detect_hardware()

    def run(
        self,
        databases: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        index_configs: Optional[List[str]] = None,
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks for specified databases and datasets.

        Args:
            databases: List of database names (default: from config)
            datasets: List of dataset names (default: from config)
            index_configs: Specific index configs to test

        Returns:
            List of benchmark results
        """
        # Determine what to benchmark
        if databases is None:
            if self.config.database.compare_all:
                databases = self.config.get_enabled_databases()
            else:
                databases = [self.config.database.active]

        if datasets is None:
            if self.config.dataset.compare_all:
                datasets = self.config.get_enabled_datasets()
            else:
                datasets = [self.config.dataset.active]

        console.print(f"\n[bold blue]VectorDB Benchmark[/bold blue]")
        console.print(f"Databases: {', '.join(databases)}")
        console.print(f"Datasets: {', '.join(datasets)}")
        console.print(f"Runs per config: {self.config.experiment.runs}")
        console.print()

        results = []

        for dataset_name in datasets:
            console.print(f"\n[bold]Loading dataset: {dataset_name}[/bold]")
            try:
                dataset = get_dataset(dataset_name, data_dir=self.config.dataset.data_dir)
                dataset.ensure_downloaded()

                for db_name in databases:
                    console.print(f"\n[bold cyan]Benchmarking: {db_name} on {dataset_name}[/bold cyan]")
                    # Flush stdout to ensure logs are captured before potential crash
                    sys.stdout.flush()

                    try:
                        result = self._run_single_benchmark(db_name, dataset, index_configs)
                        results.append(result)
                        self._print_summary(result)
                        
                        # Save intermediate results
                        self.save_results()
                        
                    except Exception as e:
                        logger.error(f"Benchmark failed for {db_name}: {e}")
                        console.print(f"[red]Error: {e}[/red]")
                    
                    # Force garbage collection after each DB run
                    gc.collect()

                # Cleanup dataset memory
                del dataset
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to load or process dataset {dataset_name}: {e}")
                console.print(f"[red]Dataset Error: {e}[/red]")

        self.results = results
        return results

    def _run_single_benchmark(
        self,
        db_name: str,
        dataset: DatasetLoader,
        index_configs: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """Run benchmark for a single database-dataset pair."""
        db_config = self.config.get_database_config(db_name)
        configs_to_test = self._get_index_configs(db_config, index_configs)

        # FIXED: Instantiate temporary DB to get correct info for metadata
        temp_db = get_database(db_name, db_config)

        result = BenchmarkResult(
            experiment_name=f"{db_name}_{dataset.name}",
            retrieval_method="vector_search",
            database_info=temp_db.info,  # FIXED: Populate database info
            dataset_info=dataset.info,
            hardware_info=self.hardware_info,
        )

        # Load data
        vectors = dataset.vectors
        queries = dataset.queries

        # === ADD THIS SAFETY SLICE ===
        if dataset.name == "msmarco" and len(vectors) > 1000000:
            print("✂️  Slicing MSMARCO to 1M vectors to save RAM...")
            vectors = vectors[:1000000]
            # =============================

        ground_truth = dataset.ground_truth

        # FIXED: Get the metric from the dataset info instead of hardcoding
        metric = dataset.info.distance_metric

        console.print(f"  Vectors: {vectors.shape}, Queries: {queries.shape}")
        console.print(f"  Metric: {metric.value}")

        # Test each index configuration
        for idx_config in configs_to_test:

            # =========================================================
            # SMART FILTER: Automatically skip mismatched configurations
            # =========================================================

            # 1. Determine Dataset's Required Metric
            # Convert dataset metric to lowercase string (e.g. "cosine", "l2")
            req_metric = metric.value.lower()
            if req_metric == 'angular': req_metric = 'cosine'
            if req_metric == 'euclidean': req_metric = 'l2'

            # 2. Determine Index Config's Metric
            # Check for keys used by different DBs (Chroma='space', Milvus='metric_type', Weaviate='distance')
            params = idx_config.params
            cfg_metric = params.get('space') or params.get('metric_type') or params.get('distance')

            if cfg_metric:
                cfg_metric = cfg_metric.lower()
                # Normalize common synonyms
                if cfg_metric == 'ip': cfg_metric = 'cosine'
                if cfg_metric == 'l2-squared': cfg_metric = 'l2'

                # 3. Compare and Skip
                # If both metrics are known but different, skip this config
                if req_metric and cfg_metric and req_metric != cfg_metric:
                    console.print(f"  [dim]Skipping {idx_config.name}: Dataset requires '{req_metric}', Config is '{cfg_metric}'[/dim]")
                    continue

            # =========================================================
            # END SMART FILTER
            # =========================================================

            console.print(f"\n  [yellow]Index: {idx_config.name}[/yellow]")

            # =========================================================
            # BUILD ONCE, SEARCH MANY
            # =========================================================
            db = None
            try:
                db = get_database(db_name, db_config)
                db.connect()  # <--- FIXED: Explicitly connect
                
                # 1. Build Index (Once)
                console.print("    [dim]Building index...[/dim]")
                with ResourceMonitor() as build_monitor:
                    build_time = db.create_index(
                        vectors,
                        index_config=idx_config,
                        distance_metric=metric,
                    )

                # If the adapter has the smart wait method, use it
                if db_name == "weaviate":
                    # Removed 300s sleep as WeaviateAdapter now handles wait
                    pass

                # Capture Build Metrics
                build_metrics = ResourceMetrics()
                build_metrics.index_build_time_sec = build_time
                build_metrics.ram_bytes_peak = build_monitor.peak_memory_bytes
                
                # Get index stats
                stats = db.get_index_stats()
                build_metrics.index_size_bytes = stats.get("index_size_bytes", 0)

                # 2. Run Searches (Many)
                runs = []
                for run_id in range(self.config.experiment.runs):
                    run_result = self._execute_search_run(
                        db=db,
                        db_name=db_name,
                        index_config=idx_config,
                        queries=queries,
                        ground_truth=ground_truth,
                        run_id=run_id,
                        metric=metric,
                        build_metrics=build_metrics,
                        num_vectors=len(vectors)
                    )
                    runs.append(run_result)
                    console.print(f"    Run {run_id + 1}: Recall@10={run_result.metrics.quality.recall_at_10:.4f}, "
                                f"Latency_p50={run_result.metrics.performance.latency_p50:.2f}ms")
                    
                    # Flush logs
                    sys.stdout.flush()

                result.runs.extend(runs)

            except Exception as e:
                logger.exception(f"Benchmark failed for config {idx_config.name}: {e}")
                console.print(f"[red]Error with config {idx_config.name}: {e}[/red]")
            
            finally:
                # 3. Cleanup (Once)
                if db:
                    try:
                        db.delete_index()
                    except Exception as e:
                        logger.error(f"Failed to delete index: {e}")
                    
                    try:
                        db.disconnect()  # <--- FIXED: Explicitly disconnect
                    except Exception as e:
                        logger.error(f"Failed to disconnect: {e}")
                
                # Force GC after index cleanup
                gc.collect()

        result.num_runs = len(result.runs)

        # Compute aggregated metrics
        if result.runs:
            result.mean_metrics = self._aggregate_metrics([r.metrics for r in result.runs])

        return result

    def _execute_search_run(
        self,
        db: VectorDBInterface,
        db_name: str,
        index_config: IndexConfig,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        run_id: int,
        metric: DistanceMetric,
        build_metrics: ResourceMetrics,
        num_vectors: int,
    ) -> BenchmarkRun:
        """Execute a single search benchmark run on an existing index."""
        run_config = RunConfig(
            database=db_name,
            dataset="",
            index_config=index_config,
            distance_metric=metric,
            k=100,
            num_queries=len(queries),
            run_id=run_id,
        )

        start_time = time.perf_counter()
        metrics = MetricsResult()
        
        # Copy build metrics
        metrics.resource = build_metrics

        # Calculate Batch Throughput (derived from build time)
        if build_metrics.index_build_time_sec > 0:
            metrics.operational.insert_throughput_batch = num_vectors / build_metrics.index_build_time_sec

        try:
            # Warmup
            warmup_queries = queries[:self.config.experiment.warmup_queries]
            for q in warmup_queries:
                db.search_single(q, k=10)

            # Search with timing
            k = 100
            search_params = index_config.search_params

            # Get first search param value if it's a list
            if search_params:
                resolved_params = {}
                for key, value in search_params.items():
                    if isinstance(value, list) and len(value) > 0:
                        resolved_params[key] = value[len(value) // 2]  # Use middle value
                    else:
                        resolved_params[key] = value
                search_params = resolved_params

            indices, distances, latencies = db.search(queries, k, search_params)

            # Compute quality metrics
            metrics.quality = compute_all_quality_metrics(indices, ground_truth)

            # Compute performance metrics
            metrics.performance = compute_all_performance_metrics(latencies)

            # ======================================================
            # CRUD Operations Benchmark (Ops Metrics)
            # ======================================================
            # console.print("    [dim]Measuring CRUD Operations...[/dim]")

            # Generate a random dummy vector for CRUD operations to avoid data leakage
            # Use the same dimension as queries
            dim = queries.shape[1]
            dummy_vec = np.random.rand(dim).astype(np.float32)

            # Warmup for CRUD
            try:
                if hasattr(db, 'insert_one') and hasattr(db, 'delete_one'):
                    warmup_id = "99999999" # Distinct from test ID
                    db.insert_one(warmup_id, dummy_vec)
                    db.delete_one(warmup_id)
            except Exception:
                pass

            # 1. Measure Single Insert Latency
            # Use a large random integer to avoid collisions and work with all DBs
            dummy_id = str(random.randint(10_000_000, 99_999_999))

            t0 = time.perf_counter()
            try:
                if hasattr(db, 'insert_one'):
                    db.insert_one(dummy_id, dummy_vec)
                    metrics.operational.insert_latency_single_ms = (time.perf_counter() - t0) * 1000
                else:
                    metrics.operational.insert_latency_single_ms = 0.0
            except Exception as e:
                logger.warning(f"Insert ops failed: {e}")

            # 2. Measure Update Latency
            t0 = time.perf_counter()
            try:
                if hasattr(db, 'update_one'):
                    updated_vec = dummy_vec + 0.01
                    db.update_one(dummy_id, updated_vec)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    metrics.operational.update_latency_ms = latency_ms
                    if latency_ms > 0:
                        metrics.operational.update_throughput = 1000 / latency_ms
            except Exception as e:
                logger.warning(f"Update ops failed: {e}")

            # 3. Measure Delete Latency
            t0 = time.perf_counter()
            try:
                if hasattr(db, 'delete_one'):
                    db.delete_one(dummy_id)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    metrics.operational.delete_latency_ms = latency_ms
                    if latency_ms > 0:
                        metrics.operational.delete_throughput = 1000 / latency_ms
            except Exception as e:
                logger.warning(f"Delete ops failed: {e}")

            success = True
            error_msg = None

        except Exception as e:
            logger.exception(f"Run failed: {e}")
            success = False
            error_msg = str(e)

        duration = time.perf_counter() - start_time

        return BenchmarkRun(
            config=run_config,
            metrics=metrics,
            run_id=run_id,
            timestamp=datetime.now(),
            success=success,
            error_message=error_msg,
            duration_sec=duration,
        )

    def _get_index_configs(
        self,
        db_config: Dict,
        filter_names: Optional[List[str]] = None,
    ) -> List[IndexConfig]:
        """Get index configurations to test."""
        configs = []
        raw_configs = db_config.get("index_configurations", [])

        for cfg in raw_configs:
            if filter_names and cfg["name"] not in filter_names:
                continue

            configs.append(IndexConfig(
                name=cfg["name"],
                type=cfg["type"],
                description=cfg.get("description", ""),
                params=cfg.get("params", {}),
                search_params=cfg.get("search_params", {}),
            ))

        # NOTE: Removed the limit [:3] here to ensure all configs (like HNSW_L2 and HNSW_Cosine)
        # are available for the Smart Filter to choose from.
        return configs

    def _aggregate_metrics(self, metrics_list: List[MetricsResult]) -> MetricsResult:
        """Aggregate metrics across multiple runs."""
        if not metrics_list:
            return MetricsResult()

        # For simplicity, return the last run's metrics
        # In production, compute mean/std across runs
        return metrics_list[-1]

    def _print_summary(self, result: BenchmarkResult) -> None:
        """Print a summary table of results."""
        if not result.mean_metrics:
            return

        table = Table(
            title=f"Results: {result.experiment_name}",
            caption="Recall is normalized by total ground truth size (typically 100). Precision is # Relevant / K."
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        m = result.mean_metrics
        table.add_row("Recall@10", f"{m.quality.recall_at_10:.4f}")
        table.add_row("Precision@10", f"{m.quality.precision_at_10:.4f}")
        table.add_row("Recall@100", f"{m.quality.recall_at_100:.4f}")
        table.add_row("MRR", f"{m.quality.mrr:.4f}")
        table.add_row("Latency p50 (ms)", f"{m.performance.latency_p50:.2f}")
        table.add_row("Latency p99 (ms)", f"{m.performance.latency_p99:.2f}")
        table.add_row("QPS", f"{m.performance.qps_single_thread:.1f}")
        table.add_row("Build Time (s)", f"{m.resource.index_build_time_sec:.2f}")

        console.print(table)

    def save_results(self, output_dir: str = "./results") -> str:
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"benchmark_results_{timestamp}.json"

        results_data = [r.to_dict() for r in self.results]

        with open(filename, 'w') as f:
            json.dump({
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "num_results": len(results_data)
                },
                "results": results_data
            }, f, indent=2, default=str)

        return str(filename)