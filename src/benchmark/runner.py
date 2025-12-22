"""
Main benchmark runner orchestrating all benchmark operations.
"""

import json
import logging
import time
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
            dataset = get_dataset(dataset_name, data_dir=self.config.dataset.data_dir)
            dataset.ensure_downloaded()

            for db_name in databases:
                console.print(f"\n[bold cyan]Benchmarking: {db_name} on {dataset_name}[/bold cyan]")

                try:
                    result = self._run_single_benchmark(db_name, dataset, index_configs)
                    results.append(result)
                    self._print_summary(result)
                except Exception as e:
                    logger.error(f"Benchmark failed for {db_name}: {e}")
                    console.print(f"[red]Error: {e}[/red]")

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
            database_info=temp_db.info,  # FIXED: Populate database info
            dataset_info=dataset.info,
            hardware_info=self.hardware_info,
        )

        # Load data
        vectors = dataset.vectors
        queries = dataset.queries
        ground_truth = dataset.ground_truth

        # FIXED: Get the metric from the dataset info instead of hardcoding
        metric = dataset.info.distance_metric

        console.print(f"  Vectors: {vectors.shape}, Queries: {queries.shape}")
        console.print(f"  Metric: {metric.value}")

        # Test each index configuration
        for idx_config in configs_to_test:
            console.print(f"\n  [yellow]Index: {idx_config.name}[/yellow]")

            runs = []
            for run_id in range(self.config.experiment.runs):
                run_result = self._execute_run(
                    db_name, db_config, idx_config, vectors, queries, ground_truth, run_id, metric
                )
                runs.append(run_result)
                console.print(f"    Run {run_id + 1}: Recall@10={run_result.metrics.quality.recall_at_10:.4f}, "
                            f"Latency_p50={run_result.metrics.performance.latency_p50:.2f}ms")

            result.runs.extend(runs)

        result.num_runs = len(result.runs)

        # Compute aggregated metrics
        if result.runs:
            result.mean_metrics = self._aggregate_metrics([r.metrics for r in result.runs])

        return result

    def _execute_run(
        self,
        db_name: str,
        db_config: Dict,
        index_config: IndexConfig,
        vectors: np.ndarray,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        run_id: int,
        metric: DistanceMetric,  # FIXED: Added metric parameter
    ) -> BenchmarkRun:
        """Execute a single benchmark run."""
        run_config = RunConfig(
            database=db_name,
            dataset="",
            index_config=index_config,
            distance_metric=metric,  # FIXED: Use passed metric
            k=100,
            num_queries=len(queries),
            run_id=run_id,
        )

        start_time = time.perf_counter()
        metrics = MetricsResult()

        try:
            db = get_database(db_name, db_config)

            with db:
                # Build index
                with ResourceMonitor() as build_monitor:
                    build_time = db.create_index(
                        vectors,
                        index_config,
                        metric,  # FIXED: Use passed metric instead of hardcoded L2
                    )

                metrics.resource.index_build_time_sec = build_time
                metrics.resource.ram_bytes_peak = build_monitor.peak_memory_bytes

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

                # Get index stats
                stats = db.get_index_stats()
                metrics.resource.index_size_bytes = stats.get("index_size_bytes", 0)

                # Cleanup
                db.delete_index()

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

        return configs[:3]  # Limit to 3 configs for initial testing

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

        table = Table(title=f"Results: {result.experiment_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        m = result.mean_metrics
        table.add_row("Recall@10", f"{m.quality.recall_at_10:.4f}")
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