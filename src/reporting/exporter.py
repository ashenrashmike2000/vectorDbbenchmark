"""
Export benchmark results to various formats.
"""

import csv
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.types import BenchmarkResult


class JSONExporter:
    """Export results to JSON format."""

    def export(
        self,
        results: List[BenchmarkResult],
        output_path: str,
        pretty: bool = True,
    ) -> str:
        """
        Export results to JSON file.

        Args:
            results: List of benchmark results
            output_path: Output file path
            pretty: Use pretty printing

        Returns:
            Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_results": len(results),
            },
            "results": [r.to_dict() for r in results],
        }

        with open(output_path, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)

        return str(output_path)


class CSVExporter:
    """Export results to CSV format."""

    def export(
        self,
        results: List[BenchmarkResult],
        output_path: str,
    ) -> str:
        """
        Export results to CSV file.

        Args:
            results: List of benchmark results
            output_path: Output file path

        Returns:
            Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten results to rows
        rows = []
        for result in results:
            if result.mean_metrics:
                row = {
                    "experiment": result.experiment_name,
                    "database": result.database_info.name if result.database_info else "",
                    "dataset": result.dataset_info.name if result.dataset_info else "",
                    "num_runs": result.num_runs,
                    **result.mean_metrics.quality.to_dict(),
                    **result.mean_metrics.performance.to_dict(),
                    **result.mean_metrics.resource.to_dict(),
                }
                rows.append(row)

        if rows:
            fieldnames = list(rows[0].keys())

            # Check if file exists to decide whether to write the header
            file_exists = os.path.isfile(output_path)

            # Open in 'a' (append) mode instead of 'w' (write)
            with open(output_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                # Only write header if the file is new
                if not file_exists:
                    writer.writeheader()

                writer.writerows(rows)

        return str(output_path)


class LaTeXExporter:
    """Export results to LaTeX tables for research papers."""

    def __init__(self, float_precision: int = 3, highlight_best: bool = True):
        """
        Initialize LaTeX exporter.

        Args:
            float_precision: Number of decimal places
            highlight_best: Bold the best values
        """
        self.precision = float_precision
        self.highlight_best = highlight_best

    def export_quality_table(
        self,
        results: List[BenchmarkResult],
        output_path: str,
        metrics: List[str] = ["recall@10", "recall@100", "mrr", "ndcg@10"],
    ) -> str:
        """
        Export quality metrics as LaTeX table.

        Args:
            results: List of benchmark results
            output_path: Output file path
            metrics: Metrics to include

        Returns:
            Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build table
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Quality Metrics Comparison}",
            r"\label{tab:quality}",
            r"\begin{tabular}{l" + "c" * len(metrics) + "}",
            r"\toprule",
            "Database & " + " & ".join(self._format_header(m) for m in metrics) + r" \\",
            r"\midrule",
        ]

        # Find best values for highlighting
        best_values = {m: 0 for m in metrics}
        for result in results:
            if result.mean_metrics:
                q = result.mean_metrics.quality.to_dict()
                for m in metrics:
                    if m in q and q[m] > best_values[m]:
                        best_values[m] = q[m]

        # Add data rows
        for result in results:
            if result.mean_metrics:
                name = result.experiment_name.split("_")[0]
                q = result.mean_metrics.quality.to_dict()
                values = []
                for m in metrics:
                    val = q.get(m, 0)
                    formatted = f"{val:.{self.precision}f}"
                    if self.highlight_best and abs(val - best_values[m]) < 0.001:
                        formatted = r"\textbf{" + formatted + "}"
                    values.append(formatted)
                lines.append(f"{name} & " + " & ".join(values) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

        return str(output_path)

    def export_performance_table(
        self,
        results: List[BenchmarkResult],
        output_path: str,
    ) -> str:
        """Export performance metrics as LaTeX table."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = ["latency_p50_ms", "latency_p99_ms", "qps_single_thread"]

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Performance Metrics Comparison}",
            r"\label{tab:performance}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Database & Latency p50 (ms) & Latency p99 (ms) & QPS \\",
            r"\midrule",
        ]

        for result in results:
            if result.mean_metrics:
                name = result.experiment_name.split("_")[0]
                p = result.mean_metrics.performance.to_dict()
                lines.append(
                    f"{name} & {p['latency_p50_ms']:.2f} & {p['latency_p99_ms']:.2f} & "
                    f"{p['qps_single_thread']:.0f} \\\\"
                )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

        return str(output_path)

    def _format_header(self, metric: str) -> str:
        """Format metric name for table header."""
        replacements = {
            "recall@": "Recall@",
            "precision@": "Prec@",
            "ndcg@": "NDCG@",
            "map@": "MAP@",
            "mrr": "MRR",
            "hit_rate@": "HR@",
            "f1@": "F1@",
        }
        result = metric
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result
