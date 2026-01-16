"""
Export benchmark results to various formats.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List
import pandas as pd

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
    """Compiles multiple JSON results into a single CSV file."""

    def compile_and_export(
        self,
        json_dir: str,
        output_path: str,
    ) -> str:
        """
        Finds all JSON results, compiles them, and exports to a single CSV.
        """
        results_dir = Path(json_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_results_data = []

        print("🔍 Finding all benchmark JSON files to compile...")
        json_files = list(results_dir.glob("**/*_results.json"))

        if not json_files:
            print("❌ No JSON result files found to compile into CSV.")
            return ""

        print(f"Found {len(json_files)} JSON files to compile.")

        for file_path in json_files:
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    for result in data.get("results", []):
                        all_results_data.append(result)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"⚠️ Could not process {file_path.name}: {e}")

        if not all_results_data:
            print("❌ No valid result data found to compile.")
            return ""

        # Use pandas to handle the flattening and CSV writing
        df = pd.json_normalize(all_results_data, sep='_')

        # Clean up column names
        df.columns = [c.replace('mean_metrics_', '') for c in df.columns]
        if 'hardware_info' in df.columns:
            df = df.drop(columns=['hardware_info'])
        if 'scalability' in df.columns:
            df = df.drop(columns=['scalability'])

        # Save to a single, fresh CSV file
        df.to_csv(output_path, mode='w', header=True, index=False)

        return str(output_path)


class LaTeXExporter:
    """Export results to LaTeX tables for research papers."""
    # ... (rest of the class remains the same)
    def __init__(self, float_precision: int = 3, highlight_best: bool = True):
        self.precision = float_precision
        self.highlight_best = highlight_best

    def export_quality_table(
        self,
        results: List[BenchmarkResult],
        output_path: str,
        metrics: List[str] = ["recall@10", "recall@100", "mrr", "ndcg@10"],
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            r"\begin{table}[htbp]", r"\centering", r"\caption{Quality Metrics Comparison}",
            r"\label{tab:quality}", r"\begin{tabular}{l" + "c" * len(metrics) + "}",
            r"\toprule", "Database & " + " & ".join(self._format_header(m) for m in metrics) + r" \\",
            r"\midrule",
        ]
        best_values = {m: 0 for m in metrics}
        for result in results:
            if result.mean_metrics:
                q = result.mean_metrics.quality.to_dict()
                for m in metrics:
                    if m in q and q[m] > best_values[m]:
                        best_values[m] = q[m]
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
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        return str(output_path)

    def export_performance_table(self, results: List[BenchmarkResult], output_path: str) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = ["latency_p50_ms", "latency_p99_ms", "qps_single_thread"]
        lines = [
            r"\begin{table}[htbp]", r"\centering", r"\caption{Performance Metrics Comparison}",
            r"\label{tab:performance}", r"\begin{tabular}{lccc}", r"\toprule",
            r"Database & Latency p50 (ms) & Latency p99 (ms) & QPS \\", r"\midrule",
        ]
        for result in results:
            if result.mean_metrics:
                name = result.experiment_name.split("_")[0]
                p = result.mean_metrics.performance.to_dict()
                lines.append(
                    f"{name} & {p['latency_p50_ms']:.2f} & {p['latency_p99_ms']:.2f} & "
                    f"{p['qps_single_thread']:.0f} \\\\"
                )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        return str(output_path)

    def _format_header(self, metric: str) -> str:
        replacements = {
            "recall@": "Recall@", "precision@": "Prec@", "ndcg@": "NDCG@", "map@": "MAP@",
            "mrr": "MRR", "hit_rate@": "HR@", "f1@": "F1@",
        }
        result = metric
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result
