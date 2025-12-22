#!/usr/bin/env python3
"""
Main entry point for running benchmarks.

Usage:
    python scripts/run_benchmark.py --database faiss --dataset sift1m
    python scripts/run_benchmark.py --all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.runner import BenchmarkRunner
from src.core.config import load_config
from src.reporting import JSONExporter, CSVExporter, LaTeXExporter, BenchmarkVisualizer


def main():
    parser = argparse.ArgumentParser(description="Run VectorDB Benchmarks")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--database", "-d", nargs="+", help="Database(s) to benchmark")
    parser.add_argument("--dataset", "-s", nargs="+", help="Dataset(s) to use")
    parser.add_argument("--output", "-o", default="./results", help="Output directory")
    parser.add_argument("--runs", "-r", type=int, default=5, help="Number of runs")
    parser.add_argument("--all", action="store_true", help="Run all enabled databases/datasets")
    parser.add_argument("--export", nargs="+", choices=["json", "csv", "latex", "plots"],
                        default=["json"], help="Export formats")

    args = parser.parse_args()

    print("=" * 60)
    print("VectorDB Benchmark Framework")
    print("=" * 60)

    # Load configuration
    config = load_config(args.config) if args.config else load_config()

    if args.runs:
        config.experiment.runs = args.runs

    if args.all:
        config.database.compare_all = True
        config.dataset.compare_all = True

    # Run benchmarks
    runner = BenchmarkRunner(config)
    results = runner.run(databases=args.database, datasets=args.dataset)

    # Export results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "json" in args.export:
        # User requested specific format: results/<db>/<dataset>_<db>_results.json
        for result in results:
            # Determine database and dataset names safely
            db_name = result.database_info.name if result.database_info else "unknown"
            dataset_name = result.dataset_info.name

            # Create specific directory: results/<database_name>
            target_dir = output_dir / db_name
            target_dir.mkdir(parents=True, exist_ok=True)

            # Construct filename: <dataset>_<database>_results.json
            filename = f"{dataset_name}_{db_name}_results.json"
            target_path = target_dir / filename

            # Export individually
            JSONExporter().export([result], str(target_path))
            print(f"Exported JSON to {target_path}")

    if "csv" in args.export:
        CSVExporter().export(results, str(output_dir / "results.csv"))
        print(f"Exported CSV to {output_dir / 'results.csv'}")

    if "latex" in args.export:
        LaTeXExporter().export_quality_table(results, str(output_dir / "tables" / "quality.tex"))
        LaTeXExporter().export_performance_table(results, str(output_dir / "tables" / "performance.tex"))
        print(f"Exported LaTeX tables to {output_dir / 'tables'}")

    if "plots" in args.export:
        visualizer = BenchmarkVisualizer()
        visualizer.generate_all_plots(results, str(output_dir / "plots"))
        print(f"Generated plots in {output_dir / 'plots'}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()