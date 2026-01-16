"""
Generates all plots from the final benchmark results CSV.
"""
import os
import pandas as pd
import sys
import argparse

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.reporting.visualizer import BenchmarkVisualizer

def main():
    parser = argparse.ArgumentParser(description="Generate plots from benchmark results.")
    parser.add_argument("--database", nargs='+', help="List of databases to include in plots.")
    parser.add_argument("--dataset", nargs='+', help="List of datasets to include in plots.")
    args = parser.parse_args()

    results_csv_path = os.path.join(PROJECT_ROOT, "results", "results.csv")
    plots_dir = os.path.join(PROJECT_ROOT, "results", "plots")

    if not os.path.exists(results_csv_path):
        print(f"❌ Error: Results file not found at {results_csv_path}")
        sys.exit(1)

    print(f"📊 Generating plots from {results_csv_path}...")

    df = pd.read_csv(results_csv_path)

    # Filter DataFrame based on command-line arguments
    if args.database:
        # Handle 'all' keyword for databases
        if 'all' not in args.database:
            df = df[df['database'].isin(args.database)]
    if args.dataset:
        df = df[df['dataset'].isin(args.dataset)]

    if df.empty:
        print("⚠️ No matching results found to plot for the given filter.")
        return

    visualizer = BenchmarkVisualizer()
    generated_plots = visualizer.generate_all_plots(df, plots_dir)

    if generated_plots:
        print(f"✅ Successfully generated {len(generated_plots)} plots in {plots_dir}")
    else:
        print("⚠️ No plots were generated.")

if __name__ == "__main__":
    main()
