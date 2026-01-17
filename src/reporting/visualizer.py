"""
Visualization utilities for benchmark results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class BenchmarkVisualizer:
    def __init__(self, style: str = "seaborn-v0_8-whitegrid", figsize: tuple = (14, 8), dpi: int = 300):
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("ggplot")
        self.figsize = figsize
        self.dpi = dpi

    def generate_all_plots(self, df: pd.DataFrame, output_dir: str) -> List[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if df.empty:
            print("⚠️ No valid results to plot.")
            return []

        # Define the mapping from original CSV column names to pretty plot labels
        metric_map = {
            "recall@100": "Recall@100",
            "precision@10": "Precision@10",
            "mrr": "MRR",
            "ndcg@100": "NDCG@100",
            "map@100": "MAP@100",
            "hit_rate@10": "HitRate@10",
            "latency_p50_ms": "Latency P50 (ms)",
            "latency_p99_ms": "Latency P99 (ms)",
            "qps_single_thread": "QPS",
            "index_build_time_sec": "Build Time (s)",
            "index_size_mb": "Index Size (MB)",
            "ram_mb_peak": "RAM Peak (MB)",
            "bytes_per_vector": "Bytes per Vector",
            "insert_latency_single_ms": "Insert Latency (ms)",
            "update_latency_ms": "Update Latency (ms)",
            "delete_latency_ms": "Delete Latency (ms)",
        }

        # Rename columns for plotting
        plot_df = df.rename(columns=metric_map)
        
        generated_plots = []

        # Define metrics to plot using the NEW (pretty) names
        metrics_to_plot = [
            ("Recall@100", "viridis"),
            ("Precision@10", "plasma"),
            ("MRR", "magma"),
            ("NDCG@100", "cividis"),
            ("MAP@100", "inferno"),
            ("HitRate@10", "cividis"),
            ("Latency P50 (ms)", "rocket"),
            ("Latency P99 (ms)", "rocket_r"),
            ("QPS", "viridis"),
            ("Build Time (s)", "mako"),
            ("Index Size (MB)", "cubehelix"),
            ("RAM Peak (MB)", "YlGnBu"),
            ("Bytes per Vector", "coolwarm"),
            ("Insert Latency (ms)", "BuPu"),
            ("Update Latency (ms)", "GnBu"),
            ("Delete Latency (ms)", "OrRd"),
        ]

        # Generate one plot per metric, comparing all experiments
        for metric_name, palette in metrics_to_plot:
            if metric_name not in plot_df.columns:
                print(f"⚠️ Metric '{metric_name}' not found in results, skipping plot.")
                continue

            # Sanitize metric name for filename
            filename_metric = metric_name.replace("@", "_").replace(" ", "_").replace("(", "").replace(")", "")
            
            generated_plots.append(self._create_grouped_bar_plot(
                plot_df, 
                "database", 
                metric_name, 
                "dataset",
                f"Comparison of {metric_name}",
                output_dir / f"{filename_metric}.png", 
                palette
            ))

        return generated_plots

    def _create_grouped_bar_plot(self, df, x, y, hue, title, filename, palette):
        plt.figure(figsize=self.figsize)
        
        ax = sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette)
        
        # Add labels to each bar, but only if there are not too many bars
        if len(df) < 30:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, rotation=45, fontsize=8)
            
        plt.title(title, fontsize=16, weight='bold')
        plt.xlabel(None)
        plt.ylabel(y, fontsize=12)
        plt.xticks(rotation=0) # No rotation needed for database names
        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        return str(filename)