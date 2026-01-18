# VectorDB-Bench: A Comprehensive Vector Database Benchmarking Framework

A research-grade, extensible framework for evaluating vector database performance across multiple dimensions. It is designed for academic research, production system evaluation, and fair comparison of vector search technologies.

## Features

### Supported Vector Databases
| Database | Type | Index Types | Filtering | GPU |
|----------|------|-------------|-----------|-----|
| **FAISS** | Library | Flat, IVF, IVFPQ, HNSW | No | Yes |
| **Qdrant** | Database | HNSW (with quantization) | Yes | No |
| **Milvus** | Database | Flat, IVF, HNSW, DiskANN | Yes | Yes |
| **LanceDB** | Database | IVF_PQ | Yes | No |
| **Weaviate** | Database | HNSW | Yes | No |
| **Chroma** | Database | HNSW | Yes | No |
| **pgvector** | Extension | IVFFlat, HNSW | Yes | No |

### Supported Benchmark Datasets
| Dataset | Vectors | Dimensions | Type | Distance |
|---------|---------|------------|------|----------|
| **SIFT1M** | 1,000,000 | 128 | SIFT descriptors | L2 |
| **DEEP1M** | 1,000,000 | 96 | GoogLeNet features | L2 |
| **GIST1M** | 1,000,000 | 960 | Global image features | L2 |
| **GloVe** | 400,000 | 100-300 | Word embeddings | Cosine |
| **MS MARCO** | 100,000 (subset) | 768 | Text embeddings | Cosine |
| **Random** | 1,000,000 | 128 | Synthetic | L2/Cosine |

---

## Evaluation Metrics

This framework provides a comprehensive set of metrics to evaluate database performance from multiple perspectives.

> **Note on Recall**: In this benchmark, **Recall@K** is normalized by the total number of true nearest neighbors in the ground truth file (typically 100). A `Recall@10` of `0.10` indicates that all 10 of the top-10 results were correct, representing perfect performance for K=10. See `METRICS.md` for detailed definitions.

#### **Quality Metrics**
- **Recall@K** (K=1, 10, 50, 100)
- **Precision@K**
- **MRR** (Mean Reciprocal Rank)
- **NDCG@K** (Normalized Discounted Cumulative Gain)
- **MAP@K** (Mean Average Precision)
- **HitRate@K**
- **F1@K**

#### **Performance Metrics**
- **Latency Percentiles** (p50, p90, p95, p99)
- **QPS** (Queries Per Second, single-thread)
- **Cold Start Latency**
- **Warmup Time**

#### **Resource Metrics**
- **Index Build Time**
- **Index Size** (Memory & Disk)
- **Peak RAM Usage** (Client-side)
- **Bytes per Vector**
- **CPU Utilization** (Client-side)

#### **Operational Metrics**
- **Insert, Update, & Delete Latency** (Single-item)
- **Batch Insert Throughput**

---

## Installation

```bash
# Clone repository and navigate into it
git clone https://github.com/your-repo/RAGdbEval.git
cd RAGdbEval

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install all required dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download a Dataset
The benchmark runner will automatically download datasets if they are not found. However, you can also pre-download them:
```bash
python scripts/run_benchmark.py --dataset sift1m --download-only
```

### 2. Run a Full Benchmark
Use the `auto_benchmark.py` script to run multiple databases and datasets. This script ensures all results are compiled correctly before generating the final plots.

```bash
# Example: Compare FAISS and Milvus on two datasets
python scripts/auto_benchmark.py \
    --database faiss milvus \
    --dataset random msmarco \
    --runs 3 \
    --export json csv latex plots
```

## Configuration

### Main Configuration (`config/default.yaml`)

```yaml
# Select active database and dataset
database:
  active: "faiss"
  compare_all: false

dataset:
  active: "sift1m"
  compare_all: false

# Experiment settings
experiment:
  runs: 5
  warmup_queries: 1000
  seed: 42

# Query settings
query:
  k_values: [1, 10, 50, 100]
```

### Database-Specific Configuration (`config/databases/faiss.yaml`)

```yaml
index_configurations:
  - name: "HNSW32"
    type: "HNSW"
    params:
      M: 32
      efConstruction: 200
    search_params:
      efSearch: [32, 64, 128, 256, 512]
```

## Output Formats

### JSON Results
```json
{
  "metadata": {
    "generated_at": "2024-01-15T10:30:00",
    "num_results": 1
  },
  "results": [
    {
      "experiment_name": "faiss_sift1m",
      "database": "faiss",
      "dataset": "sift1m",
      "mean_metrics": {
        "recall@10": 0.9847,
        "latency_p50_ms": 0.42,
        "index_build_time_sec": 12.5
      }
    }
  ]
}
```

### LaTeX Tables
```latex
\begin{table}[htbp]
\centering
\caption{Quality Metrics Comparison}
\begin{tabular}{lcccc}
\toprule
Database & Recall@10 & Recall@100 & MRR & NDCG@10 \\
\midrule
FAISS & \textbf{0.985} & 0.998 & 0.923 & 0.967 \\
Qdrant & 0.982 & \textbf{0.999} & \textbf{0.925} & \textbf{0.968} \\
\bottomrule
\end{tabular}
\end{table}
```

### Visualization Plots
The framework generates a comprehensive set of plots for each metric, comparing all databases across different datasets.
- **Quality Plots**: `Recall@100.png`, `Precision@10.png`, etc.
- **Performance Plots**: `Latency_P50_ms.png`, `QPS.png`, etc.
- **Resource Plots**: `Build_Time_s.png`, `Index_Size_MB.png`, etc.
- **Operational Plots**: `Insert_Latency_ms.png`, `Batch_Throughput_vec_s.png`, etc.

## Project Structure

```
RAGdbEval/
├── config/                 # All YAML configuration files
│   ├── default.yaml        # Main configuration (active db/dataset, runs)
│   └── databases/          # Database-specific index and search parameters
│       ├── chroma.yaml
│       ├── faiss.yaml
│       ├── lancedb.yaml
│       ├── milvus.yaml
│       ├── pgvector.yaml
│       ├── qdrant.yaml
│       └── weaviate.yaml
├── data/                   # Default location for downloaded datasets
├── results/                # All output files from benchmark runs
├── scripts/
│   ├── auto_benchmark.py   # High-level script for running multiple benchmarks
│   ├── generate_plots.py   # Standalone script to generate plots from results.csv
│   └── run_benchmark.py    # Core script for a single benchmark task
└── src/
    ├── core/               # Core abstractions and data types
    │   ├── base.py         # VectorDBInterface abstract base class
    │   ├── config.py       # Configuration loading logic
    │   └── types.py        # All dataclasses (e.g., BenchmarkResult, Metrics)
    ├── databases/          # Adapters for each vector database
    │   ├── __init__.py     # Database factory registration
    │   ├── factory.py      # Factory function to get a database adapter
    │   ├── chroma_adapter.py
    │   ├── faiss_adapter.py
    │   ├── lancedb_adapter.py
    │   ├── milvus_adapter.py
    │   ├── pgvector_adapter.py
    │   ├── qdrant_adapter.py
    │   └── weaviate_adapter.py
    ├── datasets/           # Loaders for different HDF5/binary datasets
    │   ├── __init__.py     # Dataset factory registration
    │   ├── base.py         # DatasetLoader abstract base class
    │   ├── downloader.py   # Utility for downloading dataset files
    │   ├── factory.py      # Factory function to get a dataset loader
    │   ├── deep.py
    │   ├── gist.py
    │   ├── glove.py
    │   ├── msmarco.py
    │   ├── random_dataset.py
    │   └── sift.py
    ├── metrics/            # All metric computation logic
    │   ├── __init__.py
    │   ├── performance.py  # Latency, QPS, Cold Start, etc.
    │   ├── quality.py      # Recall, Precision, MRR, NDCG, etc.
    │   └── resource.py     # Memory, CPU, and Disk monitoring
    └── reporting/          # Exporters and visualization code
        ├── __init__.py
        ├── exporter.py     # JSON, CSV, and LaTeX exporters
        └── visualizer.py   # Plot generation logic
```

## Methodology

This framework follows established benchmarking methodologies:

1. **ANN-Benchmarks** - Standard for algorithm-level comparison
2. **VectorDBBench** - Production-realistic database benchmarking
3. **Big-ANN-Benchmarks** - NeurIPS competition methodology

### Key Principles

- **Reproducibility**: Fixed random seeds, documented configurations.
- **Fair Comparison**: Same hardware, query sets, and data.
- **Statistical Rigor**: Multiple runs to ensure stable results.
- **Recall-Performance Tradeoff**: Search parameters can be tuned to evaluate the tradeoff curve between accuracy and speed.

## Adding New Components

### Adding a New Database

1. Create an adapter in `src/databases/newdb_adapter.py`.
2. Inherit from `VectorDBInterface`.
3. Implement the required methods.
4. Register your adapter using `@register_database("newdb")`.
5. Create a configuration file in `config/databases/newdb.yaml`.

### Adding a New Dataset

1. Create a loader in `src/datasets/newdata.py`.
2. Inherit from `DatasetLoader`.
3. Implement `load_vectors`, `load_queries`, and `load_ground_truth`.
4. Create a configuration file in `config/datasets/newdata.yaml`.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{RAGdbEval_2024,
  title={RAGdbEval: A Comprehensive Benchmarking Framework for Vector Databases},
  author={Your Name/Team},
  year={2024},
  url={https://github.com/your-repo/RAGdbEval}
}
```

## References

- [ANN-Benchmarks](https://ann-benchmarks.com/)
- [VectorDBBench](https://github.com/zilliztech/VectorDBBench)
- [Big-ANN-Benchmarks](https://big-ann-benchmarks.com/)
- [MS MARCO](https://microsoft.github.io/msmarco/)
- [SIFT1M/GIST1M](http://corpus-texmex.irisa.fr/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
