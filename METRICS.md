# Metric Definitions for RAGdbEval

This document provides clear, technical definitions for the metrics used in this benchmark framework. Understanding these is crucial for correctly interpreting the results.

---

## 1. Quality Metrics (Accuracy)

These metrics evaluate how "correct" the search results are compared to a pre-computed "ground truth" set of nearest neighbors.

### **Recall@K**

*   **Definition**: `(# of true neighbors found in the top K results) / (Total # of true neighbors in the ground truth)`
*   **What it means**: "What percentage of the *total* relevant items did we find by looking at the top K results?"
*   **ANN Benchmark Context**: The ground truth for datasets like SIFT1M typically contains 100 true nearest neighbors. Therefore, a `Recall@10 = 0.10` means the system correctly found 10 of the 100 true neighbors, which is a perfect score for the top 10 items.

### **Precision@K**

*   **Definition**: `(# of true neighbors found in the top K results) / K`
*   **What it means**: "Of the K items we retrieved, what percentage of them were actually correct?"

### **MRR (Mean Reciprocal Rank)**

*   **Definition**: The average of `1 / rank` for the *first* relevant item found for each query.
*   **What it means**: "How high up in the results list is the first correct item?" An MRR of 1.0 means the first result was always relevant.

### **NDCG@K (Normalized Discounted Cumulative Gain)**

*   **Definition**: A measure of ranking quality that gives more credit for finding relevant items at higher ranks.
*   **What it means**: It evaluates the "gain" of a result based on its position, rewarding systems that rank relevant items first.

### **MAP@K (Mean Average Precision)**

*   **Definition**: The mean of the Average Precision (AP) scores across all queries. AP provides a single-figure measure of quality across recall levels.

### **HitRate@K**

*   **Definition**: The fraction of queries where at least one relevant item is found in the top K results.
*   **What it means**: "Did we find *any* correct item in the top K?"

### **F1@K**

*   **Definition**: `2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)`
*   **What it means**: The harmonic mean of Precision and Recall, providing a single score that balances both metrics.

---

## 2. Performance Metrics (Speed)

### **Latency p50, p90, p95, p99 (ms)**

*   **Definition**: The 50th, 90th, 95th, and 99th percentile of query latencies in milliseconds.
*   **What it means**: p50 is the typical query time; p99 represents a "worst-case" query time.

### **QPS (Queries Per Second)**

*   **Definition**: `1000 / mean_latency_ms` (for a single thread).
*   **What it means**: The average number of queries that can be served by a single thread in one second.

### **Cold Start Latency (ms)**

*   **Definition**: The latency of the very first query after an index is loaded, before any caches are warmed up.
*   **Note**: This metric is not measured in the default benchmark run, which focuses on "warm" performance.

### **Warmup Time (ms)**

*   **Definition**: The time or number of queries required for the database's query latency to stabilize.
*   **Note**: This metric is not measured in the default benchmark run.

---

## 3. Resource Metrics (Footprint)

### **Build Time (s)**

*   **Definition**: The total time taken to insert all vectors and build the index, in seconds.

### **Index Size (MB)**

*   **Definition**: The estimated size of the vector index on disk or in memory, in megabytes. The accuracy depends on the database adapter.

### **RAM Peak (MB)**

*   **Definition**: The peak memory usage of the benchmark client process during the index build phase. This does not include the database server's own memory usage.

### **CPU Utilization (%)**

*   **Definition**: The average CPU utilization across all cores during the index build phase. This measures the CPU usage of the client-side benchmark process.

### **Bytes per Vector**

*   **Definition**: `index_size_bytes / num_vectors`. The average number of bytes used to store a single vector and its associated index overhead.

---

## 4. Operational Metrics (CRUD)

### **Insert Latency (ms)**

*   **Definition**: The time taken to insert a single, new vector into the existing index.

### **Update Latency (ms)**

*   **Definition**: The time taken to update an existing vector with a new vector.

### **Delete Latency (ms)**

*   **Definition**: The time taken to delete a single vector from the index.

### **Batch Throughput (vec/s)**

*   **Definition**: `num_vectors / index_build_time_sec`. The effective number of vectors inserted per second during the initial bulk import.
