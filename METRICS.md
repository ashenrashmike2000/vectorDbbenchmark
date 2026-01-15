# Metric Definitions for RAGdbEval

This document provides clear, technical definitions for the metrics used in this benchmark framework. Understanding these is crucial for correctly interpreting the results.

---

## 1. Quality Metrics (Accuracy)

These metrics evaluate how "correct" the search results are compared to a pre-computed "ground truth" set of nearest neighbors.

### **Recall@K**

*   **Definition**: `(# of true neighbors found in the top K results) / (Total # of true neighbors in the ground truth)`
*   **What it means**: "What percentage of the *total* relevant items did we find by looking at the top K results?"
*   **ANN Benchmark Context**:
    *   The ground truth for datasets like SIFT1M, GIST1M, etc., typically contains the **100** true nearest neighbors for each query.
    *   Therefore, `Total # of true neighbors` is **100**.
    *   A `Recall@10 = 0.10` means that in the top 10 results, we found exactly 10 of the 100 true neighbors. This is a perfect score for K=10.
    *   This value is expected to increase as K increases.

### **Precision@K**

*   **Definition**: `(# of true neighbors found in the top K results) / K`
*   **What it means**: "Of the K items we retrieved, what percentage of them were actually correct?"
*   **Example**: A `Precision@10 = 1.0` means all 10 items retrieved were relevant.

### **MRR (Mean Reciprocal Rank)**

*   **Definition**: The average of `1 / rank` for the *first* relevant item found for each query.
*   **What it means**: "How high up in the results list is the first correct item?"
*   **Example**: An `MRR = 1.0` means that for every single query, the very first item returned was a true nearest neighbor.

---

## 2. Performance Metrics (Speed)

### **Latency p50, p90, p95, p99 (ms)**

*   **Definition**: The 50th, 90th, 95th, and 99th percentile of query latencies in milliseconds.
*   **What it means**:
    *   **p50 (Median)**: The typical query time. 50% of queries were faster than this.
    *   **p99**: The "worst-case" query time for 99% of users.

### **QPS (Queries Per Second)**

*   **Definition**: `1000 / mean_latency_ms` (for a single thread).
*   **What it means**: The average number of queries that can be served by a single thread in one second.

### **Build Time (s)**

*   **Definition**: The total time taken to insert all vectors and build the index, in seconds.

---

This clear framing ensures that the technically sound results are also easily and correctly interpretable.
