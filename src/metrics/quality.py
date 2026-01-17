"""
Quality metrics for retrieval accuracy evaluation.

Implements standard information retrieval metrics following:
    - ANN-Benchmarks methodology
    - TREC evaluation standards
    - MTEB Leaderboard metrics
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.types import QualityMetrics


def compute_recall_at_k(
    retrieved: NDArray[np.int64],
    ground_truth: NDArray[np.int64],
    k: int,
) -> float:
    """
    Compute Recall@K - fraction of true neighbors found in top-k results.

    This is the PRIMARY metric used in ANN-Benchmarks.
    Definition: (Relevant items in Top K) / (Total Relevant items in GT)

    Args:
        retrieved: Retrieved indices of shape (n_queries, k)
        ground_truth: True neighbor indices of shape (n_queries, gt_k)
        k: Number of results to consider

    Returns:
        Mean recall across all queries
    """
    n_queries = len(retrieved)
    recalls = []

    for i in range(n_queries):
        retrieved_k = set(retrieved[i, :k])
        true_neighbors = set(ground_truth[i])

        retrieved_k.discard(-1)
        true_neighbors.discard(-1)

        if len(true_neighbors) == 0:
            recalls.append(1.0)
        else:
            intersection = len(retrieved_k & true_neighbors)

            # Standard Recall Definition: Intersection / Total Relevant
            denominator = len(true_neighbors)

            if denominator == 0:
                recalls.append(0.0)
            else:
                recalls.append(intersection / denominator)

    return np.mean(recalls)


def compute_precision_at_k(
    retrieved: NDArray[np.int64],
    ground_truth: NDArray[np.int64],
    k: int,
) -> float:
    """
    Compute Precision@K - fraction of retrieved items that are relevant.

    Args:
        retrieved: Retrieved indices of shape (n_queries, k)
        ground_truth: True neighbor indices
        k: Number of results to consider

    Returns:
        Mean precision across all queries
    """
    n_queries = len(retrieved)
    precisions = []

    for i in range(n_queries):
        retrieved_k = set(retrieved[i, :k])

        # Precision also checks against the full set of known neighbors
        true_neighbors = set(ground_truth[i])  # Checks against ALL 100 GT items

        retrieved_k.discard(-1)
        true_neighbors.discard(-1)

        if len(retrieved_k) == 0:
            precisions.append(0.0)
        else:
            intersection = len(retrieved_k & true_neighbors)
            precisions.append(intersection / len(retrieved_k))

    return np.mean(precisions)


def compute_mrr(
    retrieved: NDArray[np.int64],
    ground_truth: NDArray[np.int64],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR focuses on the rank of the FIRST relevant result.
    Particularly useful for question-answering systems.

    Args:
        retrieved: Retrieved indices of shape (n_queries, k)
        ground_truth: True neighbor indices

    Returns:
        Mean reciprocal rank
    """
    n_queries = len(retrieved)
    reciprocal_ranks = []

    for i in range(n_queries):
        true_neighbors = set(ground_truth[i])
        true_neighbors.discard(-1)

        rr = 0.0
        for rank, idx in enumerate(retrieved[i], start=1):
            if idx in true_neighbors:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

    return np.mean(reciprocal_ranks)


def compute_ndcg_at_k(
    retrieved: NDArray[np.int64],
    ground_truth: NDArray[np.int64],
    k: int,
    relevance_scores: Optional[NDArray[np.float64]] = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).

    NDCG accounts for the position of relevant items, with higher
    positions contributing more to the score.

    This is the default metric used in the MTEB Retrieval Leaderboard.

    Args:
        retrieved: Retrieved indices of shape (n_queries, k)
        ground_truth: True neighbor indices
        k: Number of results to consider

    Returns:
        Mean NDCG@K
    """
    n_queries = len(retrieved)
    ndcg_scores = []

    for i in range(n_queries):
        # For NDCG, the ideal set is strictly the best K from ground truth
        # But for 'dcg', we check if retrieved items are in the FULL ground truth
        all_true_neighbors = set(ground_truth[i])
        all_true_neighbors.discard(-1)

        # IDCG considers the 'best possible' k items
        ideal_neighbors_k = ground_truth[i, :k]

        # Compute DCG
        dcg = 0.0
        for rank, idx in enumerate(retrieved[i, :k], start=1):
            if idx in all_true_neighbors:
                rel = 1.0
                dcg += rel / np.log2(rank + 1)

        # Compute IDCG (ideal DCG)
        idcg = 0.0
        for rank in range(1, min(k, len(ideal_neighbors_k)) + 1):
            rel = 1.0
            idcg += rel / np.log2(rank + 1)

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return np.mean(ndcg_scores)


def compute_map_at_k(
    retrieved: NDArray[np.int64],
    ground_truth: NDArray[np.int64],
    k: int,
) -> float:
    """
    Compute Mean Average Precision (MAP@K).

    MAP is the mean of Average Precision across all queries.
    Average Precision is the area under the precision-recall curve.

    Args:
        retrieved: Retrieved indices of shape (n_queries, k)
        ground_truth: True neighbor indices
        k: Number of results to consider

    Returns:
        Mean Average Precision
    """
    n_queries = len(retrieved)
    average_precisions = []

    for i in range(n_queries):
        true_neighbors = set(ground_truth[i])
        true_neighbors.discard(-1)

        if len(true_neighbors) == 0:
            average_precisions.append(1.0)
            continue

        hits = 0
        precision_sum = 0.0
        found_relevant = set()

        # We track `found_relevant` to ensure that each distinct relevant item contributes at most once to Average Precision.
        for rank, idx in enumerate(retrieved[i, :k], start=1): # Iterate up to k
            if idx in true_neighbors and idx not in found_relevant:
                hits += 1
                precision_sum += hits / rank
                found_relevant.add(idx)

        # Normalize by the maximum possible number of distinct relevant items that can appear in the top-k results, i.e.,
        # min(len(true_neighbors), k).
        # The duplicate handling above (via `found_relevant`) ensures each relevant item contributes at most once to the precision sum.
        num_relevant = min(len(true_neighbors), k)
        if num_relevant > 0:
            average_precisions.append(precision_sum / num_relevant)
        else:
            average_precisions.append(0.0)

    return np.mean(average_precisions)


def compute_hit_rate_at_k(
    retrieved: NDArray[np.int64],
    ground_truth: NDArray[np.int64],
    k: int,
) -> float:
    """
    Compute Hit Rate@K (Success@K).

    Hit rate is the fraction of queries where at least one relevant
    item appears in the top-k results.

    Args:
        retrieved: Retrieved indices
        ground_truth: True neighbor indices
        k: Number of results to consider

    Returns:
        Hit rate
    """
    n_queries = len(retrieved)
    hits = 0

    for i in range(n_queries):
        retrieved_k = set(retrieved[i, :k])
        true_neighbors = set(ground_truth[i])

        retrieved_k.discard(-1)
        true_neighbors.discard(-1)

        if len(retrieved_k & true_neighbors) > 0:
            hits += 1

    return hits / n_queries


def compute_f1_at_k(
    retrieved: NDArray[np.int64],
    ground_truth: NDArray[np.int64],
    k: int,
) -> float:
    """
    Compute F1@K - harmonic mean of precision and recall.

    Args:
        retrieved: Retrieved indices
        ground_truth: True neighbor indices
        k: Number of results to consider

    Returns:
        F1 score
    """
    precision = compute_precision_at_k(retrieved, ground_truth, k)
    recall = compute_recall_at_k(retrieved, ground_truth, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_all_quality_metrics(
    retrieved: NDArray[np.int64],
    ground_truth: NDArray[np.int64],
    k_values: List[int] = [1, 10, 50, 100],
) -> QualityMetrics:
    """
    Compute all quality metrics at once.

    Args:
        retrieved: Retrieved indices of shape (n_queries, max_k)
        ground_truth: True neighbor indices
        k_values: List of k values to evaluate

    Returns:
        QualityMetrics dataclass with all metrics
    """
    metrics = QualityMetrics()

    # Recall
    if 1 in k_values:
        metrics.recall_at_1 = compute_recall_at_k(retrieved, ground_truth, 1)
    if 10 in k_values:
        metrics.recall_at_10 = compute_recall_at_k(retrieved, ground_truth, 10)
    if 50 in k_values:
        metrics.recall_at_50 = compute_recall_at_k(retrieved, ground_truth, 50)
    if 100 in k_values:
        metrics.recall_at_100 = compute_recall_at_k(retrieved, ground_truth, 100)

    # Precision
    if 1 in k_values:
        metrics.precision_at_1 = compute_precision_at_k(retrieved, ground_truth, 1)
    if 10 in k_values:
        metrics.precision_at_10 = compute_precision_at_k(retrieved, ground_truth, 10)
    if 50 in k_values:
        metrics.precision_at_50 = compute_precision_at_k(retrieved, ground_truth, 50)
    if 100 in k_values:
        metrics.precision_at_100 = compute_precision_at_k(retrieved, ground_truth, 100)

    # MRR
    metrics.mrr = compute_mrr(retrieved, ground_truth)

    # NDCG
    if 10 in k_values:
        metrics.ndcg_at_10 = compute_ndcg_at_k(retrieved, ground_truth, 10)
    if 100 in k_values:
        metrics.ndcg_at_100 = compute_ndcg_at_k(retrieved, ground_truth, 100)

    # MAP
    if 10 in k_values:
        metrics.map_at_10 = compute_map_at_k(retrieved, ground_truth, 10)
    if 100 in k_values:
        metrics.map_at_100 = compute_map_at_k(retrieved, ground_truth, 100)

    # Hit Rate
    if 1 in k_values:
        metrics.hit_rate_at_1 = compute_hit_rate_at_k(retrieved, ground_truth, 1)
    if 10 in k_values:
        metrics.hit_rate_at_10 = compute_hit_rate_at_k(retrieved, ground_truth, 10)

    # F1
    if 10 in k_values:
        metrics.f1_at_10 = compute_f1_at_k(retrieved, ground_truth, 10)

    return metrics