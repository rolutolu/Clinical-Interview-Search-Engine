"""
Evaluation metrics for retrieval quality.

Required by rubric:
    - Precision@K
    - Recall@K
    - Reported for: overall, patient-only, clinician-only
    - Multiple K values

Above and beyond:
    - F1@K (harmonic mean of P@K and R@K)
    - nDCG@K (Normalized Discounted Cumulative Gain)
    - MRR (Mean Reciprocal Rank)
    - MAP (Mean Average Precision)

Research basis:
    - Manning et al. 2008: IR evaluation metrics
    - Jarvelin & Kekalainen 2002: nDCG
    - Voorhees 1999: TREC evaluation methodology
"""

from typing import List, Dict, Callable
import math
import config


# ═══════════════════════════════════════════
# Core Metrics (required)
# ═══════════════════════════════════════════

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Precision@K: fraction of top-K results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for sid in top_k if sid in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Recall@K: fraction of all relevant documents found in top-K."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


# ═══════════════════════════════════════════
# Extended Metrics (above and beyond)
# ═══════════════════════════════════════════

def f1_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """F1@K: harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    Uses binary relevance (1 if relevant, 0 if not).
    DCG = sum(rel_i / log2(i+2)) for i in 0..k-1
    IDCG = same formula for ideal ranking (all relevant first)

    Reference: Jarvelin & Kekalainen, 2002
    """
    if not relevant_ids or k <= 0:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # DCG: actual ranking
    dcg = 0.0
    for i, sid in enumerate(top_k):
        rel = 1.0 if sid in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # IDCG: ideal ranking (all relevant docs at the top)
    n_relevant_in_k = min(len(relevant_ids), k)
    idcg = 0.0
    for i in range(n_relevant_in_k):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Reciprocal Rank: 1/rank of the first relevant result.
    Used to compute MRR across queries.
    """
    relevant_set = set(relevant_ids)
    for i, sid in enumerate(retrieved_ids):
        if sid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Average Precision for a single query.
    AP = (1/|R|) * sum(P@k * rel(k)) for k=1..n

    Used to compute MAP across queries.
    Reference: Manning et al. 2008
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = 0
    sum_precision = 0.0
    for i, sid in enumerate(retrieved_ids):
        if sid in relevant_set:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant_set)


# ═══════════════════════════════════════════
# Aggregate computation
# ═══════════════════════════════════════════

def compute_all_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> Dict[str, float]:
    """Compute all metrics for a single query at a given K."""
    return {
        "precision": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall": recall_at_k(retrieved_ids, relevant_ids, k),
        "f1": f1_at_k(retrieved_ids, relevant_ids, k),
        "ndcg": ndcg_at_k(retrieved_ids, relevant_ids, k),
        "mrr": reciprocal_rank(retrieved_ids[:k], relevant_ids),
        "ap": average_precision(retrieved_ids[:k], relevant_ids),
    }


def aggregate_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Average metrics across multiple queries."""
    if not per_query_metrics:
        return {m: 0.0 for m in ["precision", "recall", "f1", "ndcg", "mrr", "map"]}

    n = len(per_query_metrics)
    agg = {}
    for key in ["precision", "recall", "f1", "ndcg", "mrr"]:
        agg[key] = sum(m.get(key, 0.0) for m in per_query_metrics) / n
    # MAP = mean of AP
    agg["map"] = sum(m.get("ap", 0.0) for m in per_query_metrics) / n
    agg["num_queries"] = n
    return agg
