"""
Evaluation metrics for retrieval quality.

Required by rubric:
    - Precision@K
    - Recall@K
    - Reported for: overall, patient-only, clinician-only
    - Multiple K values: 1, 3, 5, 10

Owner: Tolu (M3) — validate, extend with additional metrics if desired.

INTERFACE CONTRACT:
    precision_at_k(retrieved_ids, relevant_ids, k) -> float
    recall_at_k(retrieved_ids, relevant_ids, k) -> float
    run_evaluation(eval_labels, search_fn) -> dict   # full evaluation report
"""

from typing import List, Dict, Callable
import config


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Precision@K: Of the top K retrieved results, what fraction are relevant?

    Args:
        retrieved_ids: Ordered list of retrieved segment IDs (by rank)
        relevant_ids: Set of truly relevant segment IDs for the query
        k: Cut-off rank

    Returns:
        Float in [0.0, 1.0]
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for sid in top_k if sid in relevant_set)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Recall@K: Of all relevant segments, what fraction appear in the top K?

    Args:
        retrieved_ids: Ordered list of retrieved segment IDs (by rank)
        relevant_ids: Set of truly relevant segment IDs for the query
        k: Cut-off rank

    Returns:
        Float in [0.0, 1.0]
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def run_evaluation(
    eval_labels: List[Dict],
    search_fn: Callable,
    k_values: List[int] = None,
    interview_id: str = None,
) -> Dict:
    """
    Run full evaluation across all queries, modes, and K values.

    Args:
        eval_labels: List of dicts with query_id, query_text, relevant_segment_ids, retrieval_mode
        search_fn: Function with signature search(query, interview_id, mode, k) -> List[dict]
        k_values: List of K values to test (default: config.K_VALUES)
        interview_id: Interview to evaluate against

    Returns:
        Dict with structure:
        {
            "overall": {1: {"precision": 0.8, "recall": 0.6}, 3: {...}, ...},
            "patient": {1: {...}, ...},
            "clinician": {1: {...}, ...},
        }
    """
    if k_values is None:
        k_values = config.K_VALUES

    # Group labels by mode
    by_mode = {"combined": [], "patient": [], "clinician": []}
    for label in eval_labels:
        mode = label.get("retrieval_mode", "combined")
        by_mode[mode].append(label)

    results = {}
    mode_name_map = {"combined": "overall", "patient": "patient", "clinician": "clinician"}

    for mode, labels in by_mode.items():
        if not labels:
            continue
        mode_results = {}
        for k in k_values:
            precisions = []
            recalls = []
            for label in labels:
                query = label["query_text"]
                relevant = label["relevant_segment_ids"]

                # Run search
                search_results = search_fn(
                    query=query,
                    interview_id=interview_id or "",
                    mode=mode,
                    k=k,
                )
                retrieved = [r["segment_id"] for r in search_results]

                precisions.append(precision_at_k(retrieved, relevant, k))
                recalls.append(recall_at_k(retrieved, relevant, k))

            mode_results[k] = {
                "precision": sum(precisions) / len(precisions) if precisions else 0.0,
                "recall": sum(recalls) / len(recalls) if recalls else 0.0,
                "num_queries": len(labels),
            }

        results[mode_name_map[mode]] = mode_results

    return results
