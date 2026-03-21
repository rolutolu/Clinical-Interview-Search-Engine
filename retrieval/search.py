"""
Main retrieval interface — speaker-aware search with multiple ranking strategies.

Supports three retrieval methods:
    1. Lexical: Postgres full-text search (BM25-style ts_rank)
    2. Semantic: Vector cosine similarity via pgvector
    3. Hybrid: Weighted combination of lexical + semantic scores

And three speaker-aware retrieval modes:
    - combined:  search all segments
    - patient:   search only PATIENT segments
    - clinician: search only CLINICIAN segments

Advanced features (local mode):
    - Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
    - Reciprocal Rank Fusion as alternative to weighted combination

Research basis:
    - Robertson & Zaragoza 2009: BM25 for lexical retrieval
    - Karpukhin et al. 2020: Dense passage retrieval
    - Cormack et al. 2009: Reciprocal Rank Fusion
"""

import config
from database.supabase_client import SupabaseClient
from typing import List, Optional

# Check for local capabilities
_HAS_EMBEDDINGS = False
_HAS_RERANKER = False
try:
    from retrieval.embeddings import generate_embedding, rerank_results
    _HAS_EMBEDDINGS = True
    _HAS_RERANKER = config.RERANKER_ENABLED
except ImportError:
    pass


def search(
    query: str,
    interview_id: str,
    mode: str = "combined",
    k: int = None,
    method: str = None,
    db: SupabaseClient = None,
    rerank: bool = None,
) -> List[dict]:
    """
    Main search function — called by all Streamlit pages and LLM modules.

    Args:
        query: Natural language search query
        interview_id: Which interview to search within
        mode: "combined" | "patient" | "clinician"
        k: Number of results (defaults to config.DEFAULT_K)
        method: "lexical" | "semantic" | "hybrid" (defaults to config.DEFAULT_SEARCH_METHOD)
        db: SupabaseClient instance
        rerank: Whether to apply cross-encoder reranking (defaults to config.RERANKER_ENABLED)

    Returns:
        List of result dicts sorted by score (descending)
    """
    if k is None:
        k = config.DEFAULT_K
    if method is None:
        method = config.DEFAULT_SEARCH_METHOD
    if db is None:
        db = SupabaseClient()
    if rerank is None:
        rerank = _HAS_RERANKER

    # Map mode to speaker_role filter
    speaker_role = None
    if mode == "patient":
        speaker_role = "PATIENT"
    elif mode == "clinician":
        speaker_role = "CLINICIAN"

    # Fall back to lexical if embeddings not available
    if not _HAS_EMBEDDINGS and method in ("semantic", "hybrid"):
        method = "lexical"

    # Fetch more candidates if reranking (reranker picks best from larger pool)
    fetch_k = k * 3 if rerank else k

    if method == "lexical":
        results = _lexical_search(db, query, fetch_k, interview_id, speaker_role)
    elif method == "semantic":
        results = _semantic_search(db, query, fetch_k, interview_id, speaker_role)
    elif method == "hybrid":
        results = _hybrid_search(db, query, fetch_k, interview_id, speaker_role)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lexical', 'semantic', or 'hybrid'.")

    # Apply cross-encoder reranking if enabled
    if rerank and _HAS_RERANKER and results:
        results = rerank_results(query, results, top_k=k)
    else:
        results = results[:k]

    return results


def _lexical_search(
    db: SupabaseClient, query: str, k: int,
    interview_id: str, speaker_role: Optional[str],
) -> List[dict]:
    """Full-text search using Postgres ts_rank_cd."""
    results = db.text_search(query, k, interview_id, speaker_role)
    for r in results:
        r["score"] = r.pop("rank", 0.0)
        r["method"] = "lexical"
    return results


def _semantic_search(
    db: SupabaseClient, query: str, k: int,
    interview_id: str, speaker_role: Optional[str],
) -> List[dict]:
    """Vector similarity search using pgvector cosine distance."""
    query_embedding = generate_embedding(query)
    results = db.vector_search(query_embedding, k, interview_id, speaker_role)
    for r in results:
        r["score"] = r.pop("similarity", 0.0)
        r["method"] = "semantic"
    return results


def _hybrid_search(
    db: SupabaseClient, query: str, k: int,
    interview_id: str, speaker_role: Optional[str],
) -> List[dict]:
    """
    Combine lexical + semantic results using Reciprocal Rank Fusion (RRF).

    RRF (Cormack et al. 2009) is preferred over raw score combination because:
    - Scores from different systems are not directly comparable
    - RRF only uses rank positions, making it scale-invariant
    - Formula: RRF_score(d) = sum( 1 / (k_rrf + rank_i(d)) ) for each system i

    Falls back to weighted score fusion if one method returns no results.
    """
    n_candidates = k * 3

    lexical_results = _lexical_search(db, query, n_candidates, interview_id, speaker_role)
    semantic_results = _semantic_search(db, query, n_candidates, interview_id, speaker_role)

    # If one method returns nothing, use the other
    if not lexical_results:
        return semantic_results[:k]
    if not semantic_results:
        return lexical_results[:k]

    # Reciprocal Rank Fusion (k_rrf=60 is standard)
    k_rrf = 60
    rrf_scores = {}

    for rank, r in enumerate(lexical_results):
        sid = r["segment_id"]
        rrf_scores[sid] = {"data": r, "score": 0.0}
        rrf_scores[sid]["score"] += 1.0 / (k_rrf + rank + 1)

    for rank, r in enumerate(semantic_results):
        sid = r["segment_id"]
        if sid not in rrf_scores:
            rrf_scores[sid] = {"data": r, "score": 0.0}
        rrf_scores[sid]["score"] += 1.0 / (k_rrf + rank + 1)

    # Build ranked list
    ranked = []
    for sid, entry in rrf_scores.items():
        result = entry["data"]
        result["score"] = round(entry["score"], 6)
        result["method"] = "hybrid"
        ranked.append(result)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:k]


def _normalize_scores(results: List[dict]) -> List[dict]:
    """Normalize scores to [0, 1] range using min-max scaling."""
    if not results:
        return results
    scores = [r["score"] for r in results]
    min_s = min(scores)
    max_s = max(scores)
    range_s = max_s - min_s if max_s != min_s else 1.0
    for r in results:
        r["score"] = (r["score"] - min_s) / range_s
    return results
