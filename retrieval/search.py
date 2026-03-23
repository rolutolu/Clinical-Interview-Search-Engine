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
    Combine lexical + semantic results using Reciprocal Rank Fusion (RRF)
    with chronological fallback when lexical returns too few results.

    This ensures hybrid ALWAYS produces different rankings from pure semantic
    by fusing semantic similarity ranks with a second signal (keyword or temporal).

    RRF formula: score(d) = sum(1 / (k_rrf + rank_i(d))) for each system i
    Reference: Cormack et al. 2009
    """
    n_candidates = k * 3

    lexical_results = _lexical_search(db, query, n_candidates, interview_id, speaker_role)
    semantic_results = _semantic_search(db, query, n_candidates, interview_id, speaker_role)

    if not semantic_results and not lexical_results:
        return []
    if not semantic_results:
        return lexical_results[:k]

    # If lexical returned too few results, supplement with chronological ordering
    # This gives hybrid a second ranking signal that differs from semantic
    if len(lexical_results) < 2:
        all_segs = db.get_segments(interview_id, speaker_role=speaker_role)
        # Use chronological order as a baseline ranking
        seen = {r["segment_id"] for r in lexical_results}
        for seg in all_segs:
            if seg["segment_id"] not in seen:
                seg["score"] = 0.001
                seg["method"] = "chronological"
                lexical_results.append(seg)

    k_rrf = 60

    # Build rank lookups from each system
    lex_rank = {r["segment_id"]: i for i, r in enumerate(lexical_results)}
    sem_rank = {r["segment_id"]: i for i, r in enumerate(semantic_results)}

    # Collect all unique segments
    all_segments = {}
    for r in lexical_results:
        all_segments[r["segment_id"]] = r
    for r in semantic_results:
        all_segments[r["segment_id"]] = r

    # Penalty rank for segments still not in one list
    penalty_rank = max(len(lexical_results), len(semantic_results)) + 10

    # Compute RRF scores
    rrf_scores = []
    for sid, data in all_segments.items():
        lr = lex_rank.get(sid, penalty_rank)
        sr = sem_rank.get(sid, penalty_rank)
        score = (1.0 / (k_rrf + lr + 1)) + (1.0 / (k_rrf + sr + 1))
        result = {**data}
        result["score"] = round(score, 6)
        result["method"] = "hybrid"
        rrf_scores.append(result)

    rrf_scores.sort(key=lambda x: x["score"], reverse=True)
    return rrf_scores[:k]

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
