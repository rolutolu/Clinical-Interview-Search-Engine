"""
Main retrieval interface — search segments by query with speaker-aware filtering.

Supports three retrieval methods:
    1. Lexical (Postgres full-text search / BM25-style)
    2. Semantic (vector cosine similarity via pgvector)
    3. Hybrid (weighted combination of lexical + semantic)

And three retrieval modes (speaker-aware):
    - combined:  search all segments
    - patient:   search only PATIENT segments
    - clinician: search only CLINICIAN segments

Owner: Tolu (M3) — implement ranking logic, hybrid combination, optional reranker.

INTERFACE CONTRACT (what the Streamlit pages call):
    search(query, interview_id, mode, k, method) -> List[dict]
    Each result dict: {"segment_id", "text", "speaker_role", "start_ms", "end_ms", "score"}
"""

import config
from database.supabase_client import SupabaseClient
from typing import List, Optional

# Check if sentence-transformers is available (not installed on Streamlit Cloud)
_EMBEDDINGS_AVAILABLE = False
try:
    from retrieval.embeddings import generate_embedding
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    pass


def search(
    query: str,
    interview_id: str,
    mode: str = "combined",
    k: int = None,
    method: str = "hybrid",
    db: SupabaseClient = None,
) -> List[dict]:
    """
    Main search function — called by Streamlit pages and LLM modules.

    Args:
        query: Natural language search query
        interview_id: Which interview to search within
        mode: "combined" | "patient" | "clinician"
        k: Number of results (defaults to config.DEFAULT_K)
        method: "lexical" | "semantic" | "hybrid"
        db: SupabaseClient instance (created if not passed)

    Returns:
        List of result dicts sorted by score (descending), each with:
            segment_id, interview_id, start_ms, end_ms,
            speaker_raw, speaker_role, text, score

    Note:
        If sentence-transformers is not installed (e.g. on Streamlit Cloud),
        semantic and hybrid methods automatically fall back to lexical search.
    """
    if k is None:
        k = config.DEFAULT_K
    if db is None:
        db = SupabaseClient()

    # Map mode to speaker_role filter
    speaker_role = None
    if mode == "patient":
        speaker_role = "PATIENT"
    elif mode == "clinician":
        speaker_role = "CLINICIAN"

    # Fall back to lexical if embeddings not available
    if not _EMBEDDINGS_AVAILABLE and method in ("semantic", "hybrid"):
        method = "lexical"

    if method == "lexical":
        results = _lexical_search(db, query, k, interview_id, speaker_role)
    elif method == "semantic":
        results = _semantic_search(db, query, k, interview_id, speaker_role)
    elif method == "hybrid":
        results = _hybrid_search(db, query, k, interview_id, speaker_role)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lexical', 'semantic', or 'hybrid'.")

    return results


def _lexical_search(
    db: SupabaseClient, query: str, k: int,
    interview_id: str, speaker_role: Optional[str]
) -> List[dict]:
    """Full-text search using Postgres ts_rank."""
    results = db.text_search(query, k, interview_id, speaker_role)
    for r in results:
        r["score"] = r.pop("rank", 0.0)
    return results


def _semantic_search(
    db: SupabaseClient, query: str, k: int,
    interview_id: str, speaker_role: Optional[str]
) -> List[dict]:
    """Vector similarity search using pgvector."""
    query_embedding = generate_embedding(query)
    results = db.vector_search(query_embedding, k, interview_id, speaker_role)
    for r in results:
        r["score"] = r.pop("similarity", 0.0)
    return results


def _hybrid_search(
    db: SupabaseClient, query: str, k: int,
    interview_id: str, speaker_role: Optional[str]
) -> List[dict]:
    """
    Combine lexical + semantic results using weighted score fusion.
    Weights from config: LEXICAL_WEIGHT + SEMANTIC_WEIGHT = 1.0
    """
    # Fetch more candidates from each method, then merge
    n_candidates = k * 3

    lexical_results = _lexical_search(db, query, n_candidates, interview_id, speaker_role)
    semantic_results = _semantic_search(db, query, n_candidates, interview_id, speaker_role)

    # Normalize scores to [0, 1] range
    lexical_results = _normalize_scores(lexical_results)
    semantic_results = _normalize_scores(semantic_results)

    # Merge by segment_id with weighted scores
    combined = {}
    for r in lexical_results:
        sid = r["segment_id"]
        combined[sid] = {**r, "score": r["score"] * config.LEXICAL_WEIGHT}

    for r in semantic_results:
        sid = r["segment_id"]
        if sid in combined:
            combined[sid]["score"] += r["score"] * config.SEMANTIC_WEIGHT
        else:
            combined[sid] = {**r, "score": r["score"] * config.SEMANTIC_WEIGHT}

    # Sort by combined score and return top K
    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
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
