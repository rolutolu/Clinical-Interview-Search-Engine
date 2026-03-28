"""
Embedding generation and management using sentence-transformers.

Model: all-MiniLM-L6-v2 (384-dim vectors)
    - Lightweight, fast inference (CPU or GPU)
    - Strong clinical/medical text similarity performance
    - Normalized embeddings for cosine similarity via dot product

Supports:
    - Single text embedding (for queries)
    - Batch embedding (for indexing segments on upload)
    - Automatic storage to Supabase via bulk_update_embeddings

Research basis:
    - Reimers & Gurevych 2019: Sentence-BERT for semantic similarity
    - Used in clinical NLP for patient note retrieval (Jiang et al. 2023)
"""

import config
from typing import List, Optional

# ── Lazy-loaded models (heavy imports) ──
_embedding_model = None
_reranker_model = None


def _get_embedding_model():
    """Load sentence-transformer model (cached after first call)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embeddings] Loading model: {config.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"[Embeddings] Model loaded ({config.EMBEDDING_DIM}-dim).")
    return _embedding_model


def _get_reranker_model():
    """Load cross-encoder reranker model (cached after first call)."""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        print(f"[Reranker] Loading model: {config.RERANKER_MODEL}")
        _reranker_model = CrossEncoder(config.RERANKER_MODEL)
        print("[Reranker] Model loaded.")
    return _reranker_model


def generate_embedding(text: str) -> List[float]:
    """
    Generate a single 384-dim embedding vector for a text string.
    Used for query embedding at search time.
    """
    model = _get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings_batch(
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts efficiently (for indexing).

    Args:
        texts: List of transcript segment texts
        batch_size: Encoding batch size
        show_progress: Show progress bar during encoding

    Returns:
        List of 384-dim vectors, one per input text
    """
    model = _get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
    return [e.tolist() for e in embeddings]


def embed_and_store_segments(
    segments: List[dict],
    db,
    batch_size: int = 32,
    progress_callback=None,
) -> int:
    """
    Generate embeddings for segments and store them in Supabase.

    Args:
        segments: List of segment dicts (must have segment_id and text)
        db: SupabaseClient instance
        batch_size: Encoding batch size
        progress_callback: Optional callable(msg) for progress updates

    Returns:
        Number of segments with embeddings stored
    """
    def _log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    texts = [s.get("text", "") for s in segments]
    ids = [s.get("segment_id", "") for s in segments]

    _log(f"Generating embeddings for {len(texts)} segments...")
    embeddings = generate_embeddings_batch(texts, batch_size=batch_size)

    _log("Storing embeddings in Supabase...")
    pairs = [{"segment_id": sid, "embedding": emb} for sid, emb in zip(ids, embeddings)]

    # Batch update in groups of 50 to avoid timeout
    stored = 0
    for i in range(0, len(pairs), 50):
        batch = pairs[i:i + 50]
        db.bulk_update_embeddings(batch)
        stored += len(batch)
        _log(f"  Stored {stored}/{len(pairs)} embeddings...")

    _log(f"Embedding storage complete: {stored} vectors stored.")
    return stored


def rerank_results(
    query: str,
    results: List[dict],
    top_k: Optional[int] = None,
) -> List[dict]:
    """
    Rerank retrieval results using a cross-encoder model.

    Cross-encoders process (query, document) pairs jointly, producing
    more accurate relevance scores than bi-encoder similarity alone.

    Args:
        query: Search query
        results: List of segment dicts with "text" field
        top_k: Return only top K after reranking (None = return all)

    Returns:
        Results sorted by cross-encoder relevance score (descending)
    """
    if not results or not config.RERANKER_ENABLED:
        return results

    model = _get_reranker_model()

    # Build (query, document) pairs
    pairs = [(query, r.get("text", "")) for r in results]
    scores = model.predict(pairs)

    # Attach scores and sort
    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)

    results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    if top_k:
        results = results[:top_k]

    return results
