"""
Embedding generation using sentence-transformers.

Model: all-MiniLM-L6-v2 (384-dim vectors)
    - Small, fast, no GPU needed for inference
    - Good quality for medical/clinical text similarity

Owner: Tolu (M3) — implement and optimize.
"""

import config
from typing import List


# ── Lazy-loaded model (heavy import) ──
_model = None


def _get_model():
    """Load sentence-transformer model (cached after first call)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("Embedding model loaded.")
    return _model


def generate_embedding(text: str) -> List[float]:
    """
    Generate a 384-dim embedding vector for a text string.

    Args:
        text: Input text (a transcript segment)

    Returns:
        List of 384 floats
    """
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for multiple texts efficiently.

    Args:
        texts: List of input texts
        batch_size: Batch size for encoding

    Returns:
        List of 384-dim vectors, one per input text
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    return [e.tolist() for e in embeddings]
