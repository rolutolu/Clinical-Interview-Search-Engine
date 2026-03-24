# Retrieval & Ranking Design

## Overview

The retrieval system implements a two-stage pipeline: candidate retrieval (lexical, semantic, or hybrid) followed by optional cross-encoder reranking. All stages support speaker-aware filtering.

## Stage 1: Candidate Retrieval

### Lexical Search (BM25-style)

Uses Postgres full-text search with `ts_rank_cd` scoring.

- **Indexing**: Each segment's text is stored as a `tsvector` column with GIN index
- **Query**: Converted to `tsquery` with Postgres `websearch_to_tsquery()`
- **Scoring**: `ts_rank_cd` (cover density ranking) — rewards proximity of query terms
- **Strengths**: Exact keyword matching, medical terminology, proper nouns
- **Weaknesses**: No semantic understanding, misses synonyms

### Semantic Search (Vector Similarity)

Uses pgvector cosine similarity on 384-dim MiniLM embeddings.

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters)
- **Indexing**: Embeddings generated on upload, stored in `vector(384)` column
- **Query**: Query text is embedded at search time, matched via `1 - cosine_distance`
- **Strengths**: Semantic similarity, synonym handling, paraphrase matching
- **Weaknesses**: May match semantically similar but factually irrelevant segments

### Hybrid Search (Reciprocal Rank Fusion)

Combines lexical and semantic results using RRF (Cormack et al. 2009).
```
RRF_score(d) = Σ 1/(k + rank_i(d))  for each system i
```

Where `k=60` (standard constant). RRF is preferred over raw score fusion because:
- Scores from different systems are not directly comparable
- RRF only uses rank positions, making it scale-invariant
- Robust to outlier scores in either system

**Process**: Fetch 3K candidates from each method → compute RRF scores → return top K.

## Stage 2: Cross-Encoder Reranking (Optional)

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Cross-encoders process (query, document) pairs jointly through a transformer, producing more accurate relevance scores than bi-encoder similarity. This is computationally expensive but applied only to the top candidates from Stage 1.

**Impact**: Typically improves P@5 by 10-20% on clinical queries.

## Speaker-Aware Filtering

All search methods support three modes:
- **combined**: Search all segments (no filter)
- **patient**: `WHERE speaker_role = 'PATIENT'`
- **clinician**: `WHERE speaker_role = 'CLINICIAN'`

This enables queries like:
- "What symptoms did the patient report?" (patient mode)
- "What questions did the clinician ask?" (clinician mode)
- "What was discussed about medications?" (combined mode)

## References

- Robertson & Zaragoza 2009: The Probabilistic Relevance Framework: BM25 and Beyond
- Reimers & Gurevych 2019: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- Cormack et al. 2009: Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods
- Nogueira & Cho 2019: Passage Re-ranking with BERT
