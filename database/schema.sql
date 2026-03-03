-- ══════════════════════════════════════════════════════════════
-- Clinical Interview IR System — Supabase Schema
-- ══════════════════════════════════════════════════════════════
-- Run this ONCE in: Supabase Dashboard → SQL Editor → New Query
-- ══════════════════════════════════════════════════════════════

-- 1. Enable pgvector extension for embedding search
CREATE EXTENSION IF NOT EXISTS vector;

-- ──────────────────────────────────────────
-- 2. Interviews table
-- ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS interviews (
    interview_id    TEXT PRIMARY KEY,
    title           TEXT,
    source_mode     TEXT DEFAULT 'offline' CHECK (source_mode IN ('offline', 'live')),
    audio_filename  TEXT,
    duration_ms     INTEGER DEFAULT 0,
    speaker_map     JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ──────────────────────────────────────────
-- 3. Segments table (the core of the system)
-- ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS segments (
    segment_id      TEXT PRIMARY KEY,
    interview_id    TEXT REFERENCES interviews(interview_id) ON DELETE CASCADE,
    start_ms        INTEGER NOT NULL,
    end_ms          INTEGER NOT NULL,
    speaker_raw     TEXT,
    speaker_role    TEXT CHECK (speaker_role IN ('PATIENT', 'CLINICIAN')),
    text            TEXT NOT NULL,
    source_mode     TEXT DEFAULT 'offline',
    embedding       vector(384),
    keywords        TEXT[] DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast speaker-aware retrieval
CREATE INDEX IF NOT EXISTS idx_segments_interview
    ON segments(interview_id);
CREATE INDEX IF NOT EXISTS idx_segments_role
    ON segments(speaker_role);
CREATE INDEX IF NOT EXISTS idx_segments_interview_role
    ON segments(interview_id, speaker_role);

-- Full-text search index (for lexical / BM25-style retrieval)
ALTER TABLE segments ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS idx_segments_fts
    ON segments USING GIN(fts);

-- ──────────────────────────────────────────
-- 4. Patient profiles
-- ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patient_profiles (
    profile_id      TEXT PRIMARY KEY,
    interview_id    TEXT REFERENCES interviews(interview_id) ON DELETE CASCADE,
    name            TEXT,
    age             INTEGER,
    chief_complaint TEXT,
    medical_history TEXT,
    input_method    TEXT DEFAULT 'text',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ──────────────────────────────────────────
-- 5. Evaluation labels (for Precision@K / Recall@K)
-- ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_labels (
    query_id            TEXT,
    query_text          TEXT,
    relevant_segment_ids TEXT[],
    retrieval_mode      TEXT DEFAULT 'combined'
        CHECK (retrieval_mode IN ('combined', 'patient', 'clinician')),
    PRIMARY KEY (query_id, retrieval_mode)
);

-- ──────────────────────────────────────────
-- 6. RPC: Vector similarity search function
-- ──────────────────────────────────────────
-- Called from Python: supabase.rpc("match_segments", {...})
CREATE OR REPLACE FUNCTION match_segments(
    query_embedding     vector(384),
    match_count         int DEFAULT 5,
    filter_interview_id text DEFAULT NULL,
    filter_speaker_role text DEFAULT NULL
)
RETURNS TABLE (
    segment_id      text,
    interview_id    text,
    start_ms        int,
    end_ms          int,
    speaker_raw     text,
    speaker_role    text,
    text            text,
    source_mode     text,
    similarity      float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.segment_id,
        s.interview_id,
        s.start_ms,
        s.end_ms,
        s.speaker_raw,
        s.speaker_role,
        s.text,
        s.source_mode,
        1 - (s.embedding <=> query_embedding) AS similarity
    FROM segments s
    WHERE
        (filter_interview_id IS NULL OR s.interview_id = filter_interview_id)
        AND (filter_speaker_role IS NULL OR s.speaker_role = filter_speaker_role)
        AND s.embedding IS NOT NULL
    ORDER BY s.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ──────────────────────────────────────────
-- 7. RPC: Full-text search function
-- ──────────────────────────────────────────
CREATE OR REPLACE FUNCTION search_segments_text(
    query_text          text,
    match_count         int DEFAULT 5,
    filter_interview_id text DEFAULT NULL,
    filter_speaker_role text DEFAULT NULL
)
RETURNS TABLE (
    segment_id      text,
    interview_id    text,
    start_ms        int,
    end_ms          int,
    speaker_raw     text,
    speaker_role    text,
    text            text,
    source_mode     text,
    rank            float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.segment_id,
        s.interview_id,
        s.start_ms,
        s.end_ms,
        s.speaker_raw,
        s.speaker_role,
        s.text,
        s.source_mode,
        ts_rank_cd(s.fts, websearch_to_tsquery('english', query_text)) AS rank
    FROM segments s
    WHERE
        s.fts @@ websearch_to_tsquery('english', query_text)
        AND (filter_interview_id IS NULL OR s.interview_id = filter_interview_id)
        AND (filter_speaker_role IS NULL OR s.speaker_role = filter_speaker_role)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;
