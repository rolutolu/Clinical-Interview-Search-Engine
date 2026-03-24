# System Architecture

## Overview

ClinIR is a conversational information retrieval system designed for clinical interviews. It processes spoken multi-speaker audio into searchable, speaker-labeled transcript segments, then provides grounded LLM-powered analysis.

## Design Principles

1. **Speaker-awareness throughout**: Every segment carries speaker identity from ingestion through retrieval to LLM output
2. **Dual-mode processing**: Offline (uploaded audio) and Live (real-time WebRTC) share the same indexing and retrieval layer
3. **Grounded generation**: All LLM outputs cite source segments — no hallucination
4. **Graceful degradation**: System detects available capabilities (GPU, embeddings, APIs) and adjusts automatically
5. **Free-tier compliance**: All external services operate within free usage limits

## Pipeline Architecture

### Offline Pipeline
```
Audio File (.mp3/.wav)
    │
    ├─── [Local] Pyannote 3.1.1 (GPU) ──→ Diarization segments [{start, end, speaker}]
    │         └─── Groq Whisper ──→ Transcription segments [{start, end, text}]
    │               └─── Temporal Alignment ──→ Merged segments with confidence scores
    │
    └─── [Cloud] AssemblyAI API ──→ Utterances with acoustic speaker labels
    │
    ├─── Groq LLM ──→ Role labeling (PATIENT/CLINICIAN) + name identification
    │
    ├─── sentence-transformers ──→ 384-dim MiniLM embeddings
    │
    └─── Supabase ──→ Indexed segments (metadata + vectors + tsvector)
```

### Live Pipeline
```
LiveKit Room (WebRTC)
    │
    ├─── Track 1 (Clinician mic) ──→ Groq Whisper ──→ Segments (role=CLINICIAN)
    └─── Track 2 (Patient mic)   ──→ Groq Whisper ──→ Segments (role=PATIENT)
    │
    └─── Supabase ──→ Indexed segments (DER = 0%, perfect attribution)
```

### Retrieval & Analysis
```
User Query
    │
    ├─── Speaker Filter (combined / patient / clinician)
    │
    ├─── Lexical Search (Postgres ts_rank_cd)
    ├─── Semantic Search (pgvector cosine similarity)
    └─── Hybrid Search (Reciprocal Rank Fusion, k=60)
         │
         └─── [Optional] Cross-Encoder Reranking (ms-marco-MiniLM)
              │
              └─── Groq LLM (Llama 3.3 70B)
                   ├─── Summarization (8-section clinical summary)
                   ├─── Symptom QA (CoT reasoning + citations)
                   └─── Interview Analyzer (8-section structured analysis)
```

## Environment Detection

The system auto-detects its runtime environment via `config._detect_environment()`:

| Capability | Local (Grading) | Streamlit Cloud |
|-----------|----------------|-----------------|
| Pyannote diarization | ✅ (GPU) | ❌ (no torch) |
| Semantic search | ✅ (sentence-transformers) | ❌ |
| Hybrid search | ✅ (RRF fusion) | ❌ |
| Cross-encoder reranking | ✅ | ❌ |
| AssemblyAI diarization | ✅ (fallback) | ✅ (primary) |
| Lexical search | ✅ | ✅ |
| LLM analysis | ✅ (streaming) | ✅ (streaming) |
| LiveKit | ✅ (local WebRTC) | ❌ (info page) |

## Data Model

The **Segment** is the atomic unit of the entire system:
```python
@dataclass
class Segment:
    interview_id: str      # FK → interviews table
    segment_id: str        # Primary key (uuid[:8])
    start_ms: int          # Temporal position
    end_ms: int
    speaker_raw: str       # "Dr. Dan (CLINICIAN_1)"
    speaker_role: str      # "PATIENT" or "CLINICIAN"
    text: str              # Transcribed content
    source_mode: str       # "offline" or "live"
    embedding: List[float] # 384-dim MiniLM vector
    keywords: List[str]    # Extracted entities
```

Every retrieval result, LLM citation, and evaluation label references Segments.

## Database Schema (Supabase)

- `interviews`: Metadata, speaker map, source mode
- `segments`: Text + embeddings (pgvector 384-dim) + tsvector FTS index
- `patient_profiles`: Optional patient context (text or voice input)
- `eval_labels`: Ground truth relevance labels for evaluation

Speaker-aware indexes enable filtered queries on `(interview_id, speaker_role)`.

## Ethics Enforcement

Ethics constraints are enforced at three levels:

1. **Prompt-level**: System prompt explicitly forbids diagnoses/treatment recommendations
2. **UI-level**: Ethics banner displayed on every page
3. **Output-level**: LLM outputs must cite source segments (grounding prevents fabrication)
