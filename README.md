# ClinIR — Clinical Interview Retrieval System

[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://clinical-interview-search-engine-af5wvzrrvndruvwainck9e.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Educational-blue)]()

**CP423 — Text Retrieval & Search Engines | Wilfrid Laurier University | Winter 2026**

An end-to-end conversational information retrieval system for clinical interviews with speaker-aware indexing, hybrid retrieval, and grounded LLM analysis.

> ⚠️ **Educational purposes only.** This system does NOT provide medical diagnoses or treatment recommendations.

---

## Features

- **Dual Diarization Pipeline**: Pyannote (GPU, acoustic) + AssemblyAI (API, fallback) + LLM role/name identification
- **LiveKit Live Interviews**: Real-time WebRTC audio with perfect speaker attribution (DER = 0%)
- **Voice Patient Profiles**: Text or voice input (Groq Whisper) for patient context
- **Hybrid Search**: Lexical (BM25/FTS) + Semantic (pgvector cosine) + Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: ms-marco-MiniLM-L-6-v2 for precision retrieval
- **Grounded LLM Analysis**: Summarization, Symptom QA, 8-section Interview Analyzer — all with segment citations
- **Streaming Output**: Token-by-token LLM display with automatic fallback
- **6 Evaluation Metrics**: P@K, R@K, F1@K, nDCG@K, MRR, MAP with LLM-as-Judge labels
- **Professional Dark UI**: Medical-themed interface with speaker timeline visualization

---

## Quick Start (< 30 minutes)

### 1. Clone & Install
```bash
git clone https://github.com/rolutolu/Clinical-Interview-Search-Engine.git
cd Clinical-Interview-Search-Engine

# For local/grading version (full stack with GPU support):
pip install -r requirements.txt

# Or for cloud-only version:
pip install -r requirements.txt  # on main branch
```

### 2. Get API Keys (all free tier)

| Service | URL | What You Need | Free Tier |
|---------|-----|---------------|-----------|
| **Supabase** | https://supabase.com | Project URL + anon key | 500 MB database |
| **Groq** | https://console.groq.com/keys | API key | 14,400 audio-sec/day |
| **HuggingFace** | https://huggingface.co/settings/tokens | Access token | Unlimited inference |
| **AssemblyAI** | https://www.assemblyai.com | API key | 100 hours audio |
| **LiveKit** | https://cloud.livekit.io | URL + API key + secret | 50 participant-min/month |

**HuggingFace Model Access** (required for Pyannote):
- Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
- Accept terms at: https://huggingface.co/pyannote/segmentation-3.0

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Set Up Supabase Database

1. Create a new project at https://supabase.com
2. Go to **SQL Editor → New Query**
3. Paste contents of `database/schema.sql`
4. Click **Run**

### 5. Run the App
```bash
python -m streamlit run app.py
```

Open http://localhost:8501 — the sidebar will show "Local (Full Stack)" with all capabilities active.

---

## System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        AUDIO INPUT                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Upload Audio  │    │ LiveKit Room │    │ Voice Profile│       │
│  │   (Offline)   │    │   (Live)     │    │  (Whisper)   │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
└─────────┼───────────────────┼───────────────────┼───────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SPEAKER SEPARATION                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Pyannote   │    │   LiveKit    │    │  AssemblyAI  │       │
│  │  (GPU local) │    │(track split) │    │  (API cloud) │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │    DER ~5-15%     │    DER = 0%       │   DER ~5%     │
└─────────┼───────────────────┼───────────────────┼───────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TRANSCRIPTION                                │
│           Groq Whisper v3 (API) — auto-chunking                 │
│           + LLM Role Labeling (PATIENT / CLINICIAN)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INDEXING (Supabase)                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Segments    │    │  pgvector    │    │  Full-Text   │       │
│  │  (metadata)  │    │ (384-dim)    │    │  (tsvector)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         Speaker-aware indexes: patient / clinician / combined   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL                                   │
│  Lexical (BM25) + Semantic (Cosine) → Reciprocal Rank Fusion   │
│  Optional: Cross-Encoder Reranking (ms-marco-MiniLM)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM ANALYSIS (Groq)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Summarizer   │  │  Symptom QA  │  │  Analyzer    │          │
│  │ (8 sections) │  │  (CoT + RAG) │  │ (8 sections) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│       All outputs grounded with [S_id time ROLE] citations      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure
```
Clinical-Interview-Search-Engine/
├── app.py                          # Main Streamlit entry point + dashboard
├── config.py                       # Central config with environment detection
├── requirements.txt                # Full local dependencies
├── .env.example                    # Environment variable template
├── packages.txt                    # System packages for Streamlit Cloud
├── .streamlit/config.toml          # Dark theme configuration
│
├── pages/                          # Streamlit multi-page app
│   ├── 1_Upload_Offline.py         # Dual diarization + voice profiles + embeddings
│   ├── 2_Live_Interview.py         # LiveKit room management + track processing
│   ├── 3_Query_Analysis.py         # Hybrid search + streaming LLM analysis
│   └── 4_Evaluation.py             # 6-metric evaluation + LLM-as-Judge
│
├── audio/                          # Audio processing pipeline
│   ├── diarize.py                  # Pyannote + AssemblyAI dual backend
│   ├── transcribe.py               # Groq Whisper with auto-chunking
│   ├── align.py                    # Temporal alignment with confidence scoring
│   └── livekit_handler.py          # LiveKit room/token/track management
│
├── database/                       # Data layer
│   ├── models.py                   # Segment, Interview, PatientProfile dataclasses
│   ├── supabase_client.py          # All DB operations (CRUD + search RPCs)
│   └── schema.sql                  # Supabase table definitions + pgvector + FTS
│
├── retrieval/                      # Information retrieval
│   ├── embeddings.py               # MiniLM embeddings + cross-encoder reranking
│   └── search.py                   # Lexical + semantic + hybrid (RRF) search
│
├── llm/                            # Grounded LLM modules
│   ├── prompts.py                  # System prompts with CoT + ethics constraints
│   └── grounded_llm.py             # Streaming summarizer, QA, analyzer
│
├── evaluation/                     # IR evaluation
│   ├── metrics.py                  # P@K, R@K, F1@K, nDCG@K, MRR, MAP
│   └── eval_data/                  # Ground truth labels
│
├── docs/                           # Documentation
│   ├── architecture.md             # System architecture deep dive
│   ├── retrieval_design.md         # Retrieval & ranking methodology
│   └── evaluation_results.md       # Evaluation methodology & analysis
│
└── tests/                          # Tests
    └── test_connection.py          # Supabase smoke test
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Multi-page app with dark medical theme |
| Database | Supabase (Postgres + pgvector) | Segments, interviews, vector search, FTS |
| Speech-to-Text | Groq Whisper v3 | Transcription with auto-chunking |
| Diarization (Local) | pyannote.audio 3.1.1 | GPU-based acoustic speaker separation |
| Diarization (Cloud) | AssemblyAI | API-based acoustic diarization |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 384-dim vectors for semantic search |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Precision reranking of retrieval results |
| LLM | Groq Llama 3.3 70B | Grounded analysis with segment citations |
| Live Audio | LiveKit | WebRTC real-time speaker separation |

**All services use FREE tiers only.**

---

## Team

| Member | Role | Key Modules |
|--------|------|-------------|
| Javier (M1) | Infrastructure, Frontend, Orchestration | app.py, pages/*, database/*, config.py, livekit_handler.py |
| Josh (M2) | Audio Processing Pipeline | audio/diarize.py, audio/transcribe.py, audio/align.py |
| Tolu (M3) | Retrieval, Evaluation, LLM | retrieval/*, llm/*, evaluation/* |

---

## Videos

| # | Video | Duration | Content |
|---|-------|----------|---------|
| 1 | System Design | 5-10 min | Architecture overview, design decisions |
| 2 | API Setup | 5-10 min | Environment setup, API configuration |
| 3 | LiveKit Demo | 5-10 min | Room creation, live interview, track processing |
| 4 | Pyannote Demo | 5-10 min | Offline pipeline, diarization, alignment |
| 5 | End-to-End Walkthrough | 5-10 min | Full use case with patient profile, retrieval, LLM output |

---

## License

This project is for educational purposes as part of CP423 at Wilfrid Laurier University.
```

---

**File 2: `.env.example`** (new file in repo root):
```
# ═══════════════════════════════════════════
# Clinical Interview IR System — Environment Variables
# ═══════════════════════════════════════════
# Copy this file to .env and fill in your API keys.
# All services use FREE tiers only.
# ═══════════════════════════════════════════

# Supabase (https://supabase.com → Project Settings → API)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here

# Groq (https://console.groq.com/keys)
GROQ_API_KEY=gsk_your-key-here

# HuggingFace (https://huggingface.co/settings/tokens)
# Also accept: https://huggingface.co/pyannote/speaker-diarization-3.1
HF_TOKEN=hf_your-token-here

# AssemblyAI (https://www.assemblyai.com → Dashboard)
ASSEMBLYAI_API_KEY=your-key-here

# LiveKit (https://cloud.livekit.io → Settings → Keys)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIyour-key-here
LIVEKIT_API_SECRET=your-secret-here
