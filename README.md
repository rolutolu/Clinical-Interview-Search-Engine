# 🏥 Clinical Interview Analysis & Retrieval System

**CP423 — Text Retrieval & Search Engines | Winter 2026**

An end-to-end conversational IR system for clinical interviews with speaker-aware indexing, retrieval, and grounded LLM analysis.

## Features

- **Offline Pipeline**: Upload audio → Pyannote diarization → Whisper transcription → speaker alignment → indexed in Supabase
- **Live Pipeline**: LiveKit real-time audio → per-speaker transcription → instant indexing
- **Speaker-Aware Retrieval**: Query patient-only, clinician-only, or combined segments
- **Hybrid Search**: Lexical (BM25/FTS) + Semantic (pgvector) with configurable weights
- **Grounded LLM Analysis**: Summarization, Symptom QA, and Interview Analyzer — all with segment citations
- **Evaluation Dashboard**: Precision@K and Recall@K across retrieval modes and K values

## Quick Start (< 30 minutes)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/clinical-interview-ir.git
cd clinical-interview-ir
pip install -r requirements.txt
```

### 2. Get API Keys (all free tier)

| Service | URL | What You Need |
|---------|-----|---------------|
| **Supabase** | https://supabase.com | Project URL + anon key |
| **Groq** | https://console.groq.com/keys | API key |
| **HuggingFace** | https://huggingface.co/settings/tokens | Access token |
| **LiveKit** | https://cloud.livekit.io | URL + API key + secret |

**Important:** For HuggingFace, you must also accept the model agreements:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Set Up Database

1. Go to your Supabase project → **SQL Editor** → **New Query**
2. Paste the contents of `database/schema.sql`
3. Click **Run**

### 5. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

OR

Run Demo Cloud App: https://clinical-interview-search-engine-af5wvzrrvndruvwainck9e.streamlit.app/

## Project Structure

```
clinical-interview-ir/
├── app.py                      # Main Streamlit entry point
├── config.py                   # Central configuration
├── requirements.txt
├── .env.example
├── pages/                      # Streamlit multi-page app
│   ├── 1_📤_Upload_Offline.py  # Offline pipeline
│   ├── 2_🎤_Live_Interview.py  # LiveKit real-time
│   ├── 3_🔍_Query_Analysis.py  # Retrieval + LLM analysis
│   └── 4_📊_Evaluation.py      # Precision@K / Recall@K
├── audio/                      # Audio processing modules
│   ├── diarize.py              # Pyannote speaker diarization
│   ├── transcribe.py           # Groq Whisper transcription
│   ├── align.py                # Whisper↔Pyannote alignment
│   └── livekit_handler.py      # LiveKit real-time handler
├── database/                   # Data layer
│   ├── models.py               # Segment, Interview, PatientProfile
│   ├── supabase_client.py      # All DB operations
│   └── schema.sql              # Supabase table definitions
├── retrieval/                  # IR modules
│   ├── embeddings.py           # Sentence-transformer embeddings
│   └── search.py               # Lexical + semantic + hybrid search
├── llm/                        # Grounded LLM modules
│   ├── prompts.py              # System prompts with ethics guardrails
│   └── grounded_llm.py         # Summarizer, QA, Analyzer
├── evaluation/                 # Metrics
│   ├── metrics.py              # Precision@K, Recall@K
│   └── eval_data/              # Ground truth labels
├── docs/                       # Documentation
└── tests/                      # Unit tests
```

## Team

| Member | Role | Modules |
|--------|------|---------|
| Javier (M1) | Infra + Frontend + Orchestration | app.py, pages/*, database/*, config.py, audio/livekit_handler.py |
| Josh (M2) | Audio Processing | audio/diarize.py, audio/transcribe.py, audio/align.py |
| Tolu (M3) | Retrieval + Evaluation + LLM | retrieval/*, llm/*, evaluation/* |

## Tech Stack

- **Frontend**: Streamlit
- **Database**: Supabase (Postgres + pgvector)
- **Speech-to-Text**: Groq Whisper API
- **Diarization**: pyannote.audio
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Groq Llama 3.3 70B
- **Live Audio**: LiveKit

All services use **free tiers only**.

## Ethics Disclaimer

⚠️ This system is for **educational purposes only**. It does NOT provide medical diagnoses or treatment recommendations. Always consult qualified healthcare professionals for medical decisions.
