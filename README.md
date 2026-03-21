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
