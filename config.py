"""
Central configuration for the Clinical Interview IR System.
All configurable parameters live here — the rubric rewards configurability.

Supports two deployment modes:
    - Cloud (Streamlit Cloud): lightweight, API-only (no torch/pyannote)
    - Local (grading): full stack with pyannote, embeddings, reranking, LiveKit
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """
    Get a secret value. Checks Streamlit Cloud secrets first,
    then falls back to environment variables (.env file).
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


# ═══════════════════════════════════════════
# Environment Detection
# ═══════════════════════════════════════════
def _detect_environment():
    """Detect whether we're running locally with full dependencies or on cloud."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False

    try:
        from sentence_transformers import SentenceTransformer
        has_embeddings = True
    except ImportError:
        has_embeddings = False

    try:
        from pyannote.audio import Pipeline
        has_pyannote = True
    except ImportError:
        has_pyannote = False

    return {
        "has_gpu": has_gpu,
        "has_embeddings": has_embeddings,
        "has_pyannote": has_pyannote,
        "is_local": has_embeddings,  # If embeddings available, we're on local
    }


ENV = _detect_environment()
IS_LOCAL = ENV["is_local"]

# ═══════════════════════════════════════════
# API Keys
# ═══════════════════════════════════════════
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
HF_TOKEN = _get_secret("HF_TOKEN")
SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_KEY")
LIVEKIT_URL = _get_secret("LIVEKIT_URL")
LIVEKIT_API_KEY = _get_secret("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = _get_secret("LIVEKIT_API_SECRET")
ASSEMBLYAI_API_KEY = _get_secret("ASSEMBLYAI_API_KEY")

# ═══════════════════════════════════════════
# Model Configuration
# ═══════════════════════════════════════════
WHISPER_MODEL = "whisper-large-v3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MODEL_FALLBACK = "llama-3.1-8b-instant"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ═══════════════════════════════════════════
# Retrieval Parameters
# ═══════════════════════════════════════════
DEFAULT_K = 5
K_VALUES = [1, 3, 5, 10]
LEXICAL_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6
RERANKER_ENABLED = IS_LOCAL  # Only on local (needs sentence-transformers)

# Search methods available per environment
SEARCH_METHODS = ["lexical", "semantic", "hybrid"] if IS_LOCAL else ["lexical"]
DEFAULT_SEARCH_METHOD = "hybrid" if IS_LOCAL else "lexical"

# ═══════════════════════════════════════════
# Diarization Strategy
# ═══════════════════════════════════════════
# Local: Pyannote (acoustic) primary, AssemblyAI fallback
# Cloud: AssemblyAI primary, LLM role labeling
DIARIZATION_PRIMARY = "pyannote" if (IS_LOCAL and ENV["has_pyannote"]) else "assemblyai"
DIARIZATION_FALLBACK = "assemblyai"

# ═══════════════════════════════════════════
# Audio Configuration
# ═══════════════════════════════════════════
MAX_AUDIO_SIZE_MB = 200
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
CHUNK_DURATION_SEC = 30
GROQ_CHUNK_LIMIT_MB = 24

# ═══════════════════════════════════════════
# Speaker Roles
# ═══════════════════════════════════════════
SPEAKER_ROLES = ["PATIENT", "CLINICIAN"]
RETRIEVAL_MODES = ["combined", "patient", "clinician"]

# ═══════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════
EVAL_METRICS = ["precision", "recall", "f1", "ndcg", "mrr", "map"]

# ═══════════════════════════════════════════
# UI Theme
# ═══════════════════════════════════════════
BRAND_PRIMARY = "#6C63FF"
BRAND_SECONDARY = "#00D4AA"
BRAND_ACCENT = "#FF6B6B"
BRAND_BG = "#0E1117"
BRAND_CARD = "#161B22"
BRAND_TEXT = "#E6EDF3"
BRAND_MUTED = "#8B949E"

PATIENT_COLOR = "#00D4AA"
PATIENT_BG = "#0D2D26"
CLINICIAN_COLOR = "#6C63FF"
CLINICIAN_BG = "#1A1740"

# ═══════════════════════════════════════════
# Ethics
# ═══════════════════════════════════════════
ETHICS_DISCLAIMER = (
    "This system is for **educational purposes only**. "
    "It does NOT provide medical diagnoses or treatment recommendations. "
    "Always consult qualified healthcare professionals for medical decisions."
)
