"""
Central configuration for the Clinical Interview IR System.
All configurable parameters live here — the rubric rewards configurability.

Secrets are loaded from two sources (in priority order):
    1. Streamlit Cloud secrets (st.secrets) — used when deployed
    2. Local .env file (via python-dotenv) — used in local development / Colab
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """
    Get a secret value. Checks Streamlit Cloud secrets first,
    then falls back to environment variables (.env file).
    """
    # Try Streamlit secrets (available when running in Streamlit)
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Fall back to environment variable
    return os.getenv(key, default)


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

# ═══════════════════════════════════════════
# Model Configuration
# ═══════════════════════════════════════════
WHISPER_MODEL = "whisper-large-v3"                          # Groq Whisper model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim vectors
EMBEDDING_DIM = 384
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
LLM_MODEL = "llama-3.3-70b-versatile"                      # Groq LLM for analysis

# ═══════════════════════════════════════════
# Retrieval Parameters
# ═══════════════════════════════════════════
DEFAULT_K = 5                       # Default number of results to retrieve
K_VALUES = [1, 3, 5, 10]           # K values tested in evaluation
LEXICAL_WEIGHT = 0.4               # Weight for BM25/FTS in hybrid ranking
SEMANTIC_WEIGHT = 0.6              # Weight for vector similarity in hybrid ranking

# ═══════════════════════════════════════════
# Audio Configuration
# ═══════════════════════════════════════════
MAX_AUDIO_SIZE_MB = 200             # Groq free tier file size limit
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
CHUNK_DURATION_SEC = 30            # For live audio chunking

# ═══════════════════════════════════════════
# Speaker Roles
# ═══════════════════════════════════════════
SPEAKER_ROLES = ["PATIENT", "CLINICIAN"]
RETRIEVAL_MODES = ["combined", "patient", "clinician"]

# ═══════════════════════════════════════════
# Ethics
# ═══════════════════════════════════════════
ETHICS_DISCLAIMER = (
    "This system is for **educational purposes only**. "
    "It does NOT provide medical diagnoses or treatment recommendations. "
    "Always consult qualified healthcare professionals for medical decisions."
)
