"""
Central configuration for the Clinical Interview IR System.
All configurable parameters live here — the rubric rewards configurability.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════
# API Keys
# ═══════════════════════════════════════════
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

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
MAX_AUDIO_SIZE_MB = 25             # Groq free tier file size limit
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
    "⚠️ This system is for **educational purposes only**. "
    "It does NOT provide medical diagnoses or treatment recommendations. "
    "Always consult qualified healthcare professionals for medical decisions."
)
