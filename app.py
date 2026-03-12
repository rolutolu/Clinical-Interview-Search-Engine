"""
Clinical Interview IR System — Main Application

Streamlit multi-page app for intelligent clinical interview analysis,
summarization, and retrieval with speaker separation.

Run with: streamlit run app.py
"""

import streamlit as st
import config

# ══════════════════════════════════════════
# Page Configuration
# ══════════════════════════════════════════
st.set_page_config(
    page_title="Clinical Interview IR System",
    page_icon="CI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════
# Custom Styling
# ══════════════════════════════════════════
st.markdown("""
<style>
    /* Clean, professional medical theme */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 2rem;
    }
    .status-card {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .speaker-patient {
        background-color: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 4px 4px 0;
    }
    .speaker-clinician {
        background-color: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 4px 4px 0;
    }
    .ethics-banner {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
        color: #856404;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Sidebar — System Status
# ══════════════════════════════════════════
with st.sidebar:
    st.title("Clinical IR System")
    st.caption("CP423 - Text Retrieval & Search Engines")

    st.divider()

    # Connection status checks
    st.subheader("System Status")

    # Supabase
    supabase_ok = bool(config.SUPABASE_URL and config.SUPABASE_KEY)
    st.write(f"{'[OK]' if supabase_ok else '[X]'} Supabase Database")

    # Groq
    groq_ok = bool(config.GROQ_API_KEY)
    st.write(f"{'[OK]' if groq_ok else '[X]'} Groq API (Whisper + LLM)")

    # HuggingFace
    hf_ok = bool(config.HF_TOKEN)
    st.write(f"{'[OK]' if hf_ok else '[X]'} HuggingFace (Pyannote)")

    # LiveKit
    lk_ok = bool(config.LIVEKIT_URL and config.LIVEKIT_API_KEY)
    st.write(f"{'[OK]' if lk_ok else '[!]'} LiveKit (optional for live mode)")

    if not all([supabase_ok, groq_ok, hf_ok]):
        st.warning("Some API keys are missing. Check your `.env` file.")

    st.divider()

    # Configuration display
    st.subheader("Configuration")
    st.write(f"**Whisper Model:** {config.WHISPER_MODEL}")
    st.write(f"**Embedding Model:** {config.EMBEDDING_MODEL.split('/')[-1]}")
    st.write(f"**LLM Model:** {config.LLM_MODEL}")
    st.write(f"**Default K:** {config.DEFAULT_K}")
    st.write(f"**Hybrid Weights:** Lex={config.LEXICAL_WEIGHT} / Sem={config.SEMANTIC_WEIGHT}")

# ══════════════════════════════════════════
# Main Page — Home / Dashboard
# ══════════════════════════════════════════
st.markdown('<p class="main-header">Clinical Interview Analysis & Retrieval System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent speaker-aware retrieval for clinical interview transcripts</p>', unsafe_allow_html=True)

# Ethics disclaimer
st.markdown(f'<div class="ethics-banner">{config.ETHICS_DISCLAIMER}</div>', unsafe_allow_html=True)


# Quick overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### Upload")
    st.write("Upload clinical interview audio for offline processing with speaker diarization.")
    st.page_link("pages/1_Upload_Offline.py", label="Go to Upload")

with col2:
    st.markdown("### Live")
    st.write("Conduct real-time interviews with LiveKit speaker separation.")
    st.page_link("pages/2_Live_Interview.py", label="Go to Live")

with col3:
    st.markdown("### Query")
    st.write("Search, summarize, and analyze interviews with grounded LLM output.")
    st.page_link("pages/3_Query_Analysis.py", label="Go to Query")

with col4:
    st.markdown("### Evaluate")
    st.write("Run Precision@K and Recall@K evaluation across retrieval modes.")
    st.page_link("pages/4_Evaluation.py", label="Go to Evaluation")


# ══════════════════════════════════════════
# System Architecture Overview
# ══════════════════════════════════════════
st.divider()
st.subheader("System Architecture")

st.markdown("""
```
Offline:  Audio File → Pyannote Diarization → Whisper Transcription → Indexing → Retrieval → LLM Analysis
Live:     LiveKit Streams → Speaker-Separated Audio → Whisper → Indexing → Retrieval → LLM Analysis
```
""")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Retrieval Modes (Speaker-Aware)**")
    st.write("• **Combined** — search all segments")
    st.write("• **Patient Only** — filter to patient speech")
    st.write("• **Clinician Only** — filter to clinician speech")

with col_b:
    st.markdown("**Analysis Modules (Grounded)**")
    st.write("• **Summarization** — structured interview summary with citations")
    st.write("• **Symptom QA** — question answering from retrieved segments")
    st.write("• **Interview Analyzer** — timeline, symptoms, meds, follow-ups")
