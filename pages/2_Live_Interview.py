"""
Page 2: Live Clinical Interview via LiveKit

LiveKit provides real-time speaker separation by giving each participant
their own audio track — no diarization needed ("perfect attribution").

This feature runs locally or in Colab due to WebRTC audio requirements.
The Streamlit Cloud version displays the architecture and setup instructions.
"""

import streamlit as st
import config
import pandas as pd

st.set_page_config(page_title="Live Interview", page_icon="LI", layout="wide")

st.title("Live Clinical Interview")
st.markdown(
    f'<div style="background:linear-gradient(135deg,#fff3cd,#ffeeba);'
    f'border:1px solid #ffc107;border-radius:8px;padding:0.8rem;'
    f'color:#856404;font-size:0.9rem;">{config.ETHICS_DISCLAIMER}</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════
# Architecture Explanation
# ══════════════════════════════════════════
st.subheader("How Live Mode Works")

st.markdown("""
**Live mode uses [LiveKit](https://livekit.io) for real-time clinical interviews with automatic speaker separation.**

Unlike the offline pipeline (which requires acoustic diarization to figure out who spoke when),
LiveKit assigns each participant their own audio track — giving us **perfect speaker attribution**
with zero diarization error.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Offline Pipeline")
    st.markdown("""
```
    Mixed Audio File
        ↓
    Acoustic Diarization (who spoke?)
        ↓
    Transcription (what was said?)
        ↓
    Alignment (match text to speaker)
        ↓
    Store in Supabase
```
    *Speaker separation is approximate*
    """)

with col2:
    st.markdown("### Live Pipeline")
    st.markdown("""
```
    LiveKit Room (2+ participants)
        ↓
    Track 1 (Clinician) → Transcribe
    Track 2 (Patient)   → Transcribe
        ↓
    Perfect Speaker Labels
        ↓
    Store in Supabase
```
    *Speaker separation is exact*
    """)

st.divider()

# ══════════════════════════════════════════
# Technical Details
# ══════════════════════════════════════════
st.subheader("Technical Architecture")

st.markdown("""
The live interview system uses these components:

1. **LiveKit Cloud** (free tier: 50 participant-minutes/month)
   - Creates a real-time audio room
   - Each participant joins with a role token (CLINICIAN or PATIENT)
   - Audio tracks are separated per-participant at the infrastructure level

2. **Groq Whisper API** (same as offline pipeline)
   - Each participant's audio track is transcribed independently
   - Timestamps are relative to the session start

3. **Supabase** (same database as offline)
   - Segments are stored with `source_mode = "live"` and perfect `speaker_role` labels
   - Immediately available for retrieval and analysis on the Query Analysis page

**Key advantage:** Because LiveKit separates audio tracks at the network level,
there is zero diarization error rate (DER = 0%). This is the gold standard
for speaker attribution in clinical settings.
""")

st.divider()

# ══════════════════════════════════════════
# Setup Instructions
# ══════════════════════════════════════════
st.subheader("Setup & Demo Instructions")

st.info(
    "**Live interview requires a local environment** (not Streamlit Cloud) "
    "because it needs browser microphone access via WebRTC. "
    "Follow the readme instructions to run the demo locally."
)

with st.expander("Step 1: Get LiveKit Credentials (Free)", expanded=False):
    st.markdown("""
    1. Go to **https://cloud.livekit.io** and sign up (free)
    2. Create a new project
    3. Go to **Settings → Keys** and copy:
       - `LIVEKIT_URL` (starts with `wss://`)
       - `LIVEKIT_API_KEY` (starts with `API`)
       - `LIVEKIT_API_SECRET`
    4. Add these to your local `.env` file
    """)

with st.expander("Step 2: Run Locally", expanded=False):
    st.markdown("""
```bash
    # Clone the repo
    git clone https://github.com/YOUR_USERNAME/Clinical-Interview-Search-Engine.git
    cd Clinical-Interview-Search-Engine

    # Install dependencies (including livekit)
    pip install -r requirements-full.txt

    # Run the app locally
    python -m streamlit run app.py
```

    Then navigate to the **Live Interview** page in your local Streamlit app.
    """)

with st.expander("Step 3: Conduct a Live Interview Demo", expanded=False):
    st.markdown("""
    1. Click **Create Room** — generates a LiveKit room
    2. Open **two browser tabs** (or two devices):
       - Tab 1 joins as **Clinician** (using the clinician join link)
       - Tab 2 joins as **Patient** (using the patient join link)
    3. Conduct a short interview (2-5 minutes is sufficient)
    4. Click **End Interview** — processes audio tracks and stores segments
    5. Go to **Query Analysis** to search and analyze the live interview
    """)

st.divider()

# ══════════════════════════════════════════
# Comparison Table
# ══════════════════════════════════════════
st.subheader("Offline vs Live Pipeline Comparison")

comparison_data = {
    "Feature": [
        "Speaker Separation Method",
        "Diarization Error Rate",
        "Transcription",
        "Max Speakers",
        "Real-time Processing",
        "Requires GPU",
        "Works on Streamlit Cloud",
        "Audio Input",
    ],
    "Offline Pipeline": [
        "AssemblyAI acoustic diarization",
        "~5-15% (acoustic model dependent)",
        "AssemblyAI + Groq Whisper",
        "10+ (AssemblyAI limit)",
        "No (batch processing)",
        "No (API-based)",
        "Yes",
        "Uploaded audio file",
    ],
    "Live Pipeline": [
        "LiveKit track separation",
        "0% (perfect attribution)",
        "Groq Whisper per-track",
        "Limited by LiveKit room",
        "Yes (streaming)",
        "No (API-based)",
        "No (needs local WebRTC)",
        "Live microphone",
    ],
}

st.dataframe(
    pd.DataFrame(comparison_data).set_index("Feature"),
    use_container_width=True,
)
