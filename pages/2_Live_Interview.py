"""
Page 2: Live Interview via LiveKit

Real-time clinical interview with automatic speaker separation.
Each participant gets their own audio track — no diarization needed.

This page will be fully implemented in Phase 4.
"""

import streamlit as st
import config

st.set_page_config(page_title="Live Interview", page_icon="LI", layout="wide")

st.title("Live Clinical Interview")
st.markdown(f'<div style="background:linear-gradient(135deg,#fff3cd,#ffeeba);border:1px solid #ffc107;border-radius:8px;padding:0.8rem;color:#856404;font-size:0.9rem;">{config.ETHICS_DISCLAIMER}</div>', unsafe_allow_html=True)

# Check LiveKit config
lk_configured = bool(config.LIVEKIT_URL and config.LIVEKIT_API_KEY)

if not lk_configured:
    st.warning(
        "LiveKit is not configured. Add `LIVEKIT_URL`, `LIVEKIT_API_KEY`, "
        "and `LIVEKIT_API_SECRET` to your `.env` file.\n\n"
        "Get free credentials at: https://cloud.livekit.io"
    )

st.subheader("How Live Mode Works")
st.markdown("""
1. **Create a room** — generates a unique interview session
2. **Two participants join** — one as Clinician, one as Patient
3. **LiveKit separates audio tracks** — each speaker has their own stream
4. **Whisper transcribes each track** — no diarization needed (perfect attribution)
5. **Segments are indexed in real-time** — searchable during and after the interview
""")

st.divider()

# ── Room Creation ──
st.subheader("Create Interview Room")

col1, col2 = st.columns(2)
with col1:
    room_name = st.text_input("Room Name", value="clinical-interview-001", placeholder="Enter a room name")
with col2:
    st.write("")  # Spacing
    st.write("")
    create_btn = st.button("Create Room", type="primary", disabled=not lk_configured)

if create_btn:
    if lk_configured:
        st.info("LiveKit integration will be implemented in Phase 4. "
                "The room creation, token generation, and audio track processing "
                "pipeline will be wired here.")
        # TODO Phase 4:
        # from audio.livekit_handler import LiveKitHandler
        # handler = LiveKitHandler()
        # room = handler.create_room(room_name)
        # clinician_token = handler.generate_token(room_name, "Clinician", "CLINICIAN")
        # patient_token = handler.generate_token(room_name, "Patient", "PATIENT")
        # st.write(f"Clinician join link: ...")
        # st.write(f"Patient join link: ...")

st.divider()

# ── Placeholder for live transcript ──
st.subheader("Live Transcript")
st.caption("Segments will appear here in real-time as participants speak.")

# Placeholder display
st.markdown("""
<div style="background:#f7fafc;border:1px dashed #cbd5e0;border-radius:8px;padding:2rem;text-align:center;color:#a0aec0;">
    <p style="font-size:1.2rem;">No active interview session</p>
    <p>Create a room and have participants join to start.</p>
</div>
""", unsafe_allow_html=True)
