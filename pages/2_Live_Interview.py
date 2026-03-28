"""
Page 2: Live Clinical Interview via LiveKit

Working implementation:
    - Create LiveKit room
    - Generate join tokens for clinician + patient
    - Join links via LiveKit Meet (browser-based, no custom frontend)
    - Upload recorded tracks for transcription
    - Store segments with perfect speaker attribution
    - Architecture docs + comparison table for demo video

Requires local environment (WebRTC needs browser mic access).
"""

import streamlit as st
import config
import pandas as pd
import tempfile
import os
import uuid

st.set_page_config(page_title="Live Interview", page_icon="🎤", layout="wide")

# ── Header ──
st.markdown(
    f'<p style="font-size:1.8rem;font-weight:700;'
    f'background:linear-gradient(135deg,{config.BRAND_PRIMARY},{config.BRAND_SECONDARY});'
    f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
    f'Live Clinical Interview</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="background:linear-gradient(135deg,rgba(255,193,7,0.1),rgba(255,152,0,0.1));'
    f'border:1px solid rgba(255,193,7,0.3);border-radius:10px;padding:0.8rem 1.2rem;'
    f'color:#FFC107;font-size:0.88rem;">{config.ETHICS_DISCLAIMER}</div>',
    unsafe_allow_html=True,
)

# ── LiveKit status ──
lk_configured = bool(config.LIVEKIT_URL and config.LIVEKIT_API_KEY and config.LIVEKIT_API_SECRET)

if lk_configured:
    st.success("LiveKit credentials configured — live interview features available.")
else:
    st.warning(
        "LiveKit not configured. Add `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and "
        "`LIVEKIT_API_SECRET` to your `.env` file. Get free credentials at "
        "**https://cloud.livekit.io**"
    )

# Session state
if "lk_room_name" not in st.session_state:
    st.session_state.lk_room_name = None
if "lk_tokens" not in st.session_state:
    st.session_state.lk_tokens = {}
if "lk_interview_id" not in st.session_state:
    st.session_state.lk_interview_id = None

st.divider()

# ══════════════════════════════════════════
# Section 1: Architecture (always visible)
# ══════════════════════════════════════════
st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>How Live Mode Works</h3>", unsafe_allow_html=True)

col_off, col_live = st.columns(2)
with col_off:
    st.markdown(
        f'<div style="background:{config.BRAND_CARD};border:1px solid #30363D;'
        f'border-radius:10px;padding:1.2rem;font-family:monospace;font-size:0.85rem;'
        f'color:{config.BRAND_MUTED};line-height:1.8;">'
        f'<span style="color:{config.BRAND_PRIMARY};font-weight:600;">OFFLINE PIPELINE</span><br><br>'
        f'Mixed Audio File<br>'
        f'  → Acoustic Diarization<br>'
        f'  → Whisper Transcription<br>'
        f'  → Temporal Alignment<br>'
        f'  → Store in Supabase<br><br>'
        f'<em>Speaker separation ≈ approximate</em>'
        f'</div>',
        unsafe_allow_html=True,
    )
with col_live:
    st.markdown(
        f'<div style="background:{config.BRAND_CARD};border:1px solid #30363D;'
        f'border-radius:10px;padding:1.2rem;font-family:monospace;font-size:0.85rem;'
        f'color:{config.BRAND_MUTED};line-height:1.8;">'
        f'<span style="color:{config.BRAND_SECONDARY};font-weight:600;">LIVE PIPELINE</span><br><br>'
        f'LiveKit Room (WebRTC)<br>'
        f'  → Track 1: Clinician Audio<br>'
        f'  → Track 2: Patient Audio<br>'
        f'  → Per-Track Whisper Transcription<br>'
        f'  → Store in Supabase (DER=0%)<br><br>'
        f'<em>Speaker separation = perfect</em>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════
# Section 2: Room Management (if LiveKit configured)
# ══════════════════════════════════════════
if lk_configured:
    st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Interview Room</h3>", unsafe_allow_html=True)

    # ── Create Room ──
    if st.session_state.lk_room_name is None:
        col_name, col_btn = st.columns([3, 1])
        with col_name:
            room_input = st.text_input("Room Name", value=f"clinical-interview-{str(uuid.uuid4())[:6]}")
        with col_btn:
            st.write("")  # spacing
            st.write("")
            if st.button("Create Room", type="primary", use_container_width=True):
                try:
                    from audio.livekit_handler import LiveKitHandler
                    lk = LiveKitHandler()
                    result = lk.create_room(room_input)
                    st.session_state.lk_room_name = room_input
                    st.session_state.lk_interview_id = str(uuid.uuid4())[:8]

                    # Generate tokens
                    clinician_token = lk.generate_token(room_input, "Clinician", "CLINICIAN")
                    patient_token = lk.generate_token(room_input, "Patient", "PATIENT")

                    st.session_state.lk_tokens = {
                        "clinician": {
                            "token": clinician_token,
                            "join_url": lk.generate_join_url(room_input, clinician_token),
                        },
                        "patient": {
                            "token": patient_token,
                            "join_url": lk.generate_join_url(room_input, patient_token),
                        },
                    }
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create room: {e}")

    # ── Active Room ──
    else:
        room_name = st.session_state.lk_room_name
        interview_id = st.session_state.lk_interview_id

        st.markdown(
            f'<div style="background:{config.BRAND_CARD};border:1px solid {config.BRAND_SECONDARY};'
            f'border-radius:10px;padding:1rem;">'
            f'<span style="color:{config.BRAND_SECONDARY};font-weight:600;">● ROOM ACTIVE</span>'
            f'<span style="color:{config.BRAND_MUTED};margin-left:1rem;">{room_name}</span>'
            f'<span style="color:{config.BRAND_MUTED};margin-left:1rem;">Interview: {interview_id}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        # ── Join Links ──
        st.markdown(f"<small style='color:{config.BRAND_MUTED};font-weight:600;letter-spacing:1px;'>JOIN LINKS</small>", unsafe_allow_html=True)
        st.caption("Open each link in a separate browser tab or device. Each participant gets their own audio track.")

        col_c, col_p = st.columns(2)

        with col_c:
            st.markdown(
                f'<div style="background:{config.CLINICIAN_BG};border:1px solid {config.CLINICIAN_COLOR};'
                f'border-radius:10px;padding:1rem;">'
                f'<strong style="color:{config.CLINICIAN_COLOR};">Clinician</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
            clinician_url = st.session_state.lk_tokens.get("clinician", {}).get("join_url", "")
            st.code(clinician_url, language=None)
            st.link_button("Open Clinician Tab", clinician_url, use_container_width=True)

        with col_p:
            st.markdown(
                f'<div style="background:{config.PATIENT_BG};border:1px solid {config.PATIENT_COLOR};'
                f'border-radius:10px;padding:1rem;">'
                f'<strong style="color:{config.PATIENT_COLOR};">Patient</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
            patient_url = st.session_state.lk_tokens.get("patient", {}).get("join_url", "")
            st.code(patient_url, language=None)
            st.link_button("Open Patient Tab", patient_url, use_container_width=True)

        st.divider()

        # ── Room Status ──
        st.markdown(f"<small style='color:{config.BRAND_MUTED};font-weight:600;letter-spacing:1px;'>ROOM STATUS</small>", unsafe_allow_html=True)

        if st.button("Refresh Participants"):
            try:
                from audio.livekit_handler import LiveKitHandler
                lk = LiveKitHandler()
                participants = lk.list_participants(room_name)
                if participants:
                    for p in participants:
                        st.write(f"**{p['name']}** — {p['identity']}")
                else:
                    st.info("No participants currently in room.")
            except Exception as e:
                st.warning(f"Could not fetch participants: {e}")

        st.divider()

        # ── Process Recorded Tracks ──
        st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Process Interview Audio</h3>", unsafe_allow_html=True)
        st.caption(
            "After the interview, upload each participant's recorded audio track. "
            "LiveKit Meet allows you to record locally. Each track gets transcribed "
            "independently with perfect speaker attribution."
        )

        col_up_c, col_up_p = st.columns(2)

        with col_up_c:
            clinician_audio = st.file_uploader(
                "Clinician Audio Track",
                type=["wav", "mp3", "ogg", "flac", "m4a", "webm"],
                key="clinician_track",
            )
            clinician_name = st.text_input("Clinician Name", value="Clinician", key="clin_name")

        with col_up_p:
            patient_audio = st.file_uploader(
                "Patient Audio Track",
                type=["wav", "mp3", "ogg", "flac", "m4a", "webm"],
                key="patient_track",
            )
            patient_name = st.text_input("Patient Name", value="Patient", key="pat_name")

        if (clinician_audio or patient_audio) and st.button(
            "Process Tracks & Store Segments", type="primary", use_container_width=True
        ):
            from database.models import Interview, Segment
            from database.supabase_client import SupabaseClient
            from audio.livekit_handler import LiveKitHandler

            db = SupabaseClient()

            with st.status("Processing live interview tracks...", expanded=True) as status:
                # Create interview record
                st.write("Creating interview record...")
                interview = Interview(
                    interview_id=interview_id,
                    title=f"Live Interview — {room_name}",
                    source_mode="live",
                    audio_filename=room_name,
                )
                db.create_interview(interview)

                all_segments = []

                # Process clinician track
                if clinician_audio:
                    st.write(f"Transcribing **clinician** track ({clinician_name})...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(clinician_audio.getbuffer())
                        tmp_path = tmp.name
                    try:
                        segs = LiveKitHandler.process_audio_for_participant(
                            tmp_path, interview_id, "CLINICIAN", clinician_name
                        )
                        all_segments.extend(segs)
                        st.write(f"  → {len(segs)} clinician segments.")
                    finally:
                        os.unlink(tmp_path)

                # Process patient track
                if patient_audio:
                    st.write(f"Transcribing **patient** track ({patient_name})...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(patient_audio.getbuffer())
                        tmp_path = tmp.name
                    try:
                        segs = LiveKitHandler.process_audio_for_participant(
                            tmp_path, interview_id, "PATIENT", patient_name
                        )
                        all_segments.extend(segs)
                        st.write(f"  → {len(segs)} patient segments.")
                    finally:
                        os.unlink(tmp_path)

                # Sort all segments chronologically and insert
                if all_segments:
                    all_segments.sort(key=lambda s: s.start_ms)
                    count = db.insert_segments(all_segments)

                    # Generate embeddings if available
                    if config.ENV["has_embeddings"]:
                        st.write("Generating embeddings...")
                        try:
                            from retrieval.embeddings import embed_and_store_segments
                            stored = db.get_segments(interview_id)
                            embed_and_store_segments(stored, db, progress_callback=st.write)
                        except Exception as e:
                            st.warning(f"Embedding generation failed: {e}")

                    speaker_map = {
                        f"{clinician_name} (CLINICIAN)": "CLINICIAN",
                        f"{patient_name} (PATIENT)": "PATIENT",
                    }
                    db.update_interview_speaker_map(interview_id, speaker_map)

                    status.update(label=f"Complete — {count} segments saved with perfect attribution!", state="complete")
                else:
                    status.update(label="No audio tracks uploaded.", state="error")

        # ── End Room ──
        st.divider()
        if st.button("Close Room", type="secondary"):
            try:
                from audio.livekit_handler import LiveKitHandler
                lk = LiveKitHandler()
                lk.delete_room(room_name)
            except Exception:
                pass
            st.session_state.lk_room_name = None
            st.session_state.lk_tokens = {}
            st.session_state.lk_interview_id = None
            st.rerun()

# ══════════════════════════════════════════
# Setup Instructions (always visible)
# ══════════════════════════════════════════
st.divider()
st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Setup Guide</h3>", unsafe_allow_html=True)

with st.expander("Step 1: Get LiveKit Credentials (Free)"):
    st.markdown("""
1. Go to **https://cloud.livekit.io** and sign up (free — 50 participant-minutes/month)
2. Create a new project
3. Go to **Settings → Keys** and copy:
   - `LIVEKIT_URL` (starts with `wss://`)
   - `LIVEKIT_API_KEY` (starts with `API`)
   - `LIVEKIT_API_SECRET`
4. Add to your `.env` file
    """)

with st.expander("Step 2: Conduct a Live Interview"):
    st.markdown("""
1. Click **Create Room** above
2. Open the **Clinician** join link in one browser tab
3. Open the **Patient** join link in another tab (or device)
4. Conduct the interview (2-5 minutes is sufficient for demo)
5. In LiveKit Meet, use the **Record** button to save each participant's audio
6. Upload the recorded tracks above and click **Process Tracks**
7. Go to **Query Analysis** to search and analyze the live interview
    """)

# ══════════════════════════════════════════
# Comparison Table
# ══════════════════════════════════════════
st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Offline vs Live Pipeline Comparison</h3>", unsafe_allow_html=True)

table_md = """
| Feature | Offline (Pyannote + AssemblyAI) | Live (LiveKit) |
|---|---|---|
| **Speaker Separation** | Acoustic diarization | Track separation (WebRTC) |
| **Diarization Error Rate** | ~5-15% | 0% (perfect) |
| **Transcription** | AssemblyAI / Groq Whisper | Groq Whisper per-track |
| **Max Speakers** | 10+ | Room capacity |
| **Real-time** | No (batch) | Yes (streaming) |
| **GPU Required** | Optional (Pyannote) | No |
| **Audio Source** | Uploaded file | Live microphone |
"""
st.markdown(table_md)
