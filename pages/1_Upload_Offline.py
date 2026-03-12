"""
Page 1: Upload & Offline Pipeline

Upload clinical interview audio -> Diarize -> Transcribe -> Align -> Store

This page handles the full offline processing pipeline and speaker role mapping.
"""

import streamlit as st
import config
import tempfile
import os

st.set_page_config(page_title="Upload Interview", page_icon="UP", layout="wide")

st.title("Upload & Process Clinical Interview")
st.markdown(f'<div class="ethics-banner" style="background:linear-gradient(135deg,#fff3cd,#ffeeba);border:1px solid #ffc107;border-radius:8px;padding:0.8rem;color:#856404;font-size:0.9rem;">{config.ETHICS_DISCLAIMER}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# Session State Initialization
# ══════════════════════════════════════════
if "interview_id" not in st.session_state:
    st.session_state.interview_id = None
if "diarization_segments" not in st.session_state:
    st.session_state.diarization_segments = None
if "transcription_segments" not in st.session_state:
    st.session_state.transcription_segments = None
if "aligned_segments" not in st.session_state:
    st.session_state.aligned_segments = None
if "speakers_detected" not in st.session_state:
    st.session_state.speakers_detected = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# ══════════════════════════════════════════
# Step 1: Patient Profile (optional)
# ══════════════════════════════════════════
st.subheader("Step 1: Patient Profile (Optional)")
st.caption("Provide context about the patient before processing the interview.")

with st.expander("Enter Patient Profile", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name", placeholder="Jane Doe")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
    with col2:
        chief_complaint = st.text_input("Chief Complaint", placeholder="Persistent headaches for 2 weeks")
        input_method = st.selectbox("Input Method", ["text", "voice"])

    medical_history = st.text_area(
        "Medical History",
        placeholder="Relevant medical history, medications, allergies...",
        height=100,
    )

    # Voice input option
    if input_method == "voice":
        st.info("Voice input: Upload a short audio recording of the patient profile narration.")
        voice_file = st.file_uploader("Upload voice profile", type=["wav", "mp3", "m4a"], key="voice_profile")
        if voice_file:
            st.audio(voice_file)
            st.caption("Voice will be transcribed via Whisper when you click 'Process Interview'.")

# ══════════════════════════════════════════
# Step 2: Upload Audio
# ══════════════════════════════════════════
st.subheader("Step 2: Upload Interview Audio")

uploaded_file = st.file_uploader(
    "Upload clinical interview recording",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    help=f"Max {config.MAX_AUDIO_SIZE_MB} MB. Supported: {', '.join(config.SUPPORTED_AUDIO_FORMATS)}",
)

if uploaded_file:
    st.audio(uploaded_file)
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.caption(f"{uploaded_file.name} - {file_size_mb:.1f} MB")

    if file_size_mb > config.MAX_AUDIO_SIZE_MB:
        st.error(f"File exceeds {config.MAX_AUDIO_SIZE_MB} MB limit. Please upload a shorter recording.")

# ══════════════════════════════════════════
# Step 3: Process Pipeline
# ══════════════════════════════════════════
st.subheader("Step 3: Process Interview")

if uploaded_file and st.button("Run Offline Pipeline", type="primary", use_container_width=True):
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        # ── Step 3a: Create Interview Record ──
        with st.status("Processing interview...", expanded=True) as status:
            from database.models import Interview, PatientProfile, Segment
            from database.supabase_client import SupabaseClient
            import uuid

            db = SupabaseClient()
            interview_id = str(uuid.uuid4())[:8]

            st.write("Creating interview record...")
            interview = Interview(
                interview_id=interview_id,
                title=f"Interview - {uploaded_file.name}",
                source_mode="offline",
                audio_filename=uploaded_file.name,
            )
            db.create_interview(interview)
            st.session_state.interview_id = interview_id

            # Save patient profile if provided
            if patient_name or chief_complaint:
                profile = PatientProfile(
                    interview_id=interview_id,
                    name=patient_name,
                    age=patient_age if patient_age > 0 else None,
                    chief_complaint=chief_complaint,
                    medical_history=medical_history,
                    input_method=input_method,
                )
                db.create_profile(profile)
                st.write("Patient profile saved.")

            # ── Step 3b: Diarization ──
            st.write("Running speaker diarization (this may take a moment)...")
            try:
                from audio.diarize import SpeakerDiarizer
                diarizer = SpeakerDiarizer()
                diar_segments = diarizer.diarize(tmp_path)
                st.session_state.diarization_segments = diar_segments
                speakers = diarizer.get_unique_speakers(diar_segments)
                st.session_state.speakers_detected = speakers
                st.write(f"Diarization complete - {len(diar_segments)} segments, {len(speakers)} speakers detected.")
            except Exception as e:
                st.error(f"Diarization failed: {e}")
                st.info("Diarization requires a GPU. Try running in Google Colab with GPU runtime.")
                status.update(label="Pipeline failed at diarization", state="error")
                st.stop()

            # ── Step 3c: Transcription ──
            st.write("Transcribing audio via Groq Whisper API...")
            try:
                from audio.transcribe import WhisperTranscriber
                transcriber = WhisperTranscriber()
                trans_segments = transcriber.transcribe(tmp_path)
                st.session_state.transcription_segments = trans_segments
                st.write(f"Transcription complete - {len(trans_segments)} segments.")
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                status.update(label="Pipeline failed at transcription", state="error")
                st.stop()

            # ── Step 3d: Alignment ──
            st.write("Aligning transcript with speaker labels...")
            from audio.align import align_segments
            aligned = align_segments(diar_segments, trans_segments, interview_id)
            st.session_state.aligned_segments = aligned
            st.write(f"Alignment complete - {len(aligned)} speaker-labeled segments.")

            status.update(label="Pipeline complete! Proceed to speaker mapping.", state="complete")

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ══════════════════════════════════════════
# Step 4: Speaker Role Mapping
# ══════════════════════════════════════════
if st.session_state.aligned_segments and not st.session_state.processing_complete:
    st.subheader("Step 4: Map Speakers to Roles")
    st.caption("Assign each detected speaker to a role (PATIENT or CLINICIAN).")

    speakers = st.session_state.speakers_detected
    speaker_map = {}

    cols = st.columns(len(speakers) if speakers else 1)
    for i, speaker in enumerate(speakers):
        with cols[i]:
            # Show a sample of what this speaker said
            sample_texts = [
                s.text for s in st.session_state.aligned_segments
                if s.speaker_raw == speaker
            ][:3]
            st.markdown(f"**{speaker}**")
            for t in sample_texts:
                st.caption(f'"{t[:100]}..."' if len(t) > 100 else f'"{t}"')

            role = st.selectbox(
                f"Role for {speaker}",
                config.SPEAKER_ROLES,
                key=f"role_{speaker}",
            )
            speaker_map[speaker] = role

    if st.button("Confirm Roles & Save Segments", type="primary", use_container_width=True):
        from database.supabase_client import SupabaseClient
        from audio.align import apply_speaker_map

        db = SupabaseClient()

        # Apply mapping
        segments = apply_speaker_map(st.session_state.aligned_segments, speaker_map)

        # Save to database
        count = db.insert_segments(segments)
        db.update_interview_speaker_map(st.session_state.interview_id, speaker_map)

        st.session_state.processing_complete = True
        st.success(f"{count} segments saved to database with speaker roles assigned!")

# ══════════════════════════════════════════
# Step 5: View Results
# ══════════════════════════════════════════
if st.session_state.processing_complete:
    st.subheader("Step 5: Interview Transcript")

    from database.supabase_client import SupabaseClient
    db = SupabaseClient()
    segments = db.get_segments(st.session_state.interview_id)

    if segments:
        # Color-coded transcript display
        for seg in segments:
            role = seg.get("speaker_role", "UNKNOWN")
            text = seg.get("text", "")
            start_ms = seg.get("start_ms", 0)
            end_ms = seg.get("end_ms", 0)
            time_str = f"{start_ms // 60000}:{(start_ms % 60000) // 1000:02d} - {end_ms // 60000}:{(end_ms % 60000) // 1000:02d}"

            if role == "PATIENT":
                st.markdown(
                    f'<div class="speaker-patient" style="background-color:#f0fff4;border-left:4px solid #38a169;padding:0.5rem 1rem;margin:0.3rem 0;border-radius:0 4px 4px 0;">'
                    f'<strong>PATIENT</strong> <small>({time_str})</small><br>{text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="speaker-clinician" style="background-color:#ebf8ff;border-left:4px solid #3182ce;padding:0.5rem 1rem;margin:0.3rem 0;border-radius:0 4px 4px 0;">'
                    f'<strong>CLINICIAN</strong> <small>({time_str})</small><br>{text}</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.success(f"Total segments: {len(segments)} - "
                   f"Patient: {sum(1 for s in segments if s['speaker_role']=='PATIENT')} / "
                   f"Clinician: {sum(1 for s in segments if s['speaker_role']=='CLINICIAN')}")

        st.info("Go to the **Query & Analysis** page to search and analyze this interview.")
