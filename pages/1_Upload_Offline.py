"""
Page 1: Upload & Offline Pipeline

Uses AssemblyAI for acoustic speaker diarization (who spoke when based on voice)
then Groq LLM for semantic role labeling (which speaker is PATIENT vs CLINICIAN).

This hybrid approach combines:
- Acoustic accuracy: voice embeddings reliably separate distinct speakers
- Semantic intelligence: LLM identifies clinical roles from conversation content
"""

import streamlit as st
import config
import tempfile
import os
import uuid
import json
import math

st.set_page_config(page_title="Upload Interview", page_icon="UP", layout="wide")
st.title("Upload & Process Clinical Interview")
st.markdown(
    f'<div style="background:linear-gradient(135deg,#fff3cd,#ffeeba);'
    f'border:1px solid #ffc107;border-radius:8px;padding:0.8rem;'
    f'color:#856404;font-size:0.9rem;">{config.ETHICS_DISCLAIMER}</div>',
    unsafe_allow_html=True,
)

# Check AssemblyAI config
if not config.ASSEMBLYAI_API_KEY:
    st.warning(
        "AssemblyAI API key not set. Add `ASSEMBLYAI_API_KEY` to your Streamlit secrets. "
        "Get a free key at https://www.assemblyai.com (100 hours free)."
    )

# ══════════════════════════════════════════
# Session State
# ══════════════════════════════════════════
for key in ["interview_id", "aligned_segments", "speakers_detected"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "speakers_detected" else []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False


# ══════════════════════════════════════════
# Core: AssemblyAI transcription + diarization
# ══════════════════════════════════════════
def transcribe_with_diarization(audio_path):
    """
    Use AssemblyAI to transcribe audio WITH acoustic speaker diarization.
    Returns list of utterances with speaker labels and timestamps.
    No GPU needed — this is a cloud API call.
    """
    import assemblyai as aai

    aai.settings.api_key = config.ASSEMBLYAI_API_KEY

    aai_config = aai.TranscriptionConfig(
        speaker_labels=True,
        language_code="en",
    )

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=aai_config)

    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(f"AssemblyAI error: {transcript.error}")

    utterances = []
    for utt in transcript.utterances:
        utterances.append({
            "start_ms": utt.start,
            "end_ms": utt.end,
            "text": utt.text.strip(),
            "speaker": utt.speaker,  # "A", "B", "C", etc.
        })

    return utterances


# ══════════════════════════════════════════
# Core: LLM role labeling on acoustically-separated speakers
# ══════════════════════════════════════════
def label_roles_with_llm(utterances):
    """
    Takes acoustically-separated utterances (Speaker A, B, C from AssemblyAI)
    and uses Groq LLM to identify roles (PATIENT/CLINICIAN) and names.

    This is MUCH more accurate than pure-LLM diarization because the acoustic
    model already correctly separated the voices — the LLM only needs to
    figure out which voice is the doctor vs the patient.
    """
    import httpx

    # Build a summary of what each speaker said
    speaker_samples = {}
    for utt in utterances:
        spk = utt["speaker"]
        if spk not in speaker_samples:
            speaker_samples[spk] = []
        if len(speaker_samples[spk]) < 8:
            speaker_samples[spk].append(utt["text"])

    # Build speaker summaries
    speaker_summary_lines = []
    for spk in sorted(speaker_samples.keys()):
        samples = speaker_samples[spk]
        count = sum(1 for u in utterances if u["speaker"] == spk)
        sample_text = " | ".join(samples)
        speaker_summary_lines.append(f"Speaker {spk} ({count} utterances): {sample_text}")
    speaker_summary = "\n".join(speaker_summary_lines)

    prompt = f"""You are analyzing a clinical/therapy transcript where speakers have been acoustically separated by voice analysis. Each speaker label (A, B, C, etc.) represents a DISTINCT voice — the acoustic model has already determined who is who based on their voice.

Your task: For each acoustic speaker, determine their ROLE and NAME.

SPEAKERS DETECTED (with sample utterances):
{speaker_summary}

RULES:
1. Each acoustic speaker (A, B, C) is a DISTINCT person — trust the voice separation
2. CLINICIAN: asks therapeutic/diagnostic questions, gives professional guidance, directs the conversation
3. PATIENT: describes personal experiences, expresses emotions, answers questions about themselves
4. A spouse pushing their partner to engage in therapy is a PATIENT, not a clinician
5. If a name is mentioned or clearly implied, include it. "Dr. Dan", "Bob", etc.
6. NAME ALIASING: If the same acoustic speaker is called different names, they are the same person (e.g., "Bob" and "Dr. Dan" for Speaker A means their full name is Dr. Bob Dan)
7. If you cannot determine a name, use "Unknown"

Respond with ONLY a JSON array, one object per acoustic speaker:
[{{"speaker": "A", "role": "CLINICIAN", "name": "Dr. Dan", "reasoning": "Asks therapeutic questions, directs sessions"}}, {{"speaker": "B", "role": "PATIENT", "name": "Calvin", "reasoning": "Expresses reluctance, describes personal experiences"}}]

No other text. Just the JSON array."""

    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.1,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    cast = json.loads(content)

    # Build lookup: acoustic speaker → display label + role
    speaker_map = {}
    for entry in cast:
        spk = entry["speaker"]
        role = entry["role"].upper()
        name = entry.get("name", "Unknown")
        if "CLINICIAN" in role or "THERAPIST" in role or "DOCTOR" in role:
            role_base = "CLINICIAN"
        else:
            role_base = "PATIENT"

        # Count how many of this role we've seen
        role_count = sum(1 for v in speaker_map.values() if v["role_base"] == role_base) + 1
        role_numbered = f"{role_base}_{role_count}"

        if name and name != "Unknown":
            display = f"{name} ({role_numbered})"
        else:
            display = role_numbered

        speaker_map[spk] = {
            "display": display,
            "role_base": role_base,
        }

    return speaker_map


# ══════════════════════════════════════════
# Step 1: Patient Profile
# ══════════════════════════════════════════
st.subheader("Step 1: Patient Profile (Optional)")
with st.expander("Enter Patient Profile", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Patient Name", placeholder="Jane Doe")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
    with col2:
        chief_complaint = st.text_input("Chief Complaint", placeholder="Persistent headaches")
        input_method = st.selectbox("Input Method", ["text", "voice"])
    medical_history = st.text_area("Medical History", placeholder="Relevant history...", height=100)

# ══════════════════════════════════════════
# Step 2: Upload Audio
# ══════════════════════════════════════════
st.subheader("Step 2: Upload Interview Audio")
uploaded_file = st.file_uploader(
    "Upload clinical interview recording",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    help=f"Max {config.MAX_AUDIO_SIZE_MB} MB. AssemblyAI handles files of any size.",
)

if uploaded_file:
    st.audio(uploaded_file)
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.caption(f"{uploaded_file.name} — {file_size_mb:.1f} MB")
    if file_size_mb > config.MAX_AUDIO_SIZE_MB:
        st.error(f"File exceeds {config.MAX_AUDIO_SIZE_MB} MB limit.")

# ══════════════════════════════════════════
# Step 3: Process
# ══════════════════════════════════════════
st.subheader("Step 3: Process Interview")

if uploaded_file and st.button("Run Offline Pipeline", type="primary", use_container_width=True):
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > config.MAX_AUDIO_SIZE_MB:
        st.error(f"File exceeds {config.MAX_AUDIO_SIZE_MB} MB limit.")
        st.stop()

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        from database.models import Interview, PatientProfile, Segment
        from database.supabase_client import SupabaseClient

        db = SupabaseClient()
        interview_id = str(uuid.uuid4())[:8]

        with st.status("Processing interview...", expanded=True) as status:
            # ── Create interview record ──
            st.write("Creating interview record...")
            interview = Interview(
                interview_id=interview_id,
                title=f"Interview — {uploaded_file.name}",
                source_mode="offline",
                audio_filename=uploaded_file.name,
            )
            db.create_interview(interview)
            st.session_state.interview_id = interview_id

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

            # ── Step A: AssemblyAI transcription + acoustic diarization ──
            st.write("Transcribing and diarizing with AssemblyAI (acoustic speaker separation)...")
            try:
                utterances = transcribe_with_diarization(tmp_path)
                unique_speakers = sorted(set(u["speaker"] for u in utterances))
                st.write(
                    f"Transcription complete — {len(utterances)} utterances, "
                    f"{len(unique_speakers)} speakers detected acoustically."
                )
            except Exception as e:
                st.error(f"AssemblyAI failed: {e}")
                status.update(label="Pipeline failed at transcription", state="error")
                st.stop()

            # ── Step B: LLM role labeling ──
            st.write("Identifying speaker roles and names with LLM analysis...")
            try:
                speaker_map = label_roles_with_llm(utterances)
                for spk, info in speaker_map.items():
                    st.write(f"  Speaker {spk} → **{info['display']}**")
            except Exception as e:
                st.warning(f"LLM labeling failed: {e}. Using acoustic labels only.")
                speaker_map = {}
                for i, spk in enumerate(unique_speakers):
                    speaker_map[spk] = {
                        "display": f"SPEAKER_{spk}",
                        "role_base": "PATIENT" if i > 0 else "CLINICIAN",
                    }

            # ── Step C: Build and save segments ──
            st.write("Saving segments to database...")
            segments_to_insert = []
            for utt in utterances:
                spk = utt["speaker"]
                info = speaker_map.get(spk, {"display": f"SPEAKER_{spk}", "role_base": "PATIENT"})
                segments_to_insert.append(Segment(
                    interview_id=interview_id,
                    segment_id=str(uuid.uuid4())[:8],
                    start_ms=utt["start_ms"],
                    end_ms=utt["end_ms"],
                    speaker_raw=info["display"],
                    speaker_role=info["role_base"],
                    text=utt["text"],
                    source_mode="offline",
                ))

            count = db.insert_segments(segments_to_insert)
            role_map = {info["display"]: info["role_base"] for info in speaker_map.values()}
            db.update_interview_speaker_map(interview_id, role_map)
            st.session_state.processing_complete = True
            status.update(label=f"Complete — {count} segments saved!", state="complete")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ══════════════════════════════════════════
# Step 4: View Results
# ══════════════════════════════════════════
if st.session_state.processing_complete:
    st.subheader("Interview Transcript")
    from database.supabase_client import SupabaseClient

    db = SupabaseClient()
    segments = db.get_segments(st.session_state.interview_id)

    if segments:
        for seg in segments:
            role = seg.get("speaker_role", "UNKNOWN")
            raw = seg.get("speaker_raw", role)
            text = seg.get("text", "")
            start_ms = seg.get("start_ms", 0)
            end_ms = seg.get("end_ms", 0)
            time_str = (
                f"{start_ms // 60000}:{(start_ms % 60000) // 1000:02d} - "
                f"{end_ms // 60000}:{(end_ms % 60000) // 1000:02d}"
            )

            if role == "PATIENT":
                color, border = "#f0fff4", "#38a169"
            else:
                color, border = "#ebf8ff", "#3182ce"

            st.markdown(
                f'<div style="background-color:{color};border-left:4px solid {border};'
                f'padding:0.5rem 1rem;margin:0.3rem 0;border-radius:0 4px 4px 0;">'
                f'<strong>{raw}</strong> <small>({time_str})</small><br>{text}</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        p_count = sum(1 for s in segments if s["speaker_role"] == "PATIENT")
        c_count = sum(1 for s in segments if s["speaker_role"] == "CLINICIAN")
        unique_raw = sorted(set(s.get("speaker_raw", "") for s in segments))
        st.success(
            f"Total: {len(segments)} segments — Patient: {p_count} / Clinician: {c_count} — "
            f"Speakers: {', '.join(unique_raw)}"
        )
        st.info("Go to **Query Analysis** to search and analyze this interview.")
