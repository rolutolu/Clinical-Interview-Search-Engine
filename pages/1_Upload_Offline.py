"""
Page 1: Upload & Offline Pipeline

TWO MODES:
  - Cloud mode (no GPU): Groq Whisper transcription (with auto-chunking
    for files over 25 MB) + Groq LLM speaker labeling
  - Full mode (GPU): Pyannote diarization + Whisper + alignment

Both are fully automatic — no manual speaker assignment needed.
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

# ── Detect environment ──
_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    pass

if _HAS_TORCH:
    st.success("GPU detected — full pipeline (pyannote diarization + transcription).")
else:
    st.info(
        "Running in **cloud mode**: Groq Whisper transcription + LLM-based speaker labeling. "
        "Large files are automatically split into chunks."
    )

# Session State
for key in [
    "interview_id", "transcription_segments", "aligned_segments",
    "diarization_segments", "speakers_detected",
]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "speakers_detected" else []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

GROQ_CHUNK_LIMIT_MB = 24  # Stay under Groq's 25 MB limit


# Helper: Chunked Whisper transcription
def transcribe_audio(audio_path, filename, status_writer):
    """
    Transcribe audio via Groq Whisper API.
    Automatically splits files over 24 MB into chunks using ffmpeg.
    """
    import subprocess

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    if file_size_mb <= GROQ_CHUNK_LIMIT_MB:
        status_writer(f"Transcribing ({file_size_mb:.1f} MB)...")
        return _whisper_api_call(audio_path, filename)
    else:
        # Get duration using ffprobe
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True
        )
        total_duration = float(result.stdout.strip())

        # Calculate how many chunks we need
        num_chunks = math.ceil(file_size_mb / GROQ_CHUNK_LIMIT_MB)
        chunk_duration = total_duration / num_chunks

        status_writer(f"File is {file_size_mb:.1f} MB — splitting into {num_chunks} chunks...")

        all_segments = []
        for i in range(num_chunks):
            start_sec = i * chunk_duration
            chunk_path = os.path.join(tempfile.gettempdir(), f"chunk_{i:02d}.wav")

            # Split with ffmpeg
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start_sec),
                 "-t", str(chunk_duration), "-ar", "16000", "-ac", "1", chunk_path],
                capture_output=True
            )

            chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            status_writer(f"  Chunk {i+1}/{num_chunks} ({chunk_size_mb:.1f} MB)...")

            try:
                chunk_segments = _whisper_api_call(chunk_path, f"chunk_{i:02d}.wav")
                offset_ms = int(start_sec * 1000)
                for seg in chunk_segments:
                    seg["start_ms"] += offset_ms
                    seg["end_ms"] += offset_ms
                all_segments.extend(chunk_segments)
            finally:
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)

        status_writer(f"Transcription complete — {len(all_segments)} segments from {num_chunks} chunks.")
        return all_segments


def _whisper_api_call(audio_path, filename):
    """Single Groq Whisper API call."""
    import httpx

    with open(audio_path, "rb") as f:
        response = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            files={"file": (filename, f)},
            data={
                "model": config.WHISPER_MODEL,
                "language": "en",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            },
            timeout=120.0,
        )
    response.raise_for_status()
    result = response.json()

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start_ms": int(seg["start"] * 1000),
            "end_ms": int(seg["end"] * 1000),
            "text": seg["text"].strip(),
        })
    return segments

# Helper: LLM-based speaker labeling
def label_speakers_with_llm(segments):
    """
    Use Groq LLM to label each segment with numbered speaker roles
    and identify names when possible.
    speaker_raw = display label (e.g. "Dr. Den (CLINICIAN_1)")
    speaker_role = base role for retrieval (PATIENT/CLINICIAN)
    """
    import httpx

    if len(segments) > 100:
        sample_segments = segments[:80] + segments[-20:]
        sample_indices = list(range(80)) + list(range(len(segments) - 20, len(segments)))
    else:
        sample_segments = segments
        sample_indices = list(range(len(segments)))

    transcript_lines = []
    for idx, seg in zip(sample_indices, sample_segments):
        transcript_lines.append(f"[{idx}] {seg['text']}")
    transcript_text = "\n".join(transcript_lines)

    prompt = f"""You are analyzing a clinical/therapy transcript. There may be MULTIPLE clinicians and/or MULTIPLE patients. The audio may contain several separate sessions spliced together.

Your task: For each numbered segment, identify the speaker using NUMBERED roles AND their name if you can determine it.

FORMAT:
- First patient seen → PATIENT_1, second distinct patient → PATIENT_2, etc.
- First clinician seen → CLINICIAN_1, second distinct clinician → CLINICIAN_2, etc.
- If a speaker's name is mentioned or clearly implied in the transcript (e.g. "Hi Calvin", "Dr. Smith", "thanks doc Adams"), include it in the "name" field. Use the name as it appears — first name, last name, or title+name.
- If you cannot confidently determine a name, set "name" to null.

IDENTIFICATION RULES:
- CLINICIAN: asks probing questions, gives professional guidance, uses therapeutic language, makes clinical observations, directs the conversation professionally
- PATIENT: describes personal experiences, expresses emotions, answers questions about themselves, reports symptoms, shows vulnerability or resistance
- A recording may contain MULTIPLE sessions with DIFFERENT therapists — each new professional voice is a new CLINICIAN_N
- If a patient is non-verbal or non-cooperative, the clinician may speak for extended stretches — consecutive segments from the same speaker are normal
- Track speaker identity by voice/style consistency — if the same clinician asks questions across multiple segments, they keep the same number
- When the conversation clearly shifts to a new session (new greeting, different tone, topic reset), consider whether new speakers have appeared

Respond with ONLY a JSON array, one object per segment:
[{{"index": 0, "role": "PATIENT_1", "name": "Calvin"}}, {{"index": 1, "role": "CLINICIAN_1", "name": "Dr. Den"}}, {{"index": 2, "role": "CLINICIAN_2", "name": null}}, ...]

No other text. Just the JSON array.

TRANSCRIPT:
{transcript_text}"""

    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8000,
            "temperature": 0.1,
        },
        timeout=90.0,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    labels = json.loads(content)
    label_map = {item["index"]: item for item in labels}

    # Build a name lookup: role → name (from first confident identification)
    name_lookup = {}
    for item in labels:
        role = item.get("role", "").upper()
        name = item.get("name")
        if name and role and role not in name_lookup:
            name_lookup[role] = name

    for i, seg in enumerate(segments):
        if i in label_map:
            item = label_map[i]
        else:
            nearest = min(label_map.keys(), key=lambda x: abs(x - i))
            item = label_map[nearest]

        role_numbered = item.get("role", "PATIENT_1").upper()
        name = item.get("name") or name_lookup.get(role_numbered)

        # speaker_raw = display label
        if name:
            seg["speaker_raw"] = f"{name} ({role_numbered})"
        else:
            seg["speaker_raw"] = role_numbered

        # speaker_role = base role for retrieval filtering
        if "CLINICIAN" in role_numbered or "THERAPIST" in role_numbered or "DOCTOR" in role_numbered:
            seg["speaker_role"] = "CLINICIAN"
        else:
            seg["speaker_role"] = "PATIENT"

    return segments

# Step 1: Patient Profile
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

# Step 2: Upload Audio
st.subheader("Step 2: Upload Interview Audio")
uploaded_file = st.file_uploader(
    "Upload clinical interview recording",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    help=f"Max {config.MAX_AUDIO_SIZE_MB} MB. Files over 25 MB are automatically chunked.",
)

if uploaded_file:
    st.audio(uploaded_file)
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.caption(f"{uploaded_file.name} — {file_size_mb:.1f} MB")
    if file_size_mb > config.MAX_AUDIO_SIZE_MB:
        st.error(f"File exceeds {config.MAX_AUDIO_SIZE_MB} MB limit.")
    elif file_size_mb > GROQ_CHUNK_LIMIT_MB:
        st.caption(f"File will be split into ~{math.ceil(file_size_mb / GROQ_CHUNK_LIMIT_MB)} chunks for transcription.")

# Step 3: Process
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

            # ── Transcription (with auto-chunking) ──
            try:
                raw_segments = transcribe_audio(
                    tmp_path, uploaded_file.name, st.write
                )

                trans_segments = []
                for seg in raw_segments:
                    trans_segments.append({
                        "segment_id": str(uuid.uuid4())[:8],
                        "interview_id": interview_id,
                        "start_ms": seg["start_ms"],
                        "end_ms": seg["end_ms"],
                        "text": seg["text"],
                        "source_mode": "offline",
                        "keywords": [],
                        "speaker_raw": "",
                        "speaker_role": "",
                    })
                st.write(f"Total: {len(trans_segments)} transcript segments.")
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                status.update(label="Pipeline failed at transcription", state="error")
                st.stop()

            # ── Speaker labeling ──
            diar_succeeded = False

            if _HAS_TORCH:
                st.write("Running pyannote speaker diarization (GPU)...")
                try:
                    import numpy as np
                    np.NaN = np.nan
                    np.NAN = np.nan
                    from pyannote.audio import Pipeline as PyannotePipeline
                    from huggingface_hub import login

                    login(token=config.HF_TOKEN, add_to_git_credential=False)
                    device = torch.device("cuda")
                    diar_pipeline = PyannotePipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    ).to(device)

                    diar_result = diar_pipeline(tmp_path)
                    diar_segments = []
                    for segment, _track, speaker in diar_result.itertracks(yield_label=True):
                        diar_segments.append({
                            "start_ms": int(segment.start * 1000),
                            "end_ms": int(segment.end * 1000),
                            "speaker": speaker,
                        })

                    speaker_durations = {}
                    for d in diar_segments:
                        spk = d["speaker"]
                        speaker_durations[spk] = speaker_durations.get(spk, 0) + (d["end_ms"] - d["start_ms"])
                    default_speaker = max(speaker_durations, key=speaker_durations.get)

                    for t_seg in trans_segments:
                        best_speaker = default_speaker
                        max_overlap = 0
                        for d_seg in diar_segments:
                            overlap = max(0, min(t_seg["end_ms"], d_seg["end_ms"]) - max(t_seg["start_ms"], d_seg["start_ms"]))
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_speaker = d_seg["speaker"]
                        t_seg["speaker_raw"] = best_speaker

                    unique_speakers = sorted(set(s["speaker_raw"] for s in trans_segments))
                    st.write(f"Diarization complete — {len(unique_speakers)} speakers.")
                    diar_succeeded = True
                except Exception as e:
                    st.warning(f"Diarization failed: {e}. Falling back to LLM labeling.")

            if not diar_succeeded:
                st.write("Labeling speakers using Groq LLM analysis...")
                try:
                    trans_segments = label_speakers_with_llm(trans_segments)
                    st.write("LLM speaker labeling complete.")
                except Exception as e:
                    st.warning(f"LLM labeling failed: {e}. All segments labeled as PATIENT.")
                    for seg in trans_segments:
                        seg["speaker_raw"] = "PATIENT"
                        seg["speaker_role"] = "PATIENT"

            st.session_state.aligned_segments = trans_segments

            if diar_succeeded:
                unique_speakers = sorted(set(s["speaker_raw"] for s in trans_segments))
                st.session_state.speakers_detected = unique_speakers
                status.update(label="Diarization complete — assign speaker roles below.", state="complete")
            else:
                speaker_map = {"PATIENT": "PATIENT", "CLINICIAN": "CLINICIAN"}
                segments_to_insert = []
                for seg in trans_segments:
                    segments_to_insert.append(Segment(
                        interview_id=seg["interview_id"],
                        segment_id=seg["segment_id"],
                        start_ms=seg["start_ms"],
                        end_ms=seg["end_ms"],
                        speaker_raw=seg["speaker_raw"],
                        speaker_role=seg["speaker_role"],
                        text=seg["text"],
                        source_mode=seg["source_mode"],
                    ))

                # ── Generate Embeddings ──
                try:
                    from retrieval.embeddings import generate_embeddings_batch
                    st.write("Generating vector embeddings for semantic search...")
                    texts = [seg["text"] for seg in trans_segments]
                    embeddings = generate_embeddings_batch(texts)
                    for seg, emb in zip(segments_to_insert, embeddings):
                        seg.embedding = emb
                except Exception as e:
                    st.warning(f"Embedding generation failed: {e}. Segments will be saved without embeddings (semantic search won't work for this interview).")

                count = db.insert_segments(segments_to_insert)
                db.update_interview_speaker_map(interview_id, speaker_map)
                st.session_state.processing_complete = True
                status.update(label=f"Complete — {count} segments saved!", state="complete")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Step 4: Speaker Role Mapping (pyannote path only)
if (st.session_state.aligned_segments
        and not st.session_state.processing_complete
        and st.session_state.speakers_detected):
    st.subheader("Step 4: Map Speakers to Roles")

    speakers = st.session_state.speakers_detected
    speaker_map = {}
    cols = st.columns(min(len(speakers), 4))
    for i, speaker in enumerate(speakers):
        with cols[i % len(cols)]:
            sample_texts = [
                s["text"] for s in st.session_state.aligned_segments
                if s["speaker_raw"] == speaker
            ][:3]
            st.markdown(f"**{speaker}**")
            for t in sample_texts:
                st.caption(f'"{t[:80]}..."' if len(t) > 80 else f'"{t}"')
            role = st.selectbox(
                f"Role for {speaker}",
                config.SPEAKER_ROLES,
                key=f"role_{speaker}",
            )
            speaker_map[speaker] = role

    if st.button("Confirm Roles & Save Segments", type="primary", use_container_width=True):
        from database.supabase_client import SupabaseClient
        from database.models import Segment

        db = SupabaseClient()
        segments_to_insert = []
        for seg in st.session_state.aligned_segments:
            seg["speaker_role"] = speaker_map.get(seg["speaker_raw"], "PATIENT")
            segments_to_insert.append(Segment(
                interview_id=seg["interview_id"],
                segment_id=seg["segment_id"],
                start_ms=seg["start_ms"],
                end_ms=seg["end_ms"],
                speaker_raw=seg["speaker_raw"],
                speaker_role=seg["speaker_role"],
                text=seg["text"],
                source_mode=seg["source_mode"],
            ))

        # ── Generate Embeddings ──
        try:
            from retrieval.embeddings import generate_embeddings_batch
            st.write("Generating vector embeddings for semantic search...")
            texts = [seg.text for seg in segments_to_insert]
            embeddings = generate_embeddings_batch(texts)
            for seg, emb in zip(segments_to_insert, embeddings):
                seg.embedding = emb
        except Exception as e:
            st.warning(f"Embedding generation failed: {e}. Segments will be saved without embeddings (semantic search won't work for this interview).")

        count = db.insert_segments(segments_to_insert)
        db.update_interview_speaker_map(st.session_state.interview_id, speaker_map)
        st.session_state.processing_complete = True
        st.success(f"{count} segments saved to database!")

# Step 5: View Results
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
