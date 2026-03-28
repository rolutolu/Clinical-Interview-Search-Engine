"""
Page 1: Upload & Offline Pipeline

Full local version with:
    - Voice input for patient profile (Groq Whisper → text)
    - Dual diarization: Pyannote (primary, GPU) + AssemblyAI (fallback, API)
    - Groq Whisper transcription with auto-chunking for large files
    - Temporal alignment with confidence scoring
    - LLM-based speaker role + name identification
    - Auto-embedding generation (sentence-transformers)
    - Temporal speaker timeline visualization
    - Professional dark medical UI

Gracefully degrades on cloud (no torch): AssemblyAI + LLM labeling only.
"""

import streamlit as st
import config
import tempfile
import os
import uuid
import json

st.set_page_config(page_title="Upload Interview", page_icon="📤", layout="wide")

# ── Page header ──
st.markdown(
    f'<p style="font-size:1.8rem;font-weight:700;'
    f'background:linear-gradient(135deg,{config.BRAND_PRIMARY},{config.BRAND_SECONDARY});'
    f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
    f'Upload & Process Interview</p>',
    unsafe_allow_html=True,
)
st.markdown(f'<div class="ethics-banner" style="background:linear-gradient(135deg,rgba(255,193,7,0.1),rgba(255,152,0,0.1));border:1px solid rgba(255,193,7,0.3);border-radius:10px;padding:0.8rem 1.2rem;color:#FFC107;font-size:0.88rem;">{config.ETHICS_DISCLAIMER}</div>', unsafe_allow_html=True)

# ── Environment badge ──
diar_label = config.DIARIZATION_PRIMARY.title()
methods = ", ".join(config.SEARCH_METHODS)
st.caption(f"Diarization: **{diar_label}** · Search: **{methods}** · Embeddings: **{'Active' if config.ENV['has_embeddings'] else 'Off'}**")

# Session State
for key in ["interview_id", "aligned_segments", "speakers_detected", "diar_result"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "speakers_detected" else []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False


# LLM role labeling (reused from cloud version)
def label_roles_with_llm(utterances):
    """
    Takes acoustically-separated utterances (Speaker A/B/C or SPEAKER_00/01)
    and uses Groq LLM to identify roles and names.
    """
    import httpx

    speaker_samples = {}
    for utt in utterances:
        spk = utt["speaker"]
        if spk not in speaker_samples:
            speaker_samples[spk] = []
        if len(speaker_samples[spk]) < 8:
            speaker_samples[spk].append(utt.get("text", ""))

    lines = []
    for spk in sorted(speaker_samples.keys()):
        samples = speaker_samples[spk]
        count = sum(1 for u in utterances if u["speaker"] == spk)
        lines.append(f"Speaker {spk} ({count} utterances): {' | '.join(samples)}")
    speaker_summary = "\n".join(lines)

    prompt = f"""You are analyzing a clinical/therapy transcript where speakers have been acoustically separated. Each speaker label represents a DISTINCT voice.

Your task: For each acoustic speaker, determine their ROLE and NAME.

SPEAKERS DETECTED (with sample utterances):
{speaker_summary}

RULES:
1. Each speaker is a DISTINCT person — trust the voice separation
2. CLINICIAN: asks therapeutic/diagnostic questions, gives professional guidance
3. PATIENT: describes personal experiences, expresses emotions, answers questions
4. A spouse pressing their partner to engage in therapy is a PATIENT, not a clinician
5. If a name is mentioned or implied, include it
6. NAME ALIASING: same speaker called different names = same person
7. If you cannot determine a name, use "Unknown"

Respond with ONLY a JSON array:
[{{"speaker": "A", "role": "CLINICIAN", "name": "Dr. Dan", "reasoning": "Asks therapeutic questions"}}]

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

    speaker_map = {}
    for entry in cast:
        spk = entry["speaker"]
        role = entry["role"].upper()
        name = entry.get("name", "Unknown")
        role_base = "CLINICIAN" if any(r in role for r in ["CLINICIAN", "THERAPIST", "DOCTOR"]) else "PATIENT"
        role_count = sum(1 for v in speaker_map.values() if v["role_base"] == role_base) + 1
        role_numbered = f"{role_base}_{role_count}"
        display = f"{name} ({role_numbered})" if name and name != "Unknown" else role_numbered
        speaker_map[spk] = {"display": display, "role_base": role_base, "reasoning": entry.get("reasoning", "")}

    return speaker_map

# Step 1: Patient Profile (text + voice)
st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Step 1: Patient Profile</h3>", unsafe_allow_html=True)
st.caption("Provide context before processing. Supports text entry or voice input via Whisper.")

with st.expander("Enter Patient Profile (Optional)", expanded=False):
    profile_tab_text, profile_tab_voice = st.tabs(["Text Input", "Voice Input"])

    with profile_tab_text:
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", placeholder="Jane Doe")
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
        with col2:
            chief_complaint = st.text_input("Chief Complaint", placeholder="Persistent headaches")
        medical_history = st.text_area("Medical History", placeholder="Relevant history...", height=100)
        input_method = "text"

    with profile_tab_voice:
        st.markdown(
            "Record a voice note describing the patient's profile. "
            "It will be transcribed via **Groq Whisper** and used as the medical history."
        )
        voice_file = st.file_uploader(
            "Upload voice recording for patient profile",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            key="voice_profile",
        )

        if voice_file:
            st.audio(voice_file)
            if st.button("Transcribe Voice Profile", key="btn_voice"):
                with st.spinner("Transcribing voice profile..."):
                    try:
                        from audio.transcribe import WhisperTranscriber
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=os.path.splitext(voice_file.name)[1]
                        ) as vtmp:
                            vtmp.write(voice_file.getbuffer())
                            vtmp_path = vtmp.name
                        transcriber = WhisperTranscriber()
                        voice_text = transcriber.transcribe_to_text(vtmp_path)
                        os.unlink(vtmp_path)
                        st.success("Voice profile transcribed!")
                        st.text_area("Transcribed Profile (edit if needed)", value=voice_text, key="voice_result", height=120)
                        # Store in session state for later use
                        st.session_state["voice_profile_text"] = voice_text
                    except Exception as e:
                        st.error(f"Voice transcription failed: {e}")

        # If voice was transcribed, use it
        if "voice_profile_text" in st.session_state:
            patient_name = patient_name or ""
            chief_complaint = chief_complaint or ""
            medical_history = st.session_state.get("voice_profile_text", "")
            input_method = "voice"

# Step 2: Upload Audio
st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Step 2: Upload Interview Audio</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload clinical interview recording",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    help=f"Max {config.MAX_AUDIO_SIZE_MB} MB.",
)

if uploaded_file:
    st.audio(uploaded_file)
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.caption(f"{uploaded_file.name} — {file_size_mb:.1f} MB")
    if file_size_mb > config.MAX_AUDIO_SIZE_MB:
        st.error(f"File exceeds {config.MAX_AUDIO_SIZE_MB} MB limit.")

# Step 3: Process
st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Step 3: Process Interview</h3>", unsafe_allow_html=True)

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

            # ── 1. Create interview record ──
            st.write("Creating interview record...")
            interview = Interview(
                interview_id=interview_id,
                title=f"Interview — {uploaded_file.name}",
                source_mode="offline",
                audio_filename=uploaded_file.name,
            )
            db.create_interview(interview)
            st.session_state.interview_id = interview_id

            # Save patient profile
            prof_name = patient_name if "patient_name" in dir() else ""
            prof_complaint = chief_complaint if "chief_complaint" in dir() else ""
            prof_history = medical_history if "medical_history" in dir() else ""
            prof_method = input_method if "input_method" in dir() else "text"

            if prof_name or prof_complaint or prof_history:
                profile = PatientProfile(
                    interview_id=interview_id,
                    name=prof_name,
                    age=patient_age if "patient_age" in dir() and patient_age > 0 else None,
                    chief_complaint=prof_complaint,
                    medical_history=prof_history,
                    input_method=prof_method,
                )
                db.create_profile(profile)
                st.write(f"Patient profile saved (input: {prof_method}).")

            # ── 2. Diarization ──
            use_pyannote = config.DIARIZATION_PRIMARY == "pyannote" and config.ENV["has_pyannote"]

            if use_pyannote:
                st.write(f"Running **Pyannote** speaker diarization (GPU: {config.ENV['has_gpu']})...")
                try:
                    from audio.diarize import SpeakerDiarizer
                    diarizer = SpeakerDiarizer(backend="pyannote")
                    diar_result = diarizer.diarize(tmp_path)
                    diar_segments = diar_result["segments"]
                    st.write(
                        f"Pyannote diarization complete — {diar_result['num_speakers']} speakers, "
                        f"{len(diar_segments)} segments in {diar_result['elapsed_sec']}s."
                    )

                    # ── 3. Transcription (Groq Whisper) ──
                    st.write("Transcribing audio via Groq Whisper API...")
                    from audio.transcribe import WhisperTranscriber
                    transcriber = WhisperTranscriber()
                    trans_segments = transcriber.transcribe(tmp_path, progress_callback=st.write)

                    # ── 4. Temporal Alignment ──
                    st.write("Aligning transcript with speaker labels...")
                    from audio.align import align_segments
                    aligned, align_metrics = align_segments(
                        diar_segments, trans_segments, interview_id, source_mode="offline"
                    )
                    st.write(
                        f"Alignment: {align_metrics['total_segments']} segments, "
                        f"avg confidence: {align_metrics['avg_confidence']:.2f}, "
                        f"{align_metrics['low_confidence_count']} low-confidence."
                    )

                    # Build utterances for LLM role labeling
                    utterances = [
                        {"speaker": seg.speaker_raw, "text": seg.text, "start_ms": seg.start_ms, "end_ms": seg.end_ms}
                        for seg in aligned
                    ]
                    diar_backend = "pyannote"

                except Exception as e:
                    st.warning(f"Pyannote failed: {e}. Falling back to AssemblyAI...")
                    use_pyannote = False

            if not use_pyannote:
                # AssemblyAI path (cloud fallback or primary on cloud)
                st.write("Transcribing and diarizing with **AssemblyAI** (acoustic speaker separation)...")
                try:
                    from audio.diarize import SpeakerDiarizer
                    diarizer = SpeakerDiarizer(backend="assemblyai")
                    diar_result = diarizer.diarize(tmp_path)
                    # AssemblyAI returns segments with text included
                    utterances = [
                        {"speaker": s["speaker"], "text": s.get("text", ""), "start_ms": s["start_ms"], "end_ms": s["end_ms"]}
                        for s in diar_result["segments"]
                    ]
                    st.write(
                        f"AssemblyAI complete — {diar_result['num_speakers']} speakers, "
                        f"{len(utterances)} utterances in {diar_result['elapsed_sec']}s."
                    )
                    aligned = None  # No separate alignment needed
                    diar_backend = "assemblyai"
                except Exception as e:
                    st.error(f"Diarization failed on both backends: {e}")
                    status.update(label="Pipeline failed at diarization", state="error")
                    st.stop()

            # ── 5. LLM Role Labeling ──
            st.write("Identifying speaker roles and names with **Groq LLM**...")
            try:
                speaker_map = label_roles_with_llm(utterances)
                for spk, info in speaker_map.items():
                    st.write(f"  Speaker {spk} → **{info['display']}** — _{info.get('reasoning', '')}_")
            except Exception as e:
                st.warning(f"LLM labeling failed: {e}. Using generic labels.")
                unique_spk = sorted(set(u["speaker"] for u in utterances))
                speaker_map = {}
                for i, spk in enumerate(unique_spk):
                    role = "CLINICIAN" if i == 0 else "PATIENT"
                    speaker_map[spk] = {"display": f"{role}_{i+1}", "role_base": role}

            # ── 6. Build and Save Segments ──
            st.write("Saving segments to database...")
            if aligned is not None:
                # Pyannote path: aligned is List[Segment], apply role mapping
                segments_to_insert = []
                for seg in aligned:
                    info = speaker_map.get(seg.speaker_raw, {"display": seg.speaker_raw, "role_base": "PATIENT"})
                    seg.speaker_raw = info["display"]
                    seg.speaker_role = info["role_base"]
                    segments_to_insert.append(seg)
            else:
                # AssemblyAI path: build from utterances
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

            # ── 7. Generate Embeddings (local only) ──
            if config.ENV["has_embeddings"]:
                st.write("Generating **semantic embeddings** (384-dim MiniLM)...")
                try:
                    from retrieval.embeddings import embed_and_store_segments
                    stored_segs = db.get_segments(interview_id)
                    n_embedded = embed_and_store_segments(stored_segs, db, progress_callback=st.write)
                    st.write(f"Embeddings stored for {n_embedded} segments.")
                except Exception as e:
                    st.warning(f"Embedding generation failed: {e}. Lexical search still works.")
            else:
                st.write("Embeddings skipped (sentence-transformers not available).")

            st.session_state.processing_complete = True
            status.update(
                label=f"Complete — {count} segments saved via {diar_backend}!",
                state="complete",
            )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Step 4: Results — Timeline + Transcript
if st.session_state.processing_complete:
    st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Interview Results</h3>", unsafe_allow_html=True)

    from database.supabase_client import SupabaseClient
    db = SupabaseClient()
    segments = db.get_segments(st.session_state.interview_id)

    if segments:
        # ── Speaker Timeline Visualization ──
        st.markdown(f"<small style='color:{config.BRAND_MUTED};font-weight:600;letter-spacing:1px;'>SPEAKER TIMELINE</small>", unsafe_allow_html=True)

        # Build timeline data
        total_duration = max(s.get("end_ms", 0) for s in segments) or 1
        timeline_html = '<div style="display:flex;height:32px;border-radius:6px;overflow:hidden;margin:0.5rem 0 1rem 0;border:1px solid #30363D;">'
        for seg in segments:
            start = seg.get("start_ms", 0)
            end = seg.get("end_ms", 0)
            width_pct = max(((end - start) / total_duration) * 100, 0.3)
            role = seg.get("speaker_role", "UNKNOWN")
            color = config.PATIENT_COLOR if role == "PATIENT" else config.CLINICIAN_COLOR
            raw = seg.get("speaker_raw", "")
            timeline_html += (
                f'<div style="width:{width_pct}%;background:{color};opacity:0.7;" '
                f'title="{raw}: {start//1000}s-{end//1000}s"></div>'
            )
        timeline_html += '</div>'

        # Legend
        timeline_html += (
            f'<div style="display:flex;gap:1.5rem;font-size:0.8rem;color:{config.BRAND_MUTED};">'
            f'<span><span style="color:{config.PATIENT_COLOR};">●</span> Patient</span>'
            f'<span><span style="color:{config.CLINICIAN_COLOR};">●</span> Clinician</span>'
            f'<span>Duration: {total_duration // 60000}:{(total_duration % 60000) // 1000:02d}</span>'
            f'</div>'
        )
        st.markdown(timeline_html, unsafe_allow_html=True)

        st.write("")

        # ── Speaker Summary ──
        unique_raw = sorted(set(s.get("speaker_raw", "") for s in segments))
        p_count = sum(1 for s in segments if s["speaker_role"] == "PATIENT")
        c_count = sum(1 for s in segments if s["speaker_role"] == "CLINICIAN")

        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Segments", len(segments))
        with cols[1]:
            st.metric("Patient Segments", p_count)
        with cols[2]:
            st.metric("Clinician Segments", c_count)
        with cols[3]:
            st.metric("Speakers Identified", len(unique_raw))

        st.write("")

        # ── Full Transcript ──
        st.markdown(f"<small style='color:{config.BRAND_MUTED};font-weight:600;letter-spacing:1px;'>TRANSCRIPT</small>", unsafe_allow_html=True)

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
                bg, border = config.PATIENT_BG, config.PATIENT_COLOR
            else:
                bg, border = config.CLINICIAN_BG, config.CLINICIAN_COLOR

            st.markdown(
                f'<div style="background-color:{bg};border-left:4px solid {border};'
                f'padding:0.6rem 1rem;margin:0.3rem 0;border-radius:0 8px 8px 0;">'
                f'<strong style="color:{border};">{raw}</strong> '
                f'<small style="color:{config.BRAND_MUTED};">({time_str})</small>'
                f'<br><span style="color:{config.BRAND_TEXT};">{text}</span></div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.success(
            f"Total: {len(segments)} segments — Patient: {p_count} / Clinician: {c_count} — "
            f"Speakers: {', '.join(unique_raw)}"
        )
        st.info("Go to **Query Analysis** to search and analyze this interview.")

# Manage Existing Interviews
st.divider()
st.subheader("Manage Existing Interviews")
with st.expander("View, Edit, and Delete Records", expanded=False):
    from database.supabase_client import SupabaseClient
    db = SupabaseClient()
    all_ivs = db.list_interviews()

    if all_ivs:
        for iv in all_ivs:
            iv_id = iv['interview_id']
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="center")
                with c1:
                    st.write(f"**{iv['title']}**")
                    st.caption(f"ID: `{iv_id}` | Created: {iv['created_at'][:16].replace('T', ' ')}")

                with c2:
                    if st.button("👥 Edit Roles", key=f"edit_{iv_id}", use_container_width=True):
                        st.session_state[f"editing_roles_{iv_id}"] = not st.session_state.get(f"editing_roles_{iv_id}", False)

                with c3:
                    if st.button("🗑️ Delete", key=f"del_{iv_id}", use_container_width=True):
                        if db.delete_interview(iv_id):
                            st.success("Deleted!")
                            if st.session_state.get('interview_id') == iv_id:
                                st.session_state.interview_id = None
                                st.session_state.processing_complete = False
                            st.rerun()
                        else:
                            st.error("Delete failed.")

                if st.session_state.get(f"editing_roles_{iv_id}", False):
                    st.markdown("##### 👥 Edit Speaker Roles")
                    current_map = iv.get("speaker_map", {})

                    if not current_map:
                        st.info("No speaker map found for this interview.")
                    else:
                        st.write("Modify the patient and clinician assignments below:")
                        new_map = {}
                        map_cols = st.columns(min(len(current_map), 4))
                        for i, (speaker, current_role) in enumerate(current_map.items()):
                            with map_cols[i % len(map_cols)]:
                                new_map[speaker] = st.selectbox(
                                    f"Role for {speaker}",
                                    config.SPEAKER_ROLES,
                                    index=config.SPEAKER_ROLES.index(current_role) if current_role in config.SPEAKER_ROLES else 0,
                                    key=f"remap_{iv_id}_{speaker}"
                                )

                        c_save, c_cancel = st.columns([1, 1])
                        with c_save:
                            if st.button("Save Roles", type="primary", key=f"save_roles_{iv_id}", use_container_width=True):
                                seg_count = db.get_segment_count(iv_id)
                                db.update_interview_speaker_map(iv_id, new_map)
                                db.update_segment_roles(iv_id, new_map)
                                st.success(f"Roles updated for {seg_count} segments!")
                                st.session_state[f"editing_roles_{iv_id}"] = False
                                st.rerun()
                        with c_cancel:
                            if st.button("Cancel", key=f"cancel_roles_{iv_id}", use_container_width=True):
                                st.session_state[f"editing_roles_{iv_id}"] = False
                                st.rerun()
    else:
        st.write("No interviews found in the database.")

