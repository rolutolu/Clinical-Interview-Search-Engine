"""
Page 3: Query & Analysis Dashboard

The centerpiece of the system — search, summarize, and analyze interviews
with speaker-aware retrieval and grounded LLM output.

Three modules:
    1. Summarization Engine
    2. Symptom-Based QA
    3. Automated Interview Analyzer
"""

import streamlit as st
import config

st.set_page_config(page_title="Query & Analysis", page_icon="QA", layout="wide")

st.title("Query & Analysis Dashboard")
st.markdown(f'<div style="background:linear-gradient(135deg,#fff3cd,#ffeeba);border:1px solid #ffc107;border-radius:8px;padding:0.8rem;color:#856404;font-size:0.9rem;">{config.ETHICS_DISCLAIMER}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# Interview Selection
# ══════════════════════════════════════════
try:
    from database.supabase_client import SupabaseClient
    db = SupabaseClient()
    interviews = db.list_interviews()
except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.info("Make sure your Supabase credentials are set in `.env`.")
    st.stop()

if not interviews:
    st.warning("No interviews found. Upload an interview on the **Upload** page first.")
    st.stop()

# Interview selector
interview_options = {
    f"{iv['title']} ({iv['interview_id']})": iv['interview_id']
    for iv in interviews
}
selected_label = st.selectbox("Select Interview", options=list(interview_options.keys()))
interview_id = interview_options[selected_label]

# ══════════════════════════════════════════
# Retrieval Controls (Sidebar-style in columns)
# ══════════════════════════════════════════
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

with col_ctrl1:
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        ["combined", "patient", "clinician"],
        help="Combined: all segments. Patient: patient speech only. Clinician: clinician speech only.",
    )
with col_ctrl2:
    k_value = st.slider("K (number of results)", min_value=1, max_value=20, value=config.DEFAULT_K)
with col_ctrl3:
    search_method = st.selectbox(
        "Search Method",
        ["hybrid", "semantic", "lexical"],
        help="Hybrid: lexical + semantic combined. Semantic: vector similarity. Lexical: keyword matching.",
    )

st.divider()

# ══════════════════════════════════════════
# Three Analysis Tabs
# ══════════════════════════════════════════
tab_summary, tab_qa, tab_analyzer = st.tabs([
    "Summarization",
    "Symptom QA",
    "Interview Analyzer",
])


# ── Helper: Display retrieved segments ──
def display_segments(segments, title="Retrieved Segments"):
    """Render retrieved segments with color-coded speaker labels."""
    if not segments:
        st.info("No segments retrieved.")
        return

    with st.expander(f"{title} ({len(segments)} segments)", expanded=False):
        for seg in segments:
            role = seg.get("speaker_role", "UNKNOWN")
            text = seg.get("text", "")
            score = seg.get("score", 0)
            start_ms = seg.get("start_ms", 0)
            end_ms = seg.get("end_ms", 0)
            sid = seg.get("segment_id", "?")
            time_str = f"{start_ms // 60000}:{(start_ms % 60000) // 1000:02d}-{end_ms // 60000}:{(end_ms % 60000) // 1000:02d}"

            color = "#f0fff4" if role == "PATIENT" else "#ebf8ff"
            border = "#38a169" if role == "PATIENT" else "#3182ce"
            icon = "[P]" if role == "PATIENT" else "[C]"

            st.markdown(
                f'<div style="background-color:{color};border-left:4px solid {border};'
                f'padding:0.5rem 1rem;margin:0.3rem 0;border-radius:0 4px 4px 0;">'
                f'<strong>{icon} {role}</strong> '
                f'<small>[S{sid} {time_str}] — score: {score:.3f}</small>'
                f'<br>{text}</div>',
                unsafe_allow_html=True,
            )


# ── Tab 1: Summarization ──
with tab_summary:
    st.subheader("Clinical Interview Summary")
    st.caption("Auto-generated summary grounded in retrieved speaker-labeled segments.")

    if st.button("Generate Summary", key="btn_summary", type="primary"):
        with st.spinner("Retrieving segments and generating summary..."):
            try:
                from retrieval.search import search
                from llm.grounded_llm import summarize

                # Retrieve all segments for the interview
                all_segments = db.get_segments(interview_id)

                if not all_segments:
                    st.warning("No segments found for this interview.")
                else:
                    # Generate summary
                    summary = summarize(all_segments, interview_id)
                    st.markdown(summary)
                    st.divider()
                    display_segments(all_segments, "Source Segments")

            except Exception as e:
                st.error(f"Summary generation failed: {e}")
                st.info("Make sure the Groq API key is configured and segments exist in the database.")


# ── Tab 2: Symptom QA ──
with tab_qa:
    st.subheader("Symptom-Based Question Answering")
    st.caption("Ask questions about the interview — answers are grounded in retrieved segments.")

    # Example queries
    example_queries = [
        "What symptoms did the patient report?",
        "When did the clinician ask about medications?",
        "Does the patient have any allergies?",
        "What is the patient's chief complaint?",
        "Were any follow-up appointments discussed?",
    ]

    query = st.text_input(
        "Enter your question",
        placeholder="What symptoms did the patient report?",
        key="qa_query",
    )

    st.caption("Example queries:")
    example_cols = st.columns(3)
    for i, eq in enumerate(example_queries):
        with example_cols[i % 3]:
            if st.button(eq, key=f"example_{i}", use_container_width=True):
                st.session_state.qa_query = eq
                st.rerun()

    if query and st.button("Search & Answer", key="btn_qa", type="primary"):
        with st.spinner(f"Searching ({retrieval_mode} mode, K={k_value})..."):
            try:
                from retrieval.search import search
                from llm.grounded_llm import answer_question

                # Retrieve relevant segments
                results = search(
                    query=query,
                    interview_id=interview_id,
                    mode=retrieval_mode,
                    k=k_value,
                    method=search_method,
                    db=db,
                )

                if not results:
                    st.warning("No relevant segments found. Try broadening your query or changing retrieval mode.")
                else:
                    # Generate grounded answer
                    answer = answer_question(query, results)
                    st.markdown("### Answer")
                    st.markdown(answer)
                    st.divider()
                    display_segments(results, "Retrieved Segments")

            except Exception as e:
                st.error(f"QA failed: {e}")


# ── Tab 3: Interview Analyzer ──
with tab_analyzer:
    st.subheader("Automated Interview Analysis")
    st.caption("Structured analysis: timeline, symptoms, medications, follow-ups, and potential gaps.")

    if st.button("Run Analysis", key="btn_analyzer", type="primary"):
        with st.spinner("Analyzing interview..."):
            try:
                from llm.grounded_llm import analyze_interview

                all_segments = db.get_segments(interview_id)

                if not all_segments:
                    st.warning("No segments found for this interview.")
                else:
                    analysis = analyze_interview(all_segments)
                    st.markdown(analysis)
                    st.divider()
                    display_segments(all_segments, "Source Segments")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
