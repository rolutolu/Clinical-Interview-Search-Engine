"""
Page 3: Query & Analysis Dashboard

Speaker-aware retrieval with grounded LLM output. Three modules:
    1. Summarization Engine — structured clinical summary with citations
    2. Symptom-Based QA — retrieval-augmented question answering
    3. Automated Interview Analyzer — 8-section structured analysis

Features:
    - All search methods available (lexical, semantic, hybrid) based on environment
    - Streaming LLM output (token-by-token display)
    - Cross-encoder reranking toggle (local mode)
    - Speaker-aware color-coded segment display
    - Professional dark medical UI
"""

import streamlit as st
import config

st.set_page_config(page_title="Query & Analysis", page_icon="🔍", layout="wide")

# ── Header ──
st.markdown(
    f'<p style="font-size:1.8rem;font-weight:700;'
    f'background:linear-gradient(135deg,{config.BRAND_PRIMARY},{config.BRAND_SECONDARY});'
    f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
    f'Query & Analysis Dashboard</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="background:linear-gradient(135deg,rgba(255,193,7,0.1),rgba(255,152,0,0.1));'
    f'border:1px solid rgba(255,193,7,0.3);border-radius:10px;padding:0.8rem 1.2rem;'
    f'color:#FFC107;font-size:0.88rem;">{config.ETHICS_DISCLAIMER}</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════
# Interview Selection
# ══════════════════════════════════════════
try:
    from database.supabase_client import SupabaseClient
    db = SupabaseClient()
    interviews = db.list_interviews()
except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.stop()

if not interviews:
    st.warning("No interviews found. Upload an interview on the **Upload** page first.")
    st.stop()

interview_options = {
    f"{iv['title']} ({iv['interview_id']})": iv['interview_id']
    for iv in interviews
}
selected_label = st.selectbox("Select Interview", options=list(interview_options.keys()))
interview_id = interview_options[selected_label]

# Show interview stats
seg_count = db.get_segment_count(interview_id)
st.caption(f"Interview **{interview_id}** — {seg_count} segments indexed")

# ══════════════════════════════════════════
# Retrieval Controls
# ══════════════════════════════════════════
st.markdown(f"<small style='color:{config.BRAND_MUTED};font-weight:600;letter-spacing:1px;'>RETRIEVAL SETTINGS</small>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    retrieval_mode = st.selectbox(
        "Speaker Filter",
        config.RETRIEVAL_MODES,
        help="combined: all segments · patient: patient speech only · clinician: clinician speech only",
    )
with col2:
    k_value = st.slider("K (results)", min_value=1, max_value=20, value=config.DEFAULT_K)
with col3:
    search_method = st.selectbox(
        "Search Method",
        config.SEARCH_METHODS,
        index=config.SEARCH_METHODS.index(config.DEFAULT_SEARCH_METHOD) if config.DEFAULT_SEARCH_METHOD in config.SEARCH_METHODS else 0,
        help="lexical: BM25 keyword · semantic: vector cosine · hybrid: RRF fusion",
    )
with col4:
    use_rerank = st.checkbox(
        "Rerank",
        value=config.RERANKER_ENABLED,
        help="Cross-encoder reranking (ms-marco-MiniLM)",
        disabled=not config.RERANKER_ENABLED,
    )

st.divider()

# ══════════════════════════════════════════
# Helper: search with current settings
# ══════════════════════════════════════════
def run_search(query, k_override=None):
    """Execute search with current control settings."""
    from retrieval.search import search
    results = search(
        query=query,
        interview_id=interview_id,
        mode=retrieval_mode,
        k=k_override or k_value,
        method=search_method,
        db=db,
        rerank=use_rerank,
    )
    if not results:
        # Fallback: return all segments for this mode
        speaker_role = None
        if retrieval_mode == "patient":
            speaker_role = "PATIENT"
        elif retrieval_mode == "clinician":
            speaker_role = "CLINICIAN"
        results = db.get_segments(interview_id, speaker_role=speaker_role)
    return results


# ══════════════════════════════════════════
# Helper: display segments with dark theme
# ══════════════════════════════════════════
def display_segments(segments, title="Retrieved Segments"):
    """Render segments with color-coded speaker labels in dark theme."""
    if not segments:
        st.info("No segments retrieved.")
        return

    with st.expander(f"{title} ({len(segments)} segments)", expanded=False):
        for seg in segments:
            role = seg.get("speaker_role", "UNKNOWN")
            raw = seg.get("speaker_raw", role)
            text = seg.get("text", "")
            score = seg.get("score", 0)
            rerank_score = seg.get("rerank_score")
            start_ms = seg.get("start_ms", 0) or 0
            end_ms = seg.get("end_ms", 0) or 0
            sid = seg.get("segment_id", "?")
            method = seg.get("method", "")
            time_str = (
                f"{start_ms // 60000}:{(start_ms % 60000) // 1000:02d}-"
                f"{end_ms // 60000}:{(end_ms % 60000) // 1000:02d}"
            )

            if role == "PATIENT":
                bg, border = config.PATIENT_BG, config.PATIENT_COLOR
            else:
                bg, border = config.CLINICIAN_BG, config.CLINICIAN_COLOR

            score_parts = []
            if score:
                score_parts.append(f"score: {score:.4f}")
            if rerank_score is not None:
                score_parts.append(f"rerank: {rerank_score:.3f}")
            if method:
                score_parts.append(method)
            score_str = " · ".join(score_parts)

            st.markdown(
                f'<div style="background-color:{bg};border-left:4px solid {border};'
                f'padding:0.6rem 1rem;margin:0.3rem 0;border-radius:0 8px 8px 0;">'
                f'<strong style="color:{border};">{raw}</strong> '
                f'<small style="color:{config.BRAND_MUTED};">[S_{sid} {time_str}] {score_str}</small>'
                f'<br><span style="color:{config.BRAND_TEXT};">{text}</span></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════
# Three Analysis Tabs
# ══════════════════════════════════════════
tab_summary, tab_qa, tab_analyzer = st.tabs([
    "📝 Summarization",
    "❓ Symptom QA",
    "📊 Interview Analyzer",
])


# ── Tab 1: Summarization ──
with tab_summary:
    st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Clinical Interview Summary</h3>", unsafe_allow_html=True)
    st.caption("Structured summary grounded in speaker-labeled segments with citations.")

    if st.button("Generate Summary", key="btn_summary", type="primary", use_container_width=True):
        all_segments = db.get_segments(interview_id)
        if not all_segments:
            st.warning("No segments found for this interview.")
        else:
            try:
                from llm.grounded_llm import summarize
                with st.spinner("Generating..."):
                    output = st.write_stream(summarize(all_segments, interview_id, stream=True))
                st.divider()
                display_segments(all_segments, "Source Segments")
            except Exception as e:
                st.error(f"Summary failed: {e}")
                # Non-streaming fallback
                try:
                    from llm.grounded_llm import summarize
                    result = summarize(all_segments, interview_id, stream=False)
                    st.markdown(result)
                    display_segments(all_segments, "Source Segments")
                except Exception as e2:
                    st.error(f"Fallback also failed: {e2}")


# ── Tab 2: Symptom QA ──
with tab_qa:
    st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Symptom-Based Question Answering</h3>", unsafe_allow_html=True)
    st.caption("Ask questions — answers grounded in retrieved speaker-labeled segments.")

    query = st.text_input(
        "Enter your question",
        placeholder="What symptoms did the patient report?",
        key="qa_input",
    )

    st.caption("Example queries (copy and paste):")
    examples = [
        "What symptoms did the patient report?",
        "What questions did the clinician ask?",
        "What is the patient's chief complaint?",
        "Were any medications mentioned?",
        "How did the patient describe their emotional state?",
        "What therapeutic approaches did the clinician use?",
        "Were any follow-up appointments discussed?",
        "Were any safety concerns raised?",
    ]
    ex_cols = st.columns(4)
    for i, eq in enumerate(examples):
        with ex_cols[i % 4]:
            st.code(eq, language=None)

    if query and st.button("Search & Answer", key="btn_qa", type="primary", use_container_width=True):
        results = run_search(query)
        if results:
            try:
                from llm.grounded_llm import answer_question

                # Show search info
                st.markdown(
                    f"<small style='color:{config.BRAND_MUTED};'>"
                    f"Retrieved {len(results)} segments via {search_method} search "
                    f"({retrieval_mode} mode, K={k_value})"
                    f"{'+ reranking' if use_rerank else ''}</small>",
                    unsafe_allow_html=True,
                )

                st.markdown(f"### Answer")
                with st.spinner("Generating..."):
                    output = st.write_stream(answer_question(query, results, stream=True))
                st.divider()
                display_segments(results, "Retrieved Segments")
            except Exception as e:
                st.error(f"QA failed: {e}")
                # Non-streaming fallback
                try:
                    from llm.grounded_llm import answer_question
                    result = answer_question(query, results, stream=False)
                    st.markdown(result)
                    display_segments(results, "Retrieved Segments")
                except Exception as e2:
                    st.error(f"Fallback also failed: {e2}")
        else:
            st.warning("No segments found.")


# ── Tab 3: Interview Analyzer ──
with tab_analyzer:
    st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Automated Interview Analysis</h3>", unsafe_allow_html=True)
    st.caption("Structured 8-section analysis: timeline, symptoms, medications, dynamics, gaps.")

    if st.button("Run Analysis", key="btn_analyzer", type="primary", use_container_width=True):
        all_segments = db.get_segments(interview_id)
        if not all_segments:
            st.warning("No segments found for this interview.")
        else:
            try:
                from llm.grounded_llm import analyze_interview
                with st.spinner("Analyzing..."):
                    output = st.write_stream(analyze_interview(all_segments, stream=True))
                st.divider()
                display_segments(all_segments, "Source Segments")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                # Non-streaming fallback
                try:
                    from llm.grounded_llm import analyze_interview
                    result = analyze_interview(all_segments, stream=False)
                    st.markdown(result)
                    display_segments(all_segments, "Source Segments")
                except Exception as e2:
                    st.error(f"Fallback also failed: {e2}")