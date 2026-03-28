"""
Page 4: Evaluation Dashboard

Research-grade retrieval evaluation with:
    - LLM-as-Judge auto-generated relevance labels (Faggioli et al. 2023)
    - 6 metrics: P@K, R@K, F1@K, nDCG@K, MRR, MAP
    - Per-mode breakdown: overall, patient-only, clinician-only
    - Method comparison: lexical vs semantic vs hybrid (local)
    - Interactive charts via Plotly
    - Per-query drill-down
    - Export results as CSV + JSON
"""

import streamlit as st
import pandas as pd
import json
import re
import config


def _parse_llm_segment_ids(content: str, valid_ids: set) -> list:
    """
    Robustly extract segment IDs from LLM response.
    Uses multiple fallback strategies and validates against actual DB IDs.
    """
    cleaned = content.strip()

    # Strip markdown code blocks (```json ... ``` or ``` ... ```)
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Strategy 1: Direct JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            ids = [str(x).strip() for x in parsed]
            return [sid for sid in ids if sid in valid_ids]
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Regex — find first JSON array in the response
    match = re.search(r'\[([^\]]*?)\]', content)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                ids = [str(x).strip() for x in parsed]
                return [sid for sid in ids if sid in valid_ids]
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Find any quoted strings that match valid segment IDs
    found = re.findall(r'["\']([a-f0-9\-]{4,})["\']', content)
    return [sid for sid in found if sid in valid_ids]


st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")

# ── Header ──
st.markdown(
    f'<p style="font-size:1.8rem;font-weight:700;'
    f'background:linear-gradient(135deg,{config.BRAND_PRIMARY},{config.BRAND_SECONDARY});'
    f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
    f'Retrieval Evaluation Dashboard</p>',
    unsafe_allow_html=True,
)
st.caption("P@K · R@K · F1@K · nDCG@K · MRR · MAP — across speaker-aware retrieval modes")


# Database + Interview Selection
try:
    from database.supabase_client import SupabaseClient
    db = SupabaseClient()
    interviews = db.list_interviews()
except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.stop()

if not interviews:
    st.warning("No interviews found. Upload an interview first.")
    st.stop()

interview_options = {
    f"{iv['title']} ({iv['interview_id']})": iv['interview_id']
    for iv in interviews
}
selected_label = st.selectbox("Select Interview", options=list(interview_options.keys()))
interview_id = interview_options[selected_label]

all_segments = db.get_segments(interview_id)
if not all_segments:
    st.warning("No segments found. Process an interview first.")
    st.stop()

p_count = sum(1 for s in all_segments if s.get("speaker_role") == "PATIENT")
c_count = sum(1 for s in all_segments if s.get("speaker_role") == "CLINICIAN")
st.caption(f"{len(all_segments)} segments — Patient: {p_count} · Clinician: {c_count}")
# ── Check for embeddings ──
_has_embeddings = False
try:
    test = db.vector_search([0.0] * config.EMBEDDING_DIM, 1, interview_id)
    _has_embeddings = bool(test)
except Exception:
    pass

if not _has_embeddings:
    st.warning("No embeddings stored for this interview. Semantic and hybrid search will return identical results to lexical.")
    if st.button("Generate Embeddings Now", type="primary"):
        with st.spinner("Generating 384-dim MiniLM embeddings for all segments..."):
            try:
                from retrieval.embeddings import embed_and_store_segments
                n = embed_and_store_segments(all_segments, db, progress_callback=st.write)
                st.success(f"Embeddings generated and stored for {n} segments. Reload the page and re-run evaluation.")
                _has_embeddings = True
            except Exception as e:
                st.error(f"Embedding generation failed: {e}")
else:
    st.success("Embeddings detected — semantic and hybrid search are fully active.")
st.divider()

# Step 1: Relevance Labels
st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Step 1: Evaluation Labels</h3>", unsafe_allow_html=True)
st.caption("LLM-as-Judge generates ground truth relevance labels for test queries.")

tab_auto, tab_manual = st.tabs(["Auto-Generate (LLM-as-Judge)", "Upload Manual Labels"])

if "eval_labels" not in st.session_state:
    st.session_state.eval_labels = None

with tab_auto:
    st.markdown(
        f"<small style='color:{config.BRAND_MUTED};'>"
        f"Based on Faggioli et al. 2023 and Thomas et al. 2024. "
        f"The LLM judges segment relevance for each test query.</small>",
        unsafe_allow_html=True,
    )

    default_queries = [
        {"query_id": "q01", "query_text": "What symptoms did the patient report?", "retrieval_mode": "combined"},
        {"query_id": "q02", "query_text": "What symptoms did the patient report?", "retrieval_mode": "patient"},
        {"query_id": "q03", "query_text": "What questions did the clinician ask?", "retrieval_mode": "combined"},
        {"query_id": "q04", "query_text": "What questions did the clinician ask?", "retrieval_mode": "clinician"},
        {"query_id": "q05", "query_text": "What is the patient's chief complaint?", "retrieval_mode": "combined"},
        {"query_id": "q06", "query_text": "What is the patient's chief complaint?", "retrieval_mode": "patient"},
        {"query_id": "q07", "query_text": "Were any medications discussed?", "retrieval_mode": "combined"},
        {"query_id": "q08", "query_text": "What follow-up steps were discussed?", "retrieval_mode": "combined"},
        {"query_id": "q09", "query_text": "How did the patient describe their emotional state?", "retrieval_mode": "patient"},
        {"query_id": "q10", "query_text": "What therapeutic approaches did the clinician use?", "retrieval_mode": "clinician"},
        {"query_id": "q11", "query_text": "Were there any signs of patient resistance?", "retrieval_mode": "combined"},
        {"query_id": "q12", "query_text": "What relationship dynamics were discussed?", "retrieval_mode": "combined"},
        {"query_id": "q13", "query_text": "Did the clinician discuss any diagnoses?", "retrieval_mode": "clinician"},
        {"query_id": "q14", "query_text": "What coping mechanisms were mentioned?", "retrieval_mode": "combined"},
        {"query_id": "q15", "query_text": "Were any safety concerns raised?", "retrieval_mode": "combined"},
    ]

    st.write(f"**{len(default_queries)} test queries** across combined, patient-only, and clinician-only modes.")

    if st.button("Generate Labels with LLM-as-Judge", type="primary"):
        import httpx

        progress = st.progress(0, text="Generating relevance labels...")
        all_labels = []

        for i, q in enumerate(default_queries):
            progress.progress((i + 1) / len(default_queries), text=f"Query {i+1}/{len(default_queries)}: {q['query_text'][:50]}...")

            if q["retrieval_mode"] == "patient":
                mode_segments = [s for s in all_segments if s.get("speaker_role") == "PATIENT"]
            elif q["retrieval_mode"] == "clinician":
                mode_segments = [s for s in all_segments if s.get("speaker_role") == "CLINICIAN"]
            else:
                mode_segments = all_segments

            if not mode_segments:
                all_labels.append({**q, "relevant_segment_ids": []})
                continue

            seg_lines = []
            for s in mode_segments:
                sid = s.get("segment_id", "?")
                text = s.get("text", "")
                role = s.get("speaker_role", "?")
                seg_lines.append(f"[{sid}] ({role}): {text}")
            seg_text = "\n".join(seg_lines)

            prompt = f"""You are an IR evaluation judge. Determine which segments are RELEVANT to the query.

A segment is relevant if it directly helps answer the query or provides essential context.

QUERY: {q['query_text']}

SEGMENTS:
{seg_text}

Respond with ONLY a JSON array of relevant segment IDs: ["id1", "id2"]
If none are relevant: []"""

            valid_ids = {s.get("segment_id", "") for s in mode_segments}

            try:
                response = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {config.GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": config.LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1000, "temperature": 0.1},
                    timeout=60.0,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                relevant_ids = _parse_llm_segment_ids(content, valid_ids)
            except Exception as e:
                st.caption(f"⚠️ LLM label generation failed for {q['query_id']}: {e}")
                relevant_ids = []

            all_labels.append({**q, "relevant_segment_ids": relevant_ids})

        progress.progress(1.0, text="Complete!")
        st.session_state.eval_labels = all_labels
        total_rel = sum(len(l["relevant_segment_ids"]) for l in all_labels)
        st.success(f"Generated labels for {len(all_labels)} queries — {total_rel} relevance judgments total.")

        with st.expander("Preview Labels"):
            for l in all_labels:
                n = len(l["relevant_segment_ids"])
                st.write(f"**{l['query_id']}** [{l['retrieval_mode']}]: \"{l['query_text']}\" → {n} relevant")

with tab_manual:
    uploaded_labels = st.file_uploader("Upload relevance_labels.json", type=["json"], key="eval_upload")
    if uploaded_labels:
        try:
            labels = json.load(uploaded_labels)
            st.write(f"Loaded {len(labels)} labels.")
            if st.button("Use These Labels"):
                st.session_state.eval_labels = labels
                st.success("Labels loaded!")
        except Exception as e:
            st.error(f"Parse failed: {e}")

# Step 2: Run Evaluation
st.divider()
st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Step 2: Run Evaluation</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    k_values = st.multiselect("K Values", options=[1, 2, 3, 5, 10, 15, 20], default=config.K_VALUES)
with col2:
    eval_methods = st.multiselect(
        "Search Methods to Compare",
        options=config.SEARCH_METHODS,
        default=config.SEARCH_METHODS,
    )
with col3:
    use_rerank = st.checkbox("Apply Reranking", value=config.RERANKER_ENABLED, disabled=not config.RERANKER_ENABLED)

if st.session_state.eval_labels is None:
    st.info("Generate or upload evaluation labels first (Step 1).")
elif st.button("Run Evaluation", type="primary", use_container_width=True):
    eval_labels = st.session_state.eval_labels

    with st.spinner("Running evaluation..."):
        from evaluation.metrics import compute_all_metrics, aggregate_metrics
        from retrieval.search import search

        by_mode = {"combined": [], "patient": [], "clinician": []}
        for label in eval_labels:
            mode = label.get("retrieval_mode", "combined")
            if mode in by_mode:
                by_mode[mode].append(label)

        mode_display = {"combined": "Overall", "patient": "Patient-Only", "clinician": "Clinician-Only"}

        # Store all results: results[method][mode][k] = {metrics}
        all_results = {}
        per_query_details = []

        for method in eval_methods:
            all_results[method] = {}
            for mode, labels in by_mode.items():
                if not labels:
                    continue
                mode_results = {}
                for k in k_values:
                    query_metrics = []
                    for label in labels:
                        relevant = label["relevant_segment_ids"]
                        if not relevant:
                            continue
                        try:
                            results = search(
                                query=label["query_text"],
                                interview_id=interview_id,
                                mode=mode, k=k, method=method, db=db, rerank=use_rerank,
                            )
                            if not results:
                                sr = None
                                if mode == "patient": sr = "PATIENT"
                                elif mode == "clinician": sr = "CLINICIAN"
                                results = db.get_segments(interview_id, speaker_role=sr)[:k]
                            retrieved = [r.get("segment_id", "") for r in results]
                        except Exception:
                            retrieved = []

                        metrics = compute_all_metrics(retrieved, relevant, k)
                        query_metrics.append(metrics)

                        per_query_details.append({
                            "method": method, "mode": mode, "k": k,
                            "query_id": label["query_id"], "query_text": label["query_text"],
                            **{m: round(v, 4) for m, v in metrics.items()},
                        })

                    if query_metrics:
                        mode_results[k] = aggregate_metrics(query_metrics)

                if mode_results:
                    all_results[method][mode_display[mode]] = mode_results

        # Display Results
        st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Results</h3>", unsafe_allow_html=True)

        if not all_results:
            st.warning("No results. Ensure labels have relevant segment IDs.")
        else:
            # ── Summary Tables per Method ──
            for method, method_data in all_results.items():
                st.markdown(f"#### Search Method: `{method}`" + (" + reranking" if use_rerank else ""))

                for mode_name, mode_results in method_data.items():
                    st.markdown(f"**{mode_name} Retrieval**")
                    rows = []
                    for k, metrics in sorted(mode_results.items()):
                        rows.append({
                            "K": k,
                            "P@K": round(metrics["precision"], 4),
                            "R@K": round(metrics["recall"], 4),
                            "F1@K": round(metrics["f1"], 4),
                            "nDCG@K": round(metrics["ndcg"], 4),
                            "MRR": round(metrics["mrr"], 4),
                            "MAP": round(metrics["map"], 4),
                            "Queries": metrics.get("num_queries", 0),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ── Charts ──
            st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Metric Charts</h3>", unsafe_allow_html=True)
            st.caption("Each chart compares search methods (lexical, semantic, hybrid) within a retrieval mode.")

            chart_rows = []
            for method, method_data in all_results.items():
                for mode_name, mode_results in method_data.items():
                    for k, metrics in sorted(mode_results.items()):
                        chart_rows.append({
                            "K": k,
                            "K_label": str(k),
                            "Precision": metrics["precision"],
                            "Recall": metrics["recall"],
                            "F1": metrics["f1"],
                            "nDCG": metrics["ndcg"],
                            "Method": method,
                            "Mode": mode_name,
                        })

            if chart_rows:
                chart_df = pd.DataFrame(chart_rows)
                # Sort by K numerically
                chart_df = chart_df.sort_values("K")
                # Use K_label for display but keep numeric order
                chart_df = chart_df.set_index("K_label", drop=False)

                # Get all modes that actually have data
                available_modes = chart_df["Mode"].unique().tolist()

                for mode_name in available_modes:
                    mode_df = chart_df[chart_df["Mode"] == mode_name].copy()
                    if mode_df.empty:
                        continue

                    st.markdown(f"#### {mode_name} Retrieval")
                    col_p, col_r = st.columns(2)
                    with col_p:
                        st.markdown("**Precision@K**")
                        pivot = mode_df.pivot_table(index="K", columns="Method", values="Precision", sort=True)
                        st.line_chart(pivot)
                    with col_r:
                        st.markdown("**Recall@K**")
                        pivot = mode_df.pivot_table(index="K", columns="Method", values="Recall", sort=True)
                        st.line_chart(pivot)

                    col_f, col_n = st.columns(2)
                    with col_f:
                        st.markdown("**F1@K**")
                        pivot = mode_df.pivot_table(index="K", columns="Method", values="F1", sort=True)
                        st.line_chart(pivot)
                    with col_n:
                        st.markdown("**nDCG@K**")
                        pivot = mode_df.pivot_table(index="K", columns="Method", values="nDCG", sort=True)
                        st.line_chart(pivot)

            # ── Per-Query Drill-Down ──
            st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Per-Query Breakdown</h3>", unsafe_allow_html=True)

            if per_query_details:
                detail_df = pd.DataFrame(per_query_details)
                with st.expander(f"All per-query metrics ({len(per_query_details)} rows)"):
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)

                # Export
                csv_data = detail_df.to_csv(index=False)
                st.download_button(
                    "Download Results as CSV",
                    data=csv_data,
                    file_name="evaluation_results.csv",
                    mime="text/csv",
                )

# Step 3: Save & Export
st.divider()
st.markdown(f"<h3 style='color:{config.BRAND_TEXT};'>Step 3: Export</h3>", unsafe_allow_html=True)

if st.session_state.eval_labels:
    col_s, col_d = st.columns(2)
    with col_s:
        if st.button("Save Labels to Supabase"):
            try:
                count = db.insert_eval_labels(st.session_state.eval_labels)
                st.success(f"{count} labels saved.")
            except Exception as e:
                st.error(f"Save failed: {e}")
    with col_d:
        st.download_button(
            "Download Labels as JSON",
            data=json.dumps(st.session_state.eval_labels, indent=2),
            file_name="relevance_labels.json",
            mime="application/json",
        )
