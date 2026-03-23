"""
Page 4: Evaluation Dashboard

Runs Precision@K and Recall@K evaluation for retrieval quality.
Reports: overall, patient-only, clinician-only across multiple K values.

Features:
    - Auto-generate relevance labels using LLM-as-judge (Faggioli et al. 2023)
    - Manual label upload
    - Charts for P@K and R@K across modes and K values
    - Per-query breakdown
"""

import streamlit as st
import pandas as pd
import json
import config

st.set_page_config(page_title="Evaluation", page_icon="EV", layout="wide")

st.title("Retrieval Evaluation Dashboard")
st.caption("Precision@K and Recall@K — overall, patient-only, and clinician-only.")

# ══════════════════════════════════════════
# Database Connection
# ══════════════════════════════════════════
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
selected_label = st.selectbox("Select Interview to Evaluate", options=list(interview_options.keys()))
interview_id = interview_options[selected_label]

# Load segments for this interview
all_segments = db.get_segments(interview_id)
if not all_segments:
    st.warning("No segments found for this interview. Process an interview first.")
    st.stop()

st.success(f"Loaded {len(all_segments)} segments for evaluation.")

st.divider()

# ══════════════════════════════════════════
# Step 1: Generate or Upload Evaluation Labels
# ══════════════════════════════════════════
st.subheader("Step 1: Evaluation Labels")
st.caption(
    "Relevance labels define which segments are relevant to each test query. "
    "You can auto-generate them using LLM-as-judge or upload manually."
)

tab_auto, tab_manual = st.tabs(["Auto-Generate (LLM-as-Judge)", "Upload Manual Labels"])

# Store labels and results in session state
if "eval_labels" not in st.session_state:
    st.session_state.eval_labels = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

with tab_auto:
    st.markdown(
        "**LLM-as-Judge** (Faggioli et al. 2023, Thomas et al. 2024): "
        "The LLM reads each segment and judges whether it is relevant to each test query. "
        "This is a widely-accepted evaluation methodology in IR research."
    )

    # Default test queries covering all retrieval modes
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

    st.write(f"Will generate relevance labels for **{len(default_queries)} queries** across combined, patient-only, and clinician-only modes.")

    if st.button("Generate Labels with LLM-as-Judge", type="primary"):
        import httpx

        progress = st.progress(0, text="Generating relevance labels...")
        all_labels = []

        for i, q in enumerate(default_queries):
            progress.progress((i + 1) / len(default_queries), text=f"Query {i+1}/{len(default_queries)}: {q['query_text'][:50]}...")

            # Filter segments by mode
            if q["retrieval_mode"] == "patient":
                mode_segments = [s for s in all_segments if s.get("speaker_role") == "PATIENT"]
            elif q["retrieval_mode"] == "clinician":
                mode_segments = [s for s in all_segments if s.get("speaker_role") == "CLINICIAN"]
            else:
                mode_segments = all_segments

            if not mode_segments:
                all_labels.append({**q, "relevant_segment_ids": []})
                continue

            # Build segment list for LLM
            seg_lines = []
            for s in mode_segments:
                sid = s.get("segment_id", "?")
                text = s.get("text", "")
                role = s.get("speaker_role", "?")
                seg_lines.append(f"[{sid}] ({role}): {text}")
            seg_text = "\n".join(seg_lines)

            prompt = f"""You are an IR evaluation judge. Given a query and a list of transcript segments, determine which segments are RELEVANT to answering the query.

A segment is relevant if it directly contains information that helps answer the query, or provides essential context for the answer.

QUERY: {q['query_text']}

SEGMENTS:
{seg_text}

Respond with ONLY a JSON array of relevant segment IDs. Example: ["abc123", "def456"]
If no segments are relevant, respond with: []

No other text. Just the JSON array."""

            try:
                response = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": config.LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000,
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

                relevant_ids = json.loads(content)
                if not isinstance(relevant_ids, list):
                    relevant_ids = []
            except Exception:
                relevant_ids = []

            all_labels.append({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "relevant_segment_ids": relevant_ids,
                "retrieval_mode": q["retrieval_mode"],
            })

        progress.progress(1.0, text="Label generation complete!")
        st.session_state.eval_labels = all_labels

        # Show summary
        total_relevant = sum(len(l["relevant_segment_ids"]) for l in all_labels)
        st.success(f"Generated labels for {len(all_labels)} queries. Total relevant judgments: {total_relevant}.")

        # Preview
        with st.expander("Preview Generated Labels"):
            for l in all_labels:
                n = len(l["relevant_segment_ids"])
                st.write(f"**{l['query_id']}** [{l['retrieval_mode']}]: \"{l['query_text']}\" → {n} relevant segments")

with tab_manual:
    st.caption("Upload a JSON file with query-relevance labels.")
    uploaded_labels = st.file_uploader("Upload relevance_labels.json", type=["json"], key="eval_upload")

    if uploaded_labels:
        try:
            labels = json.load(uploaded_labels)
            st.json(labels[:3])
            st.write(f"Loaded {len(labels)} evaluation labels.")
            if st.button("Use These Labels"):
                st.session_state.eval_labels = labels
                st.success("Labels loaded!")
        except Exception as e:
            st.error(f"Failed to parse labels: {e}")

# ══════════════════════════════════════════
# Step 2: Run Evaluation
# ══════════════════════════════════════════
st.divider()
st.subheader("Step 2: Run Evaluation")

col1, col2 = st.columns(2)
with col1:
    k_values = st.multiselect(
        "K Values to Test",
        options=[1, 2, 3, 5, 10, 15, 20],
        default=config.K_VALUES,
    )
with col2:
    available_methods = getattr(config, "SEARCH_METHODS", ["lexical", "semantic", "hybrid"])
    search_methods = st.multiselect(
        "Search Methods to Compare",
        options=available_methods,
        default=available_methods,
        key="eval_methods",
        help="Select one or more search methods to compare on the charts.",
    )

if st.session_state.eval_labels is None:
    st.info("Generate or upload evaluation labels first (Step 1).")
elif not search_methods:
    st.warning("Select at least one search method to run evaluation.")
elif st.button("Run Evaluation", type="primary", width='stretch'):
    eval_labels = st.session_state.eval_labels

    with st.spinner("Running evaluation across all methods, modes and K values..."):
        from evaluation.metrics import precision_at_k, recall_at_k
        from retrieval.search import search

        # Group labels by mode
        by_mode = {"combined": [], "patient": [], "clinician": []}
        for label in eval_labels:
            mode = label.get("retrieval_mode", "combined")
            if mode in by_mode:
                by_mode[mode].append(label)

        # results[method][mode_display] = {k: {precision, recall, num_queries}}
        results = {}
        mode_display = {"combined": "Overall", "patient": "Patient-Only", "clinician": "Clinician-Only"}

        for method in search_methods:
            method_results = {}
            for mode, labels in by_mode.items():
                if not labels:
                    continue
                mode_results = {}
                for k in k_values:
                    precisions = []
                    recalls = []
                    for label in labels:
                        query = label["query_text"]
                        relevant = label["relevant_segment_ids"]

                        if not relevant:
                            continue

                        try:
                            search_results = search(
                                query=query,
                                interview_id=interview_id,
                                mode=mode,
                                k=k,
                                method=method,
                                db=db,
                            )

                            if not search_results:
                                speaker_role = None
                                if mode == "patient":
                                    speaker_role = "PATIENT"
                                elif mode == "clinician":
                                    speaker_role = "CLINICIAN"
                                search_results = db.get_segments(interview_id, speaker_role=speaker_role)[:k]

                            retrieved = [r.get("segment_id", "") for r in search_results]
                        except Exception as e:
                            if method != "lexical":
                                st.warning(f"[!] [{method}] error: {e}")
                            retrieved = []

                        precisions.append(precision_at_k(retrieved, relevant, k))
                        recalls.append(recall_at_k(retrieved, relevant, k))

                    if precisions:
                        mode_results[k] = {
                            "precision": sum(precisions) / len(precisions),
                            "recall": sum(recalls) / len(recalls),
                            "num_queries": len(precisions),
                        }

                if mode_results:
                    method_results[mode_display[mode]] = mode_results

            if method_results:
                results[method] = method_results

        # Save results to session state so checkboxes don't wipe them
        st.session_state.eval_results = results

# ══════════════════════════════════════════
# Display Results (outside button block so it persists)
# ══════════════════════════════════════════
if st.session_state.get("eval_results"):
    results = st.session_state.eval_results
    # Tables — one per method, one per mode
    for method, method_results in results.items():
        st.markdown(f"### {method.capitalize()} Search")
        for mode_name, mode_results in method_results.items():
            st.markdown(f"**{mode_name} Retrieval**")
            rows = []
            for k, metrics in sorted(mode_results.items()):
                rows.append({
                    "K": k,
                    "Precision@K": round(metrics["precision"], 4),
                    "Recall@K": round(metrics["recall"], 4),
                    "Num Queries": metrics["num_queries"],
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, width='stretch', hide_index=True)

    # Charts — one line per search method
    st.subheader("Metric Charts")

    # Checkbox to select metrics
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        show_precision = st.checkbox("Show Precision@K Chart", value=True)
        show_recall = st.checkbox("Show Recall@K Chart", value=True)

    # Flatten results into chart rows keyed by (Method, K)
    # Average across modes for a single per-method line
    chart_rows = []
    for method, method_results in results.items():
        # Aggregate across all modes for this method
        k_agg = {}
        for mode_name, mode_results in method_results.items():
            for k, metrics in mode_results.items():
                if k not in k_agg:
                    k_agg[k] = {"precision": [], "recall": []}
                k_agg[k]["precision"].append(metrics["precision"])
                k_agg[k]["recall"].append(metrics["recall"])
        for k, agg in sorted(k_agg.items()):
            chart_rows.append({
                "K": k,
                "Precision": sum(agg["precision"]) / len(agg["precision"]),
                "Recall": sum(agg["recall"]) / len(agg["recall"]),
                "Method": method,
            })

    if chart_rows:
        chart_df = pd.DataFrame(chart_rows)

        if show_precision and show_recall:
            col_p, col_r = st.columns(2)
            with col_p:
                st.markdown("**Precision@K by Search Method**")
                pivot_p = chart_df.pivot(index="K", columns="Method", values="Precision")
                st.line_chart(pivot_p)
            with col_r:
                st.markdown("**Recall@K by Search Method**")
                pivot_r = chart_df.pivot(index="K", columns="Method", values="Recall")
                st.line_chart(pivot_r)
        elif show_precision:
            st.markdown("**Precision@K by Search Method**")
            pivot_p = chart_df.pivot(index="K", columns="Method", values="Precision")
            st.line_chart(pivot_p)
        elif show_recall:
            st.markdown("**Recall@K by Search Method**")
            pivot_r = chart_df.pivot(index="K", columns="Method", values="Recall")
            st.line_chart(pivot_r)
        else:
            st.info("Select a metric above to display the corresponding chart.")

    # Per-query breakdown
    st.subheader("Per-Query Breakdown")
    with st.expander("Show per-query results"):
        for label in st.session_state.eval_labels:
            n_relevant = len(label.get("relevant_segment_ids", []))
            if n_relevant > 0:
                st.write(
                    f"**{label['query_id']}** [{label['retrieval_mode']}]: "
                    f"\"{label['query_text']}\" — {n_relevant} relevant segments"
                )

# ══════════════════════════════════════════
# Step 3: Save Labels to Database
# ══════════════════════════════════════════
st.divider()
st.subheader("Step 3: Save Labels to Database (Optional)")
st.caption("Save your evaluation labels to Supabase for reproducibility.")

if st.session_state.eval_labels:
    if st.button("Save Labels to Supabase"):
        try:
            count = db.insert_eval_labels(st.session_state.eval_labels)
            st.success(f"{count} labels saved to database.")
        except Exception as e:
            st.error(f"Failed to save labels: {e}")

    # Download as JSON
    labels_json = json.dumps(st.session_state.eval_labels, indent=2)
    st.download_button(
        "Download Labels as JSON",
        data=labels_json,
        file_name="relevance_labels.json",
        mime="application/json",
    )
