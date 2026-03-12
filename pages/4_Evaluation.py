"""
Page 4: Evaluation Dashboard

Run and display Precision@K and Recall@K metrics for retrieval quality.
Reports: overall, patient-only, clinician-only across multiple K values.
"""

import streamlit as st
import pandas as pd
import config

st.set_page_config(page_title="Evaluation", page_icon="EV", layout="wide")

st.title("Retrieval Evaluation Dashboard")
st.caption("Precision@K and Recall@K metrics — overall, patient-only, and clinician-only.")

# ══════════════════════════════════════════
# Load Eval Labels
# ══════════════════════════════════════════
try:
    from database.supabase_client import SupabaseClient
    db = SupabaseClient()
    eval_labels = db.get_eval_labels()
except Exception as e:
    st.error(f"Could not connect to database: {e}")
    eval_labels = []

# Interview selection
try:
    interviews = db.list_interviews()
except Exception:
    interviews = []

if not interviews:
    st.warning("No interviews found. Upload an interview first.")
    st.stop()

interview_options = {
    f"{iv['title']} ({iv['interview_id']})": iv['interview_id']
    for iv in interviews
}
selected_label = st.selectbox("Select Interview to Evaluate", options=list(interview_options.keys()))
interview_id = interview_options[selected_label]

st.divider()

# ══════════════════════════════════════════
# Evaluation Controls
# ══════════════════════════════════════════
col1, col2 = st.columns(2)
with col1:
    k_values = st.multiselect(
        "K Values to Test",
        options=[1, 2, 3, 5, 10, 15, 20],
        default=config.K_VALUES,
    )
with col2:
    search_method = st.selectbox(
        "Search Method",
        ["hybrid", "semantic", "lexical"],
        key="eval_method",
    )

# ══════════════════════════════════════════
# Run Evaluation
# ══════════════════════════════════════════
if st.button("Run Evaluation", type="primary", use_container_width=True):
    if not eval_labels:
        st.warning(
            "No evaluation labels found in the database. "
            "You need to create relevance labels first. "
            "Upload them to the `eval_labels` table in Supabase."
        )
        st.markdown("""
        **Expected format** (in `eval_labels` table):
        ```json
        {
            "query_id": "q1",
            "query_text": "What symptoms did the patient report?",
            "relevant_segment_ids": ["seg_abc", "seg_def", "seg_ghi"],
            "retrieval_mode": "combined"
        }
        ```
        Create labels for `combined`, `patient`, and `clinician` modes.
        """)
    else:
        with st.spinner("Running evaluation across all modes and K values..."):
            try:
                from evaluation.metrics import run_evaluation
                from retrieval.search import search

                results = run_evaluation(
                    eval_labels=eval_labels,
                    search_fn=lambda query, interview_id, mode, k: search(
                        query=query,
                        interview_id=interview_id,
                        mode=mode,
                        k=k,
                        method=search_method,
                        db=db,
                    ),
                    k_values=k_values,
                    interview_id=interview_id,
                )

                # ── Display Results Table ──
                st.subheader("Results")

                for mode_name, mode_results in results.items():
                    st.markdown(f"### {mode_name.title()} Retrieval")

                    rows = []
                    for k, metrics in sorted(mode_results.items()):
                        rows.append({
                            "K": k,
                            "Precision@K": f"{metrics['precision']:.3f}",
                            "Recall@K": f"{metrics['recall']:.3f}",
                            "Num Queries": metrics["num_queries"],
                        })

                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # ── Charts ──
                st.subheader("Precision@K and Recall@K Charts")

                # Build chart data
                chart_rows = []
                for mode_name, mode_results in results.items():
                    for k, metrics in sorted(mode_results.items()):
                        chart_rows.append({
                            "K": k,
                            "Precision": metrics["precision"],
                            "Recall": metrics["recall"],
                            "Mode": mode_name.title(),
                        })

                if chart_rows:
                    chart_df = pd.DataFrame(chart_rows)

                    col_p, col_r = st.columns(2)

                    with col_p:
                        st.markdown("**Precision@K**")
                        # Pivot for line chart
                        pivot_p = chart_df.pivot(index="K", columns="Mode", values="Precision")
                        st.line_chart(pivot_p)

                    with col_r:
                        st.markdown("**Recall@K**")
                        pivot_r = chart_df.pivot(index="K", columns="Mode", values="Recall")
                        st.line_chart(pivot_r)

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.info("Make sure segments exist and have embeddings before running evaluation.")

# ══════════════════════════════════════════
# Manual Label Upload
# ══════════════════════════════════════════
st.divider()
st.subheader("Upload Evaluation Labels")
st.caption("Upload a JSON file with query-relevance labels for evaluation.")

uploaded_labels = st.file_uploader("Upload relevance_labels.json", type=["json"], key="eval_upload")

if uploaded_labels:
    import json
    try:
        labels = json.load(uploaded_labels)
        st.json(labels[:3])  # Preview first 3
        st.write(f"Loaded {len(labels)} evaluation labels.")

        if st.button("Save Labels to Database"):
            count = db.insert_eval_labels(labels)
            st.success(f"{count} labels saved to database.")
    except Exception as e:
        st.error(f"Failed to parse labels: {e}")
