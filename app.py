"""
Clinical Interview IR System — Main Application

Intelligent speaker-aware retrieval for clinical interview transcripts.
Supports offline (Pyannote/AssemblyAI) and live (LiveKit) pipelines.

Run with: python -m streamlit run app.py
"""

import streamlit as st
import config

st.set_page_config(
    page_title="ClinIR — Clinical Interview Retrieval",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════
# Global CSS — Professional Dark Medical Theme
# ══════════════════════════════════════════
st.markdown(f"""
<style>
    /* ── Typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* ── Hide Streamlit branding ── */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ── Brand Header ── */
    .brand-header {{
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, {config.BRAND_PRIMARY} 0%, {config.BRAND_SECONDARY} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }}
    .brand-subtitle {{
        font-size: 1.05rem;
        color: {config.BRAND_MUTED};
        margin-bottom: 2rem;
        font-weight: 300;
    }}

    /* ── Cards ── */
    .metric-card {{
        background: {config.BRAND_CARD};
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.4rem;
        text-align: center;
        transition: border-color 0.2s ease;
    }}
    .metric-card:hover {{
        border-color: {config.BRAND_PRIMARY};
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {config.BRAND_PRIMARY};
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: {config.BRAND_MUTED};
        margin-top: 0.3rem;
    }}

    /* ── Feature Cards ── */
    .feature-card {{
        background: {config.BRAND_CARD};
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.6rem;
        height: 100%;
        transition: all 0.2s ease;
    }}
    .feature-card:hover {{
        border-color: {config.BRAND_PRIMARY};
        transform: translateY(-2px);
    }}
    .feature-icon {{
        font-size: 2rem;
        margin-bottom: 0.8rem;
    }}
    .feature-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {config.BRAND_TEXT};
        margin-bottom: 0.5rem;
    }}
    .feature-desc {{
        font-size: 0.9rem;
        color: {config.BRAND_MUTED};
        line-height: 1.5;
    }}

    /* ── Status Badges ── */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }}
    .status-ok {{
        background: rgba(0, 212, 170, 0.15);
        color: {config.BRAND_SECONDARY};
        border: 1px solid rgba(0, 212, 170, 0.3);
    }}
    .status-warn {{
        background: rgba(255, 107, 107, 0.15);
        color: {config.BRAND_ACCENT};
        border: 1px solid rgba(255, 107, 107, 0.3);
    }}
    .status-info {{
        background: rgba(108, 99, 255, 0.15);
        color: {config.BRAND_PRIMARY};
        border: 1px solid rgba(108, 99, 255, 0.3);
    }}

    /* ── Ethics Banner ── */
    .ethics-banner {{
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
        color: #FFC107;
        font-size: 0.88rem;
    }}

    /* ── Pipeline Diagram ── */
    .pipeline-box {{
        background: {config.BRAND_CARD};
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1.2rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: {config.BRAND_MUTED};
        line-height: 1.8;
    }}
    .pipeline-box .highlight {{
        color: {config.BRAND_PRIMARY};
        font-weight: 600;
    }}
    .pipeline-box .accent {{
        color: {config.BRAND_SECONDARY};
        font-weight: 600;
    }}

    /* ── Speaker Colors (used globally) ── */
    .speaker-patient {{
        background-color: {config.PATIENT_BG};
        border-left: 4px solid {config.PATIENT_COLOR};
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
    }}
    .speaker-clinician {{
        background-color: {config.CLINICIAN_BG};
        border-left: 4px solid {config.CLINICIAN_COLOR};
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {config.BRAND_CARD};
        border-right: 1px solid #30363D;
    }}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] small {{
        color: {config.BRAND_TEXT} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# Sidebar — System Status
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f'<div style="text-align:center;padding:1rem 0;">'
        f'<span style="font-size:1.6rem;font-weight:700;'
        f'background:linear-gradient(135deg,{config.BRAND_PRIMARY},{config.BRAND_SECONDARY});'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        f'ClinIR</span><br>'
        f'<span style="font-size:0.75rem;color:{config.BRAND_MUTED};">'
        f'Clinical Interview Retrieval</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("CP423 — Text Retrieval & Search Engines")

    st.divider()

    # Environment
    env_label = "Local (Full Stack)" if config.IS_LOCAL else "Cloud (API Only)"
    env_class = "status-ok" if config.IS_LOCAL else "status-info"
    st.markdown(f'<span class="{env_class} status-badge">{env_label}</span>', unsafe_allow_html=True)

    st.divider()

    # Connection statuses
    st.markdown(f"<small style='color:{config.BRAND_MUTED};font-weight:600;'>CONNECTIONS</small>", unsafe_allow_html=True)

    services = [
        ("Supabase", bool(config.SUPABASE_URL and config.SUPABASE_KEY)),
        ("Groq API", bool(config.GROQ_API_KEY)),
        ("HuggingFace", bool(config.HF_TOKEN)),
        ("AssemblyAI", bool(config.ASSEMBLYAI_API_KEY)),
        ("LiveKit", bool(config.LIVEKIT_URL and config.LIVEKIT_API_KEY)),
    ]

    for name, ok in services:
        badge = "status-ok" if ok else "status-warn"
        icon = "●" if ok else "○"
        st.markdown(f'<span class="{badge} status-badge">{icon} {name}</span>', unsafe_allow_html=True)

    st.divider()

    # Capabilities
    st.markdown(f"<small style='color:{config.BRAND_MUTED};font-weight:600;'>CAPABILITIES</small>", unsafe_allow_html=True)

    caps = [
        ("Pyannote Diarization", config.ENV["has_pyannote"]),
        ("Semantic Search", config.ENV["has_embeddings"]),
        ("GPU Acceleration", config.ENV["has_gpu"]),
        ("Cross-Encoder Reranking", bool(config.ENV.get("has_embeddings") and getattr(config, "RERANKER_ENABLED", False))),
    ]

    for name, available in caps:
        badge = "status-ok" if available else "status-info"
        label = "Active" if available else "Cloud N/A"
        st.markdown(f'<span class="{badge} status-badge">{name}: {label}</span>', unsafe_allow_html=True)

    st.divider()

    # Config summary
    st.markdown(f"<small style='color:{config.BRAND_MUTED};font-weight:600;'>CONFIGURATION</small>", unsafe_allow_html=True)
    st.caption(f"LLM: {config.LLM_MODEL.split('-')[0].title()} 70B")
    st.caption(f"Embeddings: {config.EMBEDDING_MODEL.split('/')[-1]}")
    st.caption(f"Diarization: pyannote (primary)")
    st.caption(f"Default K: {config.DEFAULT_K}")
    st.caption(f"Search: hybrid (rrf)")

# ══════════════════════════════════════════
# Main Page — Dashboard
# ══════════════════════════════════════════
st.markdown('<p class="brand-header">ClinIR</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Intelligent Speaker-Aware Retrieval for Clinical Interview Transcripts</p>', unsafe_allow_html=True)

# Ethics banner
st.markdown(f'<div class="ethics-banner">{config.ETHICS_DISCLAIMER}</div>', unsafe_allow_html=True)

# ── Metrics Row ──
try:
    from database.supabase_client import SupabaseClient
    db = SupabaseClient()
    interviews = db.list_interviews()
    total_interviews = len(interviews)
    total_segments = sum(db.get_segment_count(iv["interview_id"]) for iv in interviews) if interviews else 0
except Exception:
    total_interviews = 0
    total_segments = 0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_interviews}</div><div class="metric-label">Interviews</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{total_segments}</div><div class="metric-label">Segments Indexed</div></div>', unsafe_allow_html=True)
with m3:
    methods = len(config.SEARCH_METHODS)
    st.markdown(f'<div class="metric-card"><div class="metric-value">{methods}</div><div class="metric-label">Search Methods</div></div>', unsafe_allow_html=True)
with m4:
    modules = 3
    st.markdown(f'<div class="metric-card"><div class="metric-value">{modules}</div><div class="metric-label">Analysis Modules</div></div>', unsafe_allow_html=True)

st.write("")

# ── Feature Cards ──
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="feature-card">'
        f'<div class="feature-icon">📤</div>'
        f'<div class="feature-title">Upload & Process</div>'
        f'<div class="feature-desc">Upload clinical audio. Pyannote diarization + Whisper transcription + speaker-aware indexing.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Upload_Offline.py", label="Open Upload", use_container_width=True)

with c2:
    st.markdown(
        f'<div class="feature-card">'
        f'<div class="feature-icon">🎤</div>'
        f'<div class="feature-title">Live Interview</div>'
        f'<div class="feature-desc">Real-time interviews via LiveKit. Perfect speaker attribution with zero diarization error.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Live_Interview.py", label="Open Live", use_container_width=True)

with c3:
    st.markdown(
        f'<div class="feature-card">'
        f'<div class="feature-icon">🔍</div>'
        f'<div class="feature-title">Query & Analysis</div>'
        f'<div class="feature-desc">Speaker-aware search with grounded LLM summarization, QA, and structured analysis.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Query_Analysis.py", label="Open Query", use_container_width=True)

with c4:
    st.markdown(
        f'<div class="feature-card">'
        f'<div class="feature-icon">📊</div>'
        f'<div class="feature-title">Evaluation</div>'
        f'<div class="feature-desc">Precision@K, Recall@K, nDCG, MAP, MRR. LLM-as-Judge relevance labels.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_Evaluation.py", label="Open Evaluation", use_container_width=True)

st.write("")

# ── Architecture Diagram ──
st.divider()
st.markdown(f"<small style='color:{config.BRAND_MUTED};font-weight:600;letter-spacing:1px;'>SYSTEM ARCHITECTURE</small>", unsafe_allow_html=True)
st.write("")

col_offline, col_live = st.columns(2)

with col_offline:
    st.markdown(
        f'<div class="pipeline-box">'
        f'<span class="highlight">OFFLINE PIPELINE</span><br><br>'
        f'Audio File<br>'
        f'  → <span class="accent">Pyannote</span> Diarization (GPU)<br>'
        f'  → <span class="accent">Whisper</span> Transcription (Groq API)<br>'
        f'  → Temporal Alignment<br>'
        f'  → <span class="accent">MiniLM</span> Embeddings (384-dim)<br>'
        f'  → <span class="highlight">Supabase</span> (pgvector + FTS)<br>'
        f'  → Hybrid Retrieval (BM25 + Cosine)<br>'
        f'  → <span class="accent">Llama 3.3 70B</span> Grounded Analysis'
        f'</div>',
        unsafe_allow_html=True,
    )

with col_live:
    st.markdown(
        f'<div class="pipeline-box">'
        f'<span class="highlight">LIVE PIPELINE</span><br><br>'
        f'<span class="accent">LiveKit</span> Room (WebRTC)<br>'
        f'  → Track 1: Clinician Audio<br>'
        f'  → Track 2: Patient Audio<br>'
        f'  → Per-Track <span class="accent">Whisper</span> Transcription<br>'
        f'  → Perfect Speaker Attribution (DER=0%)<br>'
        f'  → <span class="highlight">Supabase</span> (pgvector + FTS)<br>'
        f'  → Hybrid Retrieval (BM25 + Cosine)<br>'
        f'  → <span class="accent">Llama 3.3 70B</span> Grounded Analysis'
        f'</div>',
        unsafe_allow_html=True,
    )

st.write("")

# ── Tech Stack ──
st.markdown(f"<small style='color:{config.BRAND_MUTED};font-weight:600;letter-spacing:1px;'>TECHNOLOGY STACK</small>", unsafe_allow_html=True)
st.write("")

t1, t2, t3, t4, t5 = st.columns(5)
tech = [
    ("Diarization", "Pyannote 3.1 + AssemblyAI", t1),
    ("Transcription", "Groq Whisper v3", t2),
    ("Retrieval", "pgvector + BM25 Hybrid", t3),
    ("LLM", "Llama 3.3 70B (Groq)", t4),
    ("Live Audio", "LiveKit WebRTC", t5),
]
for label, value, col in tech:
    with col:
        st.markdown(
            f'<div style="text-align:center;padding:0.8rem;background:{config.BRAND_CARD};'
            f'border:1px solid #30363D;border-radius:8px;">'
            f'<div style="font-size:0.75rem;color:{config.BRAND_MUTED};">{label}</div>'
            f'<div style="font-size:0.9rem;font-weight:600;color:{config.BRAND_TEXT};margin-top:0.3rem;">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
