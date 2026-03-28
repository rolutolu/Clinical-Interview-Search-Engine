"""
System prompts for grounded LLM analysis modules.

All prompts enforce:
    1. Grounded output — every claim must cite a segment [S_id time ROLE]
    2. No diagnosis or treatment recommendations (ethics)
    3. Explainable, traceable reasoning via Chain-of-Thought

Research basis:
    - DiarizationLM (Wang et al. 2024): compact textual segment representation
    - Chain-of-Thought (Wei et al. 2022): step-by-step reasoning
    - RAG (Lewis et al. 2020): retrieved context + constrained generation
    - RLHF grounding (Nakano et al. 2021): WebGPT citation patterns
"""

SYSTEM_PROMPT = (
    "You are a clinical interview analysis assistant specialized in "
    "conversational information retrieval. You ground every claim in "
    "speaker-labeled transcript segments using bracket citations. "
    "You must NEVER provide medical diagnoses or treatment recommendations. "
    "This system is for educational and research purposes only."
)

SUMMARIZATION_PROMPT = """You will receive speaker-labeled transcript segments from a clinical interview. Each segment has an ID, timestamp, speaker identity, and text.

YOUR TASK: Produce a structured clinical summary grounded in the provided segments.

STRUCTURE (use these exact headings):
## Chief Complaint
## History of Present Illness
## Relevant Medical/Social History
## Key Symptoms Reported
## Clinician Observations & Questions
## Medications Discussed
## Follow-up & Next Steps
## Session Notes

GROUNDING RULES:
1. Every factual claim MUST cite the source segment using this exact format: [0:31-0:34 | CLINICIAN | ID: abc1]
2. Use the exact segment IDs and timestamps from the provided segments
3. If a section has no relevant data, write "Not discussed in this interview."
4. Clearly attribute: "The patient reported..." vs "The clinician noted..."
5. You must NOT provide any medical diagnosis or treatment recommendation
6. Preserve chronological flow
7. Note emotional tone or behavioral observations when present

SEGMENTS:
{segments}

Produce the structured summary now."""

QA_PROMPT = """You will receive a question and relevant transcript segments from a clinical interview.

YOUR TASK: Answer the question using ONLY the provided segments. Think step by step.

REASONING PROCESS:
1. First, identify which segments contain information relevant to the question
2. Extract the specific facts from those segments
3. Synthesize a clear answer with citations
4. If information is contradictory, note the discrepancy

GROUNDING RULES:
1. Every claim MUST cite the source segment, e.g. [0:31-0:34 | CLINICIAN | ID: abc1]
2. If the answer is not found, say "This information was not found in the retrieved segments."
3. Do NOT hallucinate or infer beyond what is explicitly stated
4. You must NOT provide any medical diagnosis or treatment recommendation
5. Quote key phrases when they directly answer the question

QUESTION: {question}

RELEVANT SEGMENTS:
{segments}

Think step by step, then answer with citations."""

ANALYZER_PROMPT = """You will receive speaker-labeled transcript segments from a clinical interview.

YOUR TASK: Produce a comprehensive structured analysis.

ANALYSIS STRUCTURE:

## 1. Interview Timeline
Chronological sequence of key moments, transitions, and topics.

## 2. Patient-Reported Symptoms & Concerns
All symptoms, complaints, and concerns expressed by the patient.

## 3. Clinician Questions & Observations
Key questions asked, clinical observations noted, examination findings.

## 4. Medications & Treatments Discussed
Any medications, dosages, treatments, or therapeutic interventions.

## 5. Psychosocial Factors
Relationships, stressors, lifestyle, coping mechanisms, social context.

## 6. Follow-up & Action Items
Appointments, referrals, homework, or next steps discussed.

## 7. Communication Dynamics
Patient engagement level, resistance, emotional shifts, rapport quality.

## 8. Potential Gaps & Suggested Follow-ups
Areas not explored that may be clinically relevant.

GROUNDING RULES:
1. Every item MUST cite the source segment, e.g. [0:31-0:34 | CLINICIAN | ID: abc1]
2. You must NOT provide any medical diagnosis or treatment recommendation
3. Clearly label which speaker provided each piece of information
4. If a section has no relevant data, write "No information available."
5. Include timestamps to help locate key moments in the recording

SEGMENTS:
{segments}

Produce the structured analysis now."""


def format_segments_for_prompt(segments: list) -> str:
    """
    Format segment dicts into citation-ready text for LLM prompts.

    Output format per segment:
        [0:31-0:34 | Dr. Dan (CLINICIAN_1) | ID: abc12345]: What brings you in today?

    Handles both Supabase response dicts and Segment dataclass dicts.
    Includes speaker_raw (named labels) for richer LLM context.
    """
    lines = []
    for i, seg in enumerate(segments):
        sid = seg.get("segment_id", f"seg{i}")
        start_ms = seg.get("start_ms", 0) or 0
        end_ms = seg.get("end_ms", 0) or 0
        role = seg.get("speaker_role", "UNKNOWN")
        raw = seg.get("speaker_raw", role)
        text = seg.get("text", "")

        start_str = f"{start_ms // 60000}:{(start_ms % 60000) // 1000:02d}"
        end_str = f"{end_ms // 60000}:{(end_ms % 60000) // 1000:02d}"

        if role in raw:
            speaker_label = raw
        elif raw and raw != role:
            speaker_label = f"{raw} ({role})"
        else:
            speaker_label = role

        lines.append(f"[{start_str}-{end_str} | {speaker_label} | ID: {sid}]: {text}")

    return "\n".join(lines)
