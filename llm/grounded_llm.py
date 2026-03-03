"""
Grounded LLM interface for all three analysis modules.

Uses Groq API (free tier) with Llama 3.3 70B for grounded analysis.
All outputs must cite source segments — no hallucination allowed.

This single file handles summarization, QA, and analysis — Tolu (M3) can
split into separate files if needed, but the interface stays the same.

Owner: Tolu (M3) — refine prompts, improve citation quality.

INTERFACE CONTRACT:
    summarize(segments, interview_id) -> str
    answer_question(query, segments) -> str
    analyze_interview(segments) -> dict
"""

import config
from llm.prompts import (
    SUMMARIZATION_PROMPT,
    QA_PROMPT,
    ANALYZER_PROMPT,
    format_segments_for_prompt,
)
from typing import List
import httpx


def _call_groq_llm(prompt: str, max_tokens: int = 2000) -> str:
    """
    Call Groq API with a prompt. Returns the LLM response text.
    Uses Llama 3.3 70B (free tier: 14,400 tokens/min).
    """
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env")

    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,  # Low temperature for factual grounding
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def summarize(segments: List[dict], interview_id: str = "") -> str:
    """
    Generate a grounded summary of clinical interview segments.

    Args:
        segments: List of segment dicts from retrieval
        interview_id: For context (optional)

    Returns:
        Summary string with [S{id} mm:ss-mm:ss ROLE] citations
    """
    formatted = format_segments_for_prompt(segments)
    prompt = SUMMARIZATION_PROMPT.format(segments=formatted)
    return _call_groq_llm(prompt, max_tokens=2000)


def answer_question(query: str, segments: List[dict]) -> str:
    """
    Answer a question using retrieved segments as context.

    Args:
        query: User's question
        segments: Retrieved segments relevant to the query

    Returns:
        Answer string with segment citations
    """
    formatted = format_segments_for_prompt(segments)
    prompt = QA_PROMPT.format(question=query, segments=formatted)
    return _call_groq_llm(prompt, max_tokens=1500)


def analyze_interview(segments: List[dict]) -> str:
    """
    Produce structured analysis: timeline, symptoms, meds, follow-ups, gaps.

    Args:
        segments: All segments for the interview

    Returns:
        Structured analysis string with segment citations
    """
    formatted = format_segments_for_prompt(segments)
    prompt = ANALYZER_PROMPT.format(segments=formatted)
    return _call_groq_llm(prompt, max_tokens=2500)
