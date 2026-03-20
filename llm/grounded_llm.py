"""
Grounded LLM interface for all three analysis modules.

Uses Groq API (free tier) with Llama 3.3 70B for grounded analysis.
All outputs must cite source segments — no hallucination allowed.

Research basis:
    - RAG (Lewis et al. 2020): retrieve then generate with citations
    - Chain-of-Thought (Wei et al. 2022): step-by-step reasoning
    - Grounded generation: constrain output to retrieved evidence
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
    """
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env or Streamlit secrets.")

    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical interview analysis assistant. "
                        "You must ground every claim in the provided transcript segments "
                        "and cite them using bracket notation. "
                        "You must NEVER provide medical diagnoses or treatment recommendations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        },
        timeout=90.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def summarize(segments: List[dict], interview_id: str = "") -> str:
    """Generate a grounded summary of clinical interview segments."""
    formatted = format_segments_for_prompt(segments)
    prompt = SUMMARIZATION_PROMPT.format(segments=formatted)
    return _call_groq_llm(prompt, max_tokens=3000)


def answer_question(query: str, segments: List[dict]) -> str:
    """Answer a question using retrieved segments as context."""
    formatted = format_segments_for_prompt(segments)
    prompt = QA_PROMPT.format(question=query, segments=formatted)
    return _call_groq_llm(prompt, max_tokens=2000)


def analyze_interview(segments: List[dict]) -> str:
    """Produce structured analysis: timeline, symptoms, meds, follow-ups, gaps."""
    formatted = format_segments_for_prompt(segments)
    prompt = ANALYZER_PROMPT.format(segments=formatted)
    return _call_groq_llm(prompt, max_tokens=4000)
