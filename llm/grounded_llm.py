"""
Grounded LLM interface for all three analysis modules.

Uses Groq API (free tier) with Llama 3.3 70B for grounded analysis.
All outputs cite source segments — no hallucination allowed.

Features:
    - Streaming output (token-by-token display in Streamlit)
    - Automatic fallback to smaller model on rate limit
    - System prompt enforcing grounding + ethics constraints

Research basis:
    - RAG (Lewis et al. 2020): retrieve then generate with citations
    - Chain-of-Thought (Wei et al. 2022): step-by-step reasoning
    - Grounded generation: constrain output to retrieved evidence
"""

import config
from llm.prompts import (
    SYSTEM_PROMPT,
    SUMMARIZATION_PROMPT,
    QA_PROMPT,
    ANALYZER_PROMPT,
    format_segments_for_prompt,
)
from typing import List, Optional, Generator
import httpx


def _call_groq_llm(
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.2,
    model: Optional[str] = None,
) -> str:
    """
    Call Groq API with system + user prompt. Returns full response text.
    Automatically falls back to smaller model on rate limit (429).
    """
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set.")

    use_model = model or config.LLM_MODEL

    try:
        return _groq_request(prompt, max_tokens, temperature, use_model)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429 and use_model != config.LLM_MODEL_FALLBACK:
            # Rate limited — try fallback model
            return _groq_request(prompt, max_tokens, temperature, config.LLM_MODEL_FALLBACK)
        raise


def _groq_request(prompt: str, max_tokens: int, temperature: float, model: str) -> str:
    """Execute a single Groq API request."""
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _stream_groq_llm(
    prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.2,
) -> Generator[str, None, None]:
    """
    Stream tokens from Groq API for real-time display.
    Yields text chunks as they arrive.
    """
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set.")

    with httpx.stream(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        },
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    import json
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except Exception:
                    continue


def summarize(segments: List[dict], interview_id: str = "", stream: bool = False):
    """
    Generate a grounded summary of clinical interview segments.

    Args:
        segments: List of segment dicts
        interview_id: For context (optional)
        stream: If True, returns a generator of text chunks

    Returns:
        str (if stream=False) or Generator[str] (if stream=True)
    """
    formatted = format_segments_for_prompt(segments)
    prompt = SUMMARIZATION_PROMPT.format(segments=formatted)
    if stream:
        return _stream_groq_llm(prompt, max_tokens=3000)
    return _call_groq_llm(prompt, max_tokens=3000)


def answer_question(query: str, segments: List[dict], stream: bool = False):
    """
    Answer a question using retrieved segments as context.

    Args:
        query: User's question
        segments: Retrieved segments
        stream: If True, returns a generator of text chunks
    """
    formatted = format_segments_for_prompt(segments)
    prompt = QA_PROMPT.format(question=query, segments=formatted)
    if stream:
        return _stream_groq_llm(prompt, max_tokens=2000)
    return _call_groq_llm(prompt, max_tokens=2000)


def analyze_interview(segments: List[dict], stream: bool = False):
    """
    Produce structured analysis: timeline, symptoms, meds, follow-ups, gaps.

    Args:
        segments: All segments for the interview
        stream: If True, returns a generator of text chunks
    """
    formatted = format_segments_for_prompt(segments)
    prompt = ANALYZER_PROMPT.format(segments=formatted)
    if stream:
        return _stream_groq_llm(prompt, max_tokens=4000)
    return _call_groq_llm(prompt, max_tokens=4000)
