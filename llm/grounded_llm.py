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
    Call Groq API with automatic fallback chain:
    1. Primary Groq model (Llama 3.3 70B)
    2. Groq fallback model (Llama 3.1 8B)
    3. Google Gemini (if Groq rate limited)
    """
    if not config.GROQ_API_KEY:
        return _call_gemini(prompt, max_tokens, temperature)

    use_model = model or config.LLM_MODEL

    try:
        return _groq_request(prompt, max_tokens, temperature, use_model)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            # Try smaller Groq model first
            if use_model != config.LLM_MODEL_FALLBACK:
                try:
                    return _groq_request(prompt, max_tokens, temperature, config.LLM_MODEL_FALLBACK)
                except httpx.HTTPStatusError as e2:
                    if e2.response.status_code == 429:
                        return _call_gemini(prompt, max_tokens, temperature)
                    raise
            # Both Groq models rate limited — use Gemini
            return _call_gemini(prompt, max_tokens, temperature)
        raise


def _call_gemini(prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
    """Fallback LLM via Google Gemini API."""
    if not config.GEMINI_API_KEY:
        raise ValueError("Both Groq and Gemini API keys are missing. Cannot generate analysis.")

    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        f"{SYSTEM_PROMPT}\n\n{prompt}",
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return response.text


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
    Stream tokens from Groq API. Falls back to Gemini (non-streaming) on 429.
    """
    if not config.GROQ_API_KEY:
        yield _call_gemini(prompt, max_tokens, temperature)
        return

    try:
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
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            yield _call_gemini(prompt, max_tokens, temperature)
        else:
            raise


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
