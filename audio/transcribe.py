"""
Audio transcription using Groq's Whisper API (free tier).

Input:  Audio file path
Output: List of dicts: [{start_ms, end_ms, text}]

Free tier limits:
    - 14,400 audio-seconds/day (~4 hours)
    - 25 MB max file size per request
    - Model: whisper-large-v3

Owner: Josh (M2) — validate and improve this scaffold.
"""

import config
from typing import List, Dict
import httpx
import os


class WhisperTranscriber:
    """Wraps Groq's Whisper API for speech-to-text transcription."""

    def __init__(self):
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set. Get one at https://console.groq.com/keys"
            )
        self.api_key = config.GROQ_API_KEY
        self.base_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def transcribe(self, audio_path: str, language: str = "en") -> List[Dict]:
        """
        Transcribe an audio file using Groq Whisper API.

        Args:
            audio_path: Path to audio file
            language: Language code (default "en")

        Returns:
            List of dicts, each with:
                - start_ms (int): Segment start in milliseconds
                - end_ms (int): Segment end in milliseconds
                - text (str): Transcribed text
        """
        # Check file size
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb > config.MAX_AUDIO_SIZE_MB:
            raise ValueError(
                f"Audio file is {file_size_mb:.1f} MB — exceeds Groq's "
                f"{config.MAX_AUDIO_SIZE_MB} MB limit. Split the file first."
            )

        print(f"🔄 Transcribing: {audio_path} ({file_size_mb:.1f} MB)")

        with open(audio_path, "rb") as f:
            response = httpx.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (os.path.basename(audio_path), f)},
                data={
                    "model": config.WHISPER_MODEL,
                    "language": language,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "segment",
                },
                timeout=120.0,
            )

        response.raise_for_status()
        result = response.json()

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start_ms": int(seg["start"] * 1000),
                "end_ms": int(seg["end"] * 1000),
                "text": seg["text"].strip(),
            })

        print(f"✅ Transcription complete: {len(segments)} segments.")
        return segments

    def transcribe_to_text(self, audio_path: str, language: str = "en") -> str:
        """Simple transcription — returns full text only (no timestamps)."""
        segments = self.transcribe(audio_path, language)
        return " ".join(seg["text"] for seg in segments)
