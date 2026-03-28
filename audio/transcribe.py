"""
Audio transcription using Groq's Whisper API (free tier).

Supports:
    - Single file transcription (under 24 MB)
    - Auto-chunking for large files (splits via ffmpeg, reassembles with correct offsets)
    - Voice profile transcription (record → transcribe → return text for patient profile)

Free tier limits:
    - 14,400 audio-seconds/day (~4 hours)
    - 25 MB max file size per request
    - Model: whisper-large-v3
"""

import config
from typing import List, Dict
import httpx
import os
import math
import tempfile


class WhisperTranscriber:
    """Wraps Groq's Whisper API for speech-to-text with auto-chunking."""

    def __init__(self):
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Get one at https://console.groq.com/keys")
        self.api_key = config.GROQ_API_KEY
        self.base_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        progress_callback=None,
    ) -> List[Dict]:
        """
        Transcribe an audio file. Auto-chunks files over 24 MB.

        Args:
            audio_path: Path to audio file
            language: Language code (default "en")
            progress_callback: Optional callable(message: str) for status updates

        Returns:
            List of dicts with start_ms, end_ms, text
        """
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

        def _log(msg):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        if file_size_mb <= config.GROQ_CHUNK_LIMIT_MB:
            _log(f"Transcribing ({file_size_mb:.1f} MB)...")
            return self._api_call(audio_path, language)
        else:
            return self._chunked_transcribe(audio_path, language, _log)

    def _chunked_transcribe(
        self, audio_path: str, language: str, log_fn
    ) -> List[Dict]:
        """Split large files via ffmpeg and transcribe each chunk."""
        import subprocess

        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

        # Get total duration via ffprobe
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True,
        )
        total_duration = float(result.stdout.strip())

        num_chunks = math.ceil(file_size_mb / config.GROQ_CHUNK_LIMIT_MB)
        chunk_duration = total_duration / num_chunks

        log_fn(f"File is {file_size_mb:.1f} MB — splitting into {num_chunks} chunks...")

        all_segments = []
        for i in range(num_chunks):
            start_sec = i * chunk_duration
            chunk_path = os.path.join(tempfile.gettempdir(), f"chunk_{i:02d}.wav")

            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start_sec),
                 "-t", str(chunk_duration), "-ar", "16000", "-ac", "1", chunk_path],
                capture_output=True,
            )

            chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            log_fn(f"  Chunk {i+1}/{num_chunks} ({chunk_size_mb:.1f} MB)...")

            try:
                chunk_segments = self._api_call(chunk_path, language)
                offset_ms = int(start_sec * 1000)
                for seg in chunk_segments:
                    seg["start_ms"] += offset_ms
                    seg["end_ms"] += offset_ms
                all_segments.extend(chunk_segments)
            finally:
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)

        log_fn(f"Transcription complete — {len(all_segments)} segments from {num_chunks} chunks.")
        return all_segments

    def _api_call(self, audio_path: str, language: str) -> List[Dict]:
        """Single Groq Whisper API call."""
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
        return segments

    def transcribe_to_text(self, audio_path: str, language: str = "en") -> str:
        """Simple transcription — returns full text only (for voice profile input)."""
        segments = self.transcribe(audio_path, language)
        return " ".join(seg["text"] for seg in segments)
