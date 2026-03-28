"""
Speaker diarization module — detects who spoke when in audio recordings.

Supports two backends (selected automatically via config.DIARIZATION_PRIMARY):
    1. Pyannote (local with GPU): Acoustic diarization using pyannote.audio 3.1.1
    2. AssemblyAI (cloud API): Acoustic diarization via AssemblyAI's speaker labels

Pyannote compatibility (tested March 2026):
    - pyannote.audio==3.1.1, torch==2.4.0, torchaudio==2.4.0
    - huggingface_hub<1.0, transformers>=4.38,<5.0
    - numpy==1.26.4 with np.NaN = np.nan polyfill

Research basis:
    - Bredin et al. 2023: pyannote.audio speaker diarization pipeline
    - DiarizationLM (Wang et al. 2024): LLM post-processing for label correction
"""

import config
from typing import List, Dict, Optional
import time


class SpeakerDiarizer:
    """
    Unified speaker diarization interface.
    Automatically selects the best available backend based on environment.
    """

    def __init__(self, backend: Optional[str] = None):
        """
        Initialize diarizer with specified or auto-detected backend.

        Args:
            backend: "pyannote", "assemblyai", or None (auto-detect from config)
        """
        self.backend = backend or config.DIARIZATION_PRIMARY
        self._pipeline = None  # Lazy-loaded for pyannote

        if self.backend == "pyannote":
            self._init_pyannote()
        elif self.backend == "assemblyai":
            if not config.ASSEMBLYAI_API_KEY:
                raise ValueError("ASSEMBLYAI_API_KEY not set for AssemblyAI backend.")
        else:
            raise ValueError(f"Unknown diarization backend: {self.backend}")

    def _init_pyannote(self):
        """Initialize pyannote.audio pipeline with tested dependency polyfills."""
        import numpy as np
        np.NaN = np.nan
        np.NAN = np.nan

        import torch
        from pyannote.audio import Pipeline
        from huggingface_hub import login

        if not config.HF_TOKEN:
            raise ValueError(
                "HF_TOKEN not set. Get one at https://huggingface.co/settings/tokens "
                "and accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        login(token=config.HF_TOKEN, add_to_git_credential=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pyannote] Loading diarization model on {self.device}...")

        self._pipeline = Pipeline.from_pretrained(
            config.DIARIZATION_MODEL,
        ).to(self.device)

        print("[Pyannote] Diarization pipeline loaded.")

    def diarize(self, audio_path: str) -> Dict:
        """
        Run speaker diarization on an audio file.

        Returns:
            Dict with:
                - segments: List[Dict] with start_ms, end_ms, speaker
                - num_speakers: int
                - speakers: List[str] of unique speaker labels
                - speaker_durations: Dict[str, int] mapping speaker -> total ms
                - elapsed_sec: float
                - backend: str
        """
        start = time.time()

        if self.backend == "pyannote":
            segments = self._diarize_pyannote(audio_path)
        else:
            segments = self._diarize_assemblyai(audio_path)

        elapsed = time.time() - start

        # Compute speaker statistics
        speakers = sorted(set(s["speaker"] for s in segments))
        durations = {}
        for s in segments:
            spk = s["speaker"]
            durations[spk] = durations.get(spk, 0) + (s["end_ms"] - s["start_ms"])

        result = {
            "segments": segments,
            "num_speakers": len(speakers),
            "speakers": speakers,
            "speaker_durations": durations,
            "elapsed_sec": round(elapsed, 1),
            "backend": self.backend,
        }

        print(
            f"[{self.backend}] Diarization complete in {elapsed:.1f}s: "
            f"{len(segments)} segments, {len(speakers)} speakers."
        )
        return result

    def _diarize_pyannote(self, audio_path: str) -> List[Dict]:
        """Run pyannote.audio diarization pipeline."""
        diarization = self._pipeline(audio_path)

        segments = []
        for segment, _track, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start_ms": int(segment.start * 1000),
                "end_ms": int(segment.end * 1000),
                "speaker": speaker,
            })
        return segments

    def _diarize_assemblyai(self, audio_path: str) -> List[Dict]:
        """Run AssemblyAI cloud diarization."""
        import assemblyai as aai

        aai.settings.api_key = config.ASSEMBLYAI_API_KEY

        aai_config = aai.TranscriptionConfig(
            speaker_labels=True,
            speech_models=["universal-3-pro", "universal-2"],
            language_detection=True,
        )

        transcript = aai.Transcriber().transcribe(audio_path, config=aai_config)

        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"AssemblyAI error: {transcript.error}")

        segments = []
        for utt in transcript.utterances:
            segments.append({
                "start_ms": utt.start,
                "end_ms": utt.end,
                "speaker": utt.speaker,
                "text": utt.text.strip(),  # AssemblyAI includes text
            })
        return segments

    @staticmethod
    def get_unique_speakers(segments: List[Dict]) -> List[str]:
        """Extract sorted unique speaker labels from diarization segments."""
        return sorted(set(s["speaker"] for s in segments))
