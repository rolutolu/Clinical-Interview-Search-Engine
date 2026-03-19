"""
Speaker diarization using pyannote.audio.

Input:  Audio file path (.wav recommended)
Output: List of dicts: [{start_ms, end_ms, speaker}]

Working combination (tested in Colab March 2026):
    - pyannote.audio==3.1.1
    - torch==2.4.0
    - torchaudio==2.4.0
    - huggingface_hub<1.0
    - transformers>=4.38,<5.0
    - numpy==1.26.4 with np.NaN = np.nan hack

Owner: Josh (M2) — validate and improve this scaffold.
"""

import config
from typing import List, Dict


class SpeakerDiarizer:
    """Wraps pyannote.audio for speaker diarization of clinical interviews."""

    def __init__(self):
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

        # Authenticate via login() — avoids keyword argument issues
        login(token=config.HF_TOKEN, add_to_git_credential=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading diarization model on {self.device}...")

        # No token= arg needed — login() handles auth
        self.pipeline = Pipeline.from_pretrained(
            config.DIARIZATION_MODEL,
        ).to(self.device)

        print("Diarization pipeline loaded.")

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Run speaker diarization on an audio file.

        Returns:
            List of dicts with start_ms, end_ms, speaker
        """
        import time

        print(f"Diarizing: {audio_path}")
        start_time = time.time()

        diarization = self.pipeline(audio_path)

        elapsed = time.time() - start_time
        segments = []
        for segment, _track, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start_ms": int(segment.start * 1000),
                "end_ms": int(segment.end * 1000),
                "speaker": speaker,
            })

        print(f"Diarization complete in {elapsed:.1f}s: {len(segments)} segments, "
              f"{len(set(s['speaker'] for s in segments))} speakers.")
        return segments

    def get_unique_speakers(self, segments: List[Dict]) -> List[str]:
        return sorted(set(s["speaker"] for s in segments))
