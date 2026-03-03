"""
Speaker diarization using pyannote.audio.

Input:  Audio file path (.wav recommended)
Output: List of dicts: [{start_ms, end_ms, speaker}]

Requirements:
    - HF_TOKEN must be set in .env
    - Must accept model agreement at:
      https://huggingface.co/pyannote/speaker-diarization-3.1
      https://huggingface.co/pyannote/segmentation-3.0
    - GPU strongly recommended (CPU works but is very slow)
    - Best run in Google Colab with GPU runtime

Owner: Josh (M2) — validate and improve this scaffold.
"""

import config
from typing import List, Dict


class SpeakerDiarizer:
    """Wraps pyannote.audio for speaker diarization of clinical interviews."""

    def __init__(self):
        # Lazy imports — pyannote + torch are heavy
        import torch
        from pyannote.audio import Pipeline

        if not config.HF_TOKEN:
            raise ValueError(
                "HF_TOKEN not set. Get one at https://huggingface.co/settings/tokens "
                "and accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔄 Loading diarization model on {self.device}...")

        self.pipeline = Pipeline.from_pretrained(
            config.DIARIZATION_MODEL,
            use_auth_token=config.HF_TOKEN,
        ).to(self.device)

        print("✅ Diarization pipeline loaded.")

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Run speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file (.wav recommended)

        Returns:
            List of dicts, each with:
                - start_ms (int): Segment start in milliseconds
                - end_ms (int): Segment end in milliseconds
                - speaker (str): Speaker label, e.g. "SPEAKER_00"
        """
        print(f"🔄 Diarizing: {audio_path}")
        diarization = self.pipeline(audio_path)

        segments = []
        for segment, _track, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start_ms": int(segment.start * 1000),
                "end_ms": int(segment.end * 1000),
                "speaker": speaker,
            })

        print(f"✅ Diarization complete: {len(segments)} segments, "
              f"{len(set(s['speaker'] for s in segments))} speakers detected.")
        return segments

    def get_unique_speakers(self, segments: List[Dict]) -> List[str]:
        """Extract unique speaker labels from diarization output."""
        return sorted(set(s["speaker"] for s in segments))
