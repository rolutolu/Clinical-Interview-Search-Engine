"""
Align Whisper transcript segments with diarization segments.

Uses maximum temporal overlap to assign a speaker identity to each
transcript chunk. Includes confidence scoring based on overlap quality.

Input:
    diarization_segments: [{start_ms, end_ms, speaker}]       from diarize.py
    transcription_segments: [{start_ms, end_ms, text}]        from transcribe.py

Output:
    List[Segment] ready for database insertion, with alignment confidence scores.

Research basis:
    - DiarizationLM (Wang et al. 2024): temporal alignment between ASR and diarization
    - Standard overlap-based alignment used in NIST RT evaluations
"""

from database.models import Segment
from typing import List, Dict, Tuple
import uuid


def align_segments(
    diarization_segments: List[Dict],
    transcription_segments: List[Dict],
    interview_id: str,
    source_mode: str = "offline",
) -> Tuple[List[Segment], Dict]:
    """
    For each transcription segment, find the diarization segment with
    maximum temporal overlap and assign that speaker label.

    Args:
        diarization_segments: Output from SpeakerDiarizer.diarize()["segments"]
        transcription_segments: Output from WhisperTranscriber.transcribe()
        interview_id: FK to interviews table
        source_mode: "offline" or "live"

    Returns:
        Tuple of:
            - List[Segment] with speaker_raw set
            - Dict with alignment quality metrics:
                - avg_confidence: mean overlap confidence (0-1)
                - low_confidence_count: segments with <0.3 confidence
                - unmatched_count: segments with zero overlap
    """
    if not diarization_segments:
        raise ValueError("No diarization segments — diarization may have failed.")
    if not transcription_segments:
        raise ValueError("No transcription segments — transcription may have failed.")

    # Compute default speaker (longest total duration)
    speaker_durations: Dict[str, int] = {}
    for d in diarization_segments:
        spk = d["speaker"]
        speaker_durations[spk] = speaker_durations.get(spk, 0) + (d["end_ms"] - d["start_ms"])
    default_speaker = max(speaker_durations, key=speaker_durations.get)

    aligned = []
    confidences = []
    unmatched = 0

    for t_seg in transcription_segments:
        t_start = t_seg["start_ms"]
        t_end = t_seg["end_ms"]
        t_duration = max(t_end - t_start, 1)  # Avoid division by zero

        best_speaker = default_speaker
        max_overlap = 0

        for d_seg in diarization_segments:
            overlap = max(0, min(t_end, d_seg["end_ms"]) - max(t_start, d_seg["start_ms"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = d_seg["speaker"]

        # Confidence = fraction of transcript segment covered by best diarization match
        confidence = max_overlap / t_duration if t_duration > 0 else 0.0
        confidences.append(confidence)

        if max_overlap == 0:
            unmatched += 1

        aligned.append(
            Segment(
                interview_id=interview_id,
                segment_id=str(uuid.uuid4())[:8],
                start_ms=t_start,
                end_ms=t_end,
                speaker_raw=best_speaker,
                speaker_role="",  # Mapped later
                text=t_seg["text"],
                source_mode=source_mode,
            )
        )

    # Compute alignment quality metrics
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    low_confidence = sum(1 for c in confidences if c < 0.3)

    metrics = {
        "avg_confidence": round(avg_confidence, 3),
        "low_confidence_count": low_confidence,
        "unmatched_count": unmatched,
        "total_segments": len(aligned),
    }

    print(
        f"Alignment complete: {len(aligned)} segments, "
        f"avg confidence: {avg_confidence:.2f}, "
        f"{low_confidence} low-confidence, {unmatched} unmatched."
    )
    return aligned, metrics


def apply_speaker_map(
    segments: List[Segment], speaker_map: Dict[str, str]
) -> List[Segment]:
    """
    Apply a speaker role mapping to aligned segments.

    Args:
        segments: Aligned segments with speaker_raw set
        speaker_map: e.g. {"SPEAKER_00": "PATIENT", "SPEAKER_01": "CLINICIAN"}

    Returns:
        Same segments with speaker_role updated
    """
    for seg in segments:
        seg.speaker_role = speaker_map.get(seg.speaker_raw, "UNKNOWN")
    return segments
