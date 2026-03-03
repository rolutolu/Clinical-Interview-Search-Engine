"""
Align Whisper transcript segments with Pyannote diarization segments.

Uses maximum temporal overlap to assign a speaker identity to each
transcript chunk. This is the critical bridge between diarization
(who spoke when) and transcription (what was said when).

Input:
    diarization_segments: [{start_ms, end_ms, speaker}]       from diarize.py
    transcription_segments: [{start_ms, end_ms, text}]        from transcribe.py

Output:
    List[Segment] ready for database insertion

Owner: Josh (M2) — validate alignment logic, add edge-case handling.
"""

from database.models import Segment
from typing import List, Dict


def align_segments(
    diarization_segments: List[Dict],
    transcription_segments: List[Dict],
    interview_id: str,
    source_mode: str = "offline",
) -> List[Segment]:
    """
    For each transcription segment, find the diarization segment with
    maximum temporal overlap and assign that speaker label.

    Algorithm:
        For each transcript chunk T:
            For each diarization chunk D:
                overlap = max(0, min(T.end, D.end) - max(T.start, D.start))
            Assign speaker from D with largest overlap.

    Args:
        diarization_segments: Output from SpeakerDiarizer.diarize()
        transcription_segments: Output from WhisperTranscriber.transcribe()
        interview_id: FK to interviews table
        source_mode: "offline" or "live"

    Returns:
        List[Segment] with speaker_raw set (speaker_role assigned later via UI)
    """
    if not diarization_segments:
        raise ValueError("No diarization segments — diarization may have failed.")
    if not transcription_segments:
        raise ValueError("No transcription segments — transcription may have failed.")

    # Get the default speaker (most frequent in diarization)
    speaker_counts: Dict[str, int] = {}
    for d in diarization_segments:
        spk = d["speaker"]
        speaker_counts[spk] = speaker_counts.get(spk, 0) + (d["end_ms"] - d["start_ms"])
    default_speaker = max(speaker_counts, key=speaker_counts.get)

    aligned = []

    for t_seg in transcription_segments:
        t_start = t_seg["start_ms"]
        t_end = t_seg["end_ms"]

        best_speaker = default_speaker
        max_overlap = 0

        for d_seg in diarization_segments:
            d_start = d_seg["start_ms"]
            d_end = d_seg["end_ms"]

            # Calculate temporal overlap
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = d_seg["speaker"]

        aligned.append(
            Segment(
                interview_id=interview_id,
                start_ms=t_start,
                end_ms=t_end,
                speaker_raw=best_speaker,
                speaker_role="",  # Mapped later via UI
                text=t_seg["text"],
                source_mode=source_mode,
            )
        )

    print(f"✅ Alignment complete: {len(aligned)} segments aligned.")
    return aligned


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
