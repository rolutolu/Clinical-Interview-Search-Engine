"""
Canonical data models for the Clinical Interview IR System.

IMPORTANT: This file defines the shared contract between all team members.
Every module produces or consumes these dataclasses. Do not modify field names
without coordinating with the full team.

Segment is the atomic unit — retrieval, LLM prompts, and evaluation all
reference segments.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import uuid
from datetime import datetime, timezone


@dataclass
class Segment:
    """
    The atomic unit of the entire system.
    Represents one chunk of speech from one speaker in one interview.

    Every retrieval result, every LLM citation, every evaluation label
    references a Segment by its segment_id.
    """

    interview_id: str                    # FK → interviews table
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_ms: int = 0                    # Start time in milliseconds
    end_ms: int = 0                      # End time in milliseconds
    speaker_raw: str = ""                # Raw diarization label: "SPEAKER_0", "SPEAKER_1"
    speaker_role: str = ""               # Mapped role: "PATIENT" or "CLINICIAN"
    text: str = ""                       # Transcribed text content
    source_mode: str = "offline"         # "offline" or "live"
    embedding: Optional[List[float]] = None  # Vector embedding (384-dim for MiniLM)
    keywords: Optional[List[str]] = None     # Extracted medical entities (optional)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization (excludes embedding)."""
        d = asdict(self)
        d.pop("embedding", None)  # Don't serialize large vectors
        return d

    def to_db_dict(self) -> dict:
        """Convert to dict for Supabase insertion (includes embedding)."""
        d = asdict(self)
        if d["keywords"] is None:
            d["keywords"] = []
        if d["embedding"] is None:
            d.pop("embedding")
        return d

    def citation_tag(self) -> str:
        """
        Format: [S{id} {mm:ss}-{mm:ss} {ROLE}]
        Used by LLM modules to cite source segments in their output.
        """
        start_min = self.start_ms // 60000
        start_sec = (self.start_ms % 60000) // 1000
        end_min = self.end_ms // 60000
        end_sec = (self.end_ms % 60000) // 1000
        return f"[S{self.segment_id} {start_min}:{start_sec:02d}-{end_min}:{end_sec:02d} {self.speaker_role}]"

    def time_range_str(self) -> str:
        """Human-readable time range: '0:00 - 0:15'"""
        start_min = self.start_ms // 60000
        start_sec = (self.start_ms % 60000) // 1000
        end_min = self.end_ms // 60000
        end_sec = (self.end_ms % 60000) // 1000
        return f"{start_min}:{start_sec:02d} - {end_min}:{end_sec:02d}"


@dataclass
class Interview:
    """Represents one clinical interview session (offline upload or live room)."""

    interview_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    source_mode: str = "offline"         # "offline" or "live"
    audio_filename: str = ""
    duration_ms: int = 0
    speaker_map: dict = field(default_factory=dict)  # {"SPEAKER_0": "PATIENT", ...}
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_db_dict(self) -> dict:
        return asdict(self)


@dataclass
class PatientProfile:
    """
    Optional patient context submitted before or during an interview.
    Can be entered via text form or voice input (transcribed via Whisper).
    """

    profile_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    interview_id: str = ""
    name: str = ""
    age: Optional[int] = None
    chief_complaint: str = ""
    medical_history: str = ""
    input_method: str = "text"           # "text" or "voice"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_db_dict(self) -> dict:
        return asdict(self)
