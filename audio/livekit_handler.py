"""
LiveKit integration for real-time clinical interviews.

Each participant gets their own audio track — this means NO diarization is
needed for live mode. LiveKit provides "perfect speaker attribution."

Free tier: 50 participant-minutes/month on LiveKit Cloud.

Owner: Javier (M1) — implements in Phase 4.
"""

import config
from database.models import Segment
from typing import List, Optional


class LiveKitHandler:
    """Manages LiveKit rooms for real-time clinical interviews."""

    def __init__(self):
        # Lazy import — only needed when live mode is used
        from livekit import api as livekit_api

        if not config.LIVEKIT_URL or not config.LIVEKIT_API_KEY:
            raise ValueError(
                "LiveKit credentials not set in .env. "
                "Get them at https://cloud.livekit.io → Settings → Keys"
            )

        self.lk_api = livekit_api.LiveKitAPI(
            config.LIVEKIT_URL,
            config.LIVEKIT_API_KEY,
            config.LIVEKIT_API_SECRET,
        )
        print("LiveKit handler initialized.")

    def create_room(self, room_name: str) -> dict:
        """Create a LiveKit room for the interview session."""
        # TODO: Implement in Phase 4
        raise NotImplementedError("LiveKit room creation — Phase 4")

    def generate_token(
        self, room_name: str, participant_name: str, role: str = "CLINICIAN"
    ) -> str:
        """Generate a JWT token for a participant to join the room."""
        # TODO: Implement in Phase 4
        raise NotImplementedError("LiveKit token generation — Phase 4")

    async def process_track(
        self,
        track_audio_path: str,
        interview_id: str,
        speaker_role: str,
    ) -> List[Segment]:
        """
        Process a single participant's audio track.
        No diarization needed — LiveKit provides clean per-speaker audio.
        """
        # TODO: Implement in Phase 4
        raise NotImplementedError("LiveKit track processing — Phase 4")
