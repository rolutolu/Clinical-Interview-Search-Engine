"""
LiveKit integration for real-time clinical interviews.

Each participant gets their own audio track — NO diarization needed.
LiveKit provides "perfect speaker attribution" (DER = 0%).

Components:
    - Room creation via LiveKit Server API
    - JWT token generation for clinician + patient participants
    - Per-track audio capture and Whisper transcription
    - Segment storage with perfect speaker_role labels

Free tier: 50 participant-minutes/month on LiveKit Cloud.
"""

import config
from database.models import Segment
from typing import List, Optional, Dict
import uuid
import asyncio
import datetime


class LiveKitHandler:
    """Manages LiveKit rooms and token generation for clinical interviews."""

    def __init__(self):
        if not config.LIVEKIT_URL or not config.LIVEKIT_API_KEY or not config.LIVEKIT_API_SECRET:
            raise ValueError(
                "LiveKit credentials not set. Add LIVEKIT_URL, LIVEKIT_API_KEY, "
                "and LIVEKIT_API_SECRET to your .env file.\n"
                "Get free credentials at: https://cloud.livekit.io"
            )

        self.url = config.LIVEKIT_URL
        self.api_key = config.LIVEKIT_API_KEY
        self.api_secret = config.LIVEKIT_API_SECRET

    def create_room(self, room_name: str) -> Dict:
        """
        Create a LiveKit room for the interview session.

        LiveKit rooms are created automatically when the first participant joins,
        but we use the API to create explicitly for better control.

        Returns:
            Dict with room info including name and sid.
        """
        from livekit import api

        async def _create():
            lk = api.LiveKitAPI(self.url, self.api_key, self.api_secret)
            try:
                room = await lk.room.create_room(
                    api.CreateRoomRequest(
                        name=room_name,
                        empty_timeout=300,  # Close after 5 min if empty
                        max_participants=5,
                    )
                )
                return {
                    "name": room.name,
                    "sid": room.sid,
                    "created": True,
                }
            finally:
                await lk.aclose()

        return asyncio.run(_create())

    def generate_token(
        self,
        room_name: str,
        participant_name: str,
        role: str = "CLINICIAN",
    ) -> str:
        """
        Generate a JWT token for a participant to join the room.

        The participant's role (CLINICIAN/PATIENT) is embedded in the token
        metadata for downstream processing.

        Args:
            room_name: Name of the LiveKit room
            participant_name: Display name for the participant
            role: "CLINICIAN" or "PATIENT"

        Returns:
            JWT token string
        """
        from livekit import api

        token = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity(f"{role.lower()}_{participant_name.lower().replace(' ', '_')}")
            .with_name(f"{participant_name} ({role})")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .with_metadata(f'{{"role": "{role}", "name": "{participant_name}"}}')
            .with_ttl(datetime.timedelta(hours=1))
        )
        return token.to_jwt()

    def generate_join_url(self, room_name: str, token: str) -> str:
        """
        Generate a LiveKit Meet join URL for browser-based participation.

        Uses LiveKit's hosted meet app (meet.livekit.io) which provides
        a ready-made WebRTC interface — no custom frontend needed.
        """
        # Convert wss:// to the project identifier for meet URL
        # e.g. wss://my-project.livekit.cloud → my-project
        host = self.url.replace("wss://", "").replace("ws://", "")
        return f"https://meet.livekit.io/custom?liveKitUrl={self.url}&token={token}"

    def list_rooms(self) -> List[Dict]:
        """List all active LiveKit rooms."""
        from livekit import api

        async def _list():
            lk = api.LiveKitAPI(self.url, self.api_key, self.api_secret)
            try:
                response = await lk.room.list_rooms(api.ListRoomsRequest())
                return [
                    {"name": r.name, "sid": r.sid, "num_participants": r.num_participants}
                    for r in response.rooms
                ]
            finally:
                await lk.aclose()

        return asyncio.run(_list())

    def list_participants(self, room_name: str) -> List[Dict]:
        """List participants in a room."""
        from livekit import api

        async def _list():
            lk = api.LiveKitAPI(self.url, self.api_key, self.api_secret)
            try:
                response = await lk.room.list_participants(
                    api.ListParticipantsRequest(room=room_name)
                )
                return [
                    {
                        "identity": p.identity,
                        "name": p.name,
                        "metadata": p.metadata,
                        "state": str(p.state),
                    }
                    for p in response.participants
                ]
            finally:
                await lk.aclose()

        return asyncio.run(_list())

    def delete_room(self, room_name: str) -> bool:
        """Delete/close a LiveKit room."""
        from livekit import api

        async def _delete():
            lk = api.LiveKitAPI(self.url, self.api_key, self.api_secret)
            try:
                await lk.room.delete_room(
                    api.DeleteRoomRequest(room=room_name)
                )
                return True
            finally:
                await lk.aclose()

        return asyncio.run(_delete())

    @staticmethod
    def process_audio_for_participant(
        audio_path: str,
        interview_id: str,
        speaker_role: str,
        speaker_name: str = "",
    ) -> List[Segment]:
        """
        Process a single participant's recorded audio track.

        Since LiveKit separates audio per-participant, we get perfect
        speaker attribution without any diarization.

        Args:
            audio_path: Path to the participant's audio file
            interview_id: Interview this belongs to
            speaker_role: "PATIENT" or "CLINICIAN"
            speaker_name: Display name

        Returns:
            List[Segment] with perfect speaker labels
        """
        from audio.transcribe import WhisperTranscriber

        transcriber = WhisperTranscriber()
        trans_segments = transcriber.transcribe(audio_path)

        display = f"{speaker_name} ({speaker_role})" if speaker_name else speaker_role

        segments = []
        for seg in trans_segments:
            segments.append(
                Segment(
                    interview_id=interview_id,
                    segment_id=str(uuid.uuid4())[:8],
                    start_ms=seg["start_ms"],
                    end_ms=seg["end_ms"],
                    speaker_raw=display,
                    speaker_role=speaker_role,
                    text=seg["text"],
                    source_mode="live",
                )
            )

        return segments
