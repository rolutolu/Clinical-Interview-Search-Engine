"""
Supabase client for all database operations.

This module is the ONLY way any other module should interact with the database.
All team members import from here — never call supabase directly.

Provides:
    - CRUD for interviews, segments, patient profiles
    - Vector similarity search (for semantic retrieval)
    - Full-text search (for lexical retrieval)
    - Speaker-aware filtering (patient-only, clinician-only, combined)
"""

import config
from database.models import Segment, Interview, PatientProfile
from typing import List, Optional, Dict
from supabase import create_client, Client
import json


class SupabaseClient:
    """Singleton-style database client. Initialize once, use everywhere."""

    def __init__(self):
        if not config.SUPABASE_URL or not config.SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in .env file. "
                "Get them from: Supabase Dashboard → Settings → API"
            )
        self.client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

    # ═══════════════════════════════════════
    # Connection Test
    # ═══════════════════════════════════════

    def test_connection(self) -> bool:
        """Test that Supabase is reachable and tables exist."""
        try:
            result = self.client.table("interviews").select("interview_id").limit(1).execute()
            return True
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
            return False

    # ═══════════════════════════════════════
    # Interview CRUD
    # ═══════════════════════════════════════

    def create_interview(self, interview: Interview) -> dict:
        """Insert a new interview record."""
        data = interview.to_db_dict()
        # speaker_map is a dict — Supabase stores it as JSONB
        data["speaker_map"] = json.dumps(data["speaker_map"])
        result = self.client.table("interviews").insert(data).execute()
        return result.data[0] if result.data else {}

    def get_interview(self, interview_id: str) -> Optional[dict]:
        """Fetch a single interview by ID."""
        result = (
            self.client.table("interviews")
            .select("*")
            .eq("interview_id", interview_id)
            .execute()
        )
        if result.data:
            row = result.data[0]
            if isinstance(row.get("speaker_map"), str):
                row["speaker_map"] = json.loads(row["speaker_map"])
            return row
        return None

    def list_interviews(self) -> List[dict]:
        """List all interviews, newest first."""
        result = (
            self.client.table("interviews")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        for row in result.data:
            if isinstance(row.get("speaker_map"), str):
                row["speaker_map"] = json.loads(row["speaker_map"])
        return result.data

    def update_interview_speaker_map(
        self, interview_id: str, speaker_map: dict
    ) -> dict:
        """Update speaker role mapping for an interview."""
        result = (
            self.client.table("interviews")
            .update({"speaker_map": json.dumps(speaker_map)})
            .eq("interview_id", interview_id)
            .execute()
        )
        return result.data[0] if result.data else {}

    # ═══════════════════════════════════════
    # Segment CRUD
    # ═══════════════════════════════════════

    def insert_segments(self, segments: List[Segment]) -> int:
        """
        Bulk insert segments into the database.
        Returns the number of segments inserted.
        """
        if not segments:
            return 0

        rows = []
        for seg in segments:
            row = seg.to_db_dict()
            # Remove None keywords
            if not row.get("keywords"):
                row["keywords"] = []
            rows.append(row)

        result = self.client.table("segments").insert(rows).execute()
        return len(result.data)

    def get_segments(
        self,
        interview_id: str,
        speaker_role: Optional[str] = None,
    ) -> List[dict]:
        """
        Get all segments for an interview, optionally filtered by speaker role.

        Args:
            interview_id: The interview to fetch segments for
            speaker_role: None = all, "PATIENT" = patient only, "CLINICIAN" = clinician only
        """
        query = (
            self.client.table("segments")
            .select("segment_id, interview_id, start_ms, end_ms, speaker_raw, speaker_role, text, source_mode, keywords, created_at")
            .eq("interview_id", interview_id)
            .order("start_ms", desc=False)
        )

        if speaker_role:
            query = query.eq("speaker_role", speaker_role)

        result = query.execute()
        return result.data

    def update_segment_roles(
        self, interview_id: str, speaker_map: dict
    ) -> int:
        """
        Apply speaker role mapping to all segments in an interview.
        speaker_map: {"SPEAKER_0": "PATIENT", "SPEAKER_1": "CLINICIAN"}
        Returns number of segments updated.
        """
        count = 0
        for raw_label, role in speaker_map.items():
            result = (
                self.client.table("segments")
                .update({"speaker_role": role})
                .eq("interview_id", interview_id)
                .eq("speaker_raw", raw_label)
                .execute()
            )
            count += len(result.data)
        return count

    def update_segment_embedding(
        self, segment_id: str, embedding: List[float]
    ) -> bool:
        """Store an embedding vector for a single segment."""
        result = (
            self.client.table("segments")
            .update({"embedding": embedding})
            .eq("segment_id", segment_id)
            .execute()
        )
        return len(result.data) > 0

    def bulk_update_embeddings(
        self, updates: List[Dict]
    ) -> int:
        """
        Bulk update embeddings. Each dict: {"segment_id": ..., "embedding": [...]}
        Returns count of updated segments.
        """
        count = 0
        for item in updates:
            if self.update_segment_embedding(item["segment_id"], item["embedding"]):
                count += 1
        return count

    # ═══════════════════════════════════════
    # Vector Similarity Search (Semantic)
    # ═══════════════════════════════════════

    def vector_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        interview_id: Optional[str] = None,
        speaker_role: Optional[str] = None,
    ) -> List[dict]:
        """
        Semantic search using pgvector cosine similarity.
        Calls the match_segments RPC function defined in schema.sql.

        Args:
            query_embedding: 384-dim vector from sentence-transformers
            k: Number of results to return
            interview_id: Filter to specific interview (optional)
            speaker_role: "PATIENT", "CLINICIAN", or None for combined
        """
        params = {
            "query_embedding": query_embedding,
            "match_count": k,
        }
        if interview_id:
            params["filter_interview_id"] = interview_id
        if speaker_role:
            params["filter_speaker_role"] = speaker_role

        result = self.client.rpc("match_segments", params).execute()
        return result.data

    # ═══════════════════════════════════════
    # Full-Text Search (Lexical)
    # ═══════════════════════════════════════

    def text_search(
        self,
        query: str,
        k: int = 5,
        interview_id: Optional[str] = None,
        speaker_role: Optional[str] = None,
    ) -> List[dict]:
        """
        Lexical search using Postgres full-text search (ts_rank_cd).
        Calls the search_segments_text RPC function defined in schema.sql.

        Args:
            query: Natural language search query
            k: Number of results
            interview_id: Filter to specific interview (optional)
            speaker_role: "PATIENT", "CLINICIAN", or None for combined
        """
        params = {
            "query_text": query,
            "match_count": k,
        }
        if interview_id:
            params["filter_interview_id"] = interview_id
        if speaker_role:
            params["filter_speaker_role"] = speaker_role

        result = self.client.rpc("search_segments_text", params).execute()
        return result.data

    # ═══════════════════════════════════════
    # Patient Profile CRUD
    # ═══════════════════════════════════════

    def create_profile(self, profile: PatientProfile) -> dict:
        """Insert a patient profile."""
        data = profile.to_db_dict()
        result = self.client.table("patient_profiles").insert(data).execute()
        return result.data[0] if result.data else {}

    def get_profile(self, interview_id: str) -> Optional[dict]:
        """Get patient profile for an interview."""
        result = (
            self.client.table("patient_profiles")
            .select("*")
            .eq("interview_id", interview_id)
            .execute()
        )
        return result.data[0] if result.data else None

    # ═══════════════════════════════════════
    # Evaluation Labels
    # ═══════════════════════════════════════

    def get_eval_labels(self) -> List[dict]:
        """Get all evaluation labels for Precision@K / Recall@K."""
        result = self.client.table("eval_labels").select("*").execute()
        return result.data

    def insert_eval_labels(self, labels: List[dict]) -> int:
        """Bulk insert evaluation labels."""
        result = self.client.table("eval_labels").insert(labels).execute()
        return len(result.data)

    # ═══════════════════════════════════════
    # Utility
    # ═══════════════════════════════════════

    def delete_interview(self, interview_id: str) -> bool:
        """Delete an interview and all its segments (CASCADE)."""
        result = (
            self.client.table("interviews")
            .delete()
            .eq("interview_id", interview_id)
            .execute()
        )
        return len(result.data) > 0

    def get_segment_count(self, interview_id: str) -> int:
        """Get segment count for an interview."""
        result = (
            self.client.table("segments")
            .select("segment_id", count="exact")
            .eq("interview_id", interview_id)
            .execute()
        )
        return result.count or 0
