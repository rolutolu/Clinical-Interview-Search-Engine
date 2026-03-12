"""
Quick connection test — run this after setting up .env and Supabase schema.

Usage:
    python tests/test_connection.py

Expected output: All checks pass ✅
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from database.models import Segment, Interview, PatientProfile
from database.supabase_client import SupabaseClient


def test_env_vars():
    """Check that all required environment variables are set."""
    print("── Checking environment variables ──")
    checks = {
        "SUPABASE_URL": bool(config.SUPABASE_URL),
        "SUPABASE_KEY": bool(config.SUPABASE_KEY),
        "GROQ_API_KEY": bool(config.GROQ_API_KEY),
        "HF_TOKEN": bool(config.HF_TOKEN),
    }
    for name, ok in checks.items():
        print(f"  {'[OK]' if ok else '[X]'} {name}")
    return all(checks.values())


def test_supabase_connection():
    """Test Supabase connection and table existence."""
    print("\n── Testing Supabase connection ──")
    try:
        db = SupabaseClient()
        ok = db.test_connection()
        print(f"  {'[OK]' if ok else '[X]'} Supabase connection")
        return ok
    except Exception as e:
        print(f"  [X] Supabase connection failed: {e}")
        return False


def test_crud_operations():
    """Test basic CRUD: create interview -> create segment -> read -> delete."""
    print("\n── Testing CRUD operations ──")
    db = SupabaseClient()
    test_id = "test_001"

    try:
        # Create interview
        interview = Interview(
            interview_id=test_id,
            title="Test Interview",
            source_mode="offline",
            speaker_map={"SPEAKER_0": "PATIENT", "SPEAKER_1": "CLINICIAN"},
        )
        db.create_interview(interview)
        print("  [OK] Create interview")

        # Create segments
        segments = [
            Segment(
                interview_id=test_id,
                start_ms=0,
                end_ms=5000,
                speaker_raw="SPEAKER_0",
                speaker_role="PATIENT",
                text="I've been having headaches for two weeks.",
                source_mode="offline",
            ),
            Segment(
                interview_id=test_id,
                start_ms=5000,
                end_ms=10000,
                speaker_raw="SPEAKER_1",
                speaker_role="CLINICIAN",
                text="Can you describe the pain? Is it constant or intermittent?",
                source_mode="offline",
            ),
        ]
        count = db.insert_segments(segments)
        print(f"  [OK] Insert segments ({count} inserted)")

        # Read segments
        fetched = db.get_segments(test_id)
        assert len(fetched) == 2, f"Expected 2, got {len(fetched)}"
        print(f"  [OK] Read segments ({len(fetched)} found)")

        # Read patient-only
        patient_segs = db.get_segments(test_id, speaker_role="PATIENT")
        assert len(patient_segs) == 1
        print(f"  [OK] Speaker filter (patient-only: {len(patient_segs)} found)")

        # Read interview
        iv = db.get_interview(test_id)
        assert iv is not None
        print(f"  [OK] Read interview: {iv['title']}")

        # Cleanup
        db.delete_interview(test_id)
        print("  [OK] Delete interview (cascade)")

        return True

    except Exception as e:
        print(f"  [X] CRUD test failed: {e}")
        # Attempt cleanup
        try:
            db.delete_interview(test_id)
        except:
            pass
        return False


def test_data_model():
    """Test that data model utilities work correctly."""
    print("\n── Testing data models ──")

    seg = Segment(
        interview_id="test",
        start_ms=65000,
        end_ms=78000,
        speaker_raw="SPEAKER_0",
        speaker_role="PATIENT",
        text="My chest hurts when I breathe deeply.",
    )

    # Test citation tag
    tag = seg.citation_tag()
    assert "PATIENT" in tag
    print(f"  [OK] Citation tag: {tag}")

    # Test time range
    time_str = seg.time_range_str()
    assert "1:05" in time_str
    print(f"  [OK] Time range: {time_str}")

    # Test to_dict (no embedding)
    d = seg.to_dict()
    assert "embedding" not in d
    print(f"  [OK] to_dict excludes embedding")

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Clinical Interview IR System - Connection Test")
    print("=" * 50)

    results = []
    results.append(("Environment Variables", test_env_vars()))
    results.append(("Data Models", test_data_model()))

    if results[0][1]:  # Only test DB if env vars are set
        results.append(("Supabase Connection", test_supabase_connection()))
        if results[2][1]:  # Only test CRUD if connection works
            results.append(("CRUD Operations", test_crud_operations()))

    print("\n" + "=" * 50)
    print("RESULTS:")
    all_pass = True
    for name, passed in results:
        print(f"  {'[OK]' if passed else '[X]'} {name}")
        if not passed:
            all_pass = False

    print("=" * 50)
    if all_pass:
        print("All tests passed! Your system is ready.")
    else:
        print("Some tests failed. Check your .env file and Supabase setup.")

    sys.exit(0 if all_pass else 1)
