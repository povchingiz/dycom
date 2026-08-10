"""
Simple password authentication for FaceSim demo.
Uses session cookies, no database required.
"""
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, status
from fastapi.security import HTTPBasicCredentials
from pydantic import BaseModel


class Session(BaseModel):
    session_id: str
    created_at: datetime
    dicom_path: Optional[str] = None
    nifti_path: Optional[str] = None
    seg_dir: Optional[str] = None
    stl_dir: Optional[str] = None
    zip_path: Optional[str] = None
    status: str = "created"  # created, processing, completed, failed, downloaded
    error: Optional[str] = None
    # Surgical plan requested at upload, and the simulation summary produced for it.
    scenario: Optional[dict] = None
    simulation: Optional[dict] = None


# In-memory session storage (cleaned on server restart)
sessions: dict[str, Session] = {}

# Password from environment
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD")
if not DEMO_PASSWORD:
    raise RuntimeError("DEMO_PASSWORD environment variable not set. Copy .env.example to .env and set a password.")


def verify_password(password: str) -> bool:
    """Check if provided password matches DEMO_PASSWORD."""
    return secrets.compare_digest(password, DEMO_PASSWORD)


def create_session() -> Session:
    """Create a new session with unique ID."""
    session_id = secrets.token_urlsafe(16)
    session = Session(
        session_id=session_id,
        created_at=datetime.now(),
    )
    sessions[session_id] = session
    return session


def get_session(session_id: str) -> Optional[Session]:
    """Retrieve session by ID."""
    return sessions.get(session_id)


def cleanup_old_sessions(max_age_hours: int = 24 * 7) -> int:
    """Remove sessions older than max_age_hours. Returns count of removed sessions."""
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    old_ids = [
        sid for sid, session in sessions.items()
        if session.created_at < cutoff
    ]
    for sid in old_ids:
        del sessions[sid]
    return len(old_ids)
