# database/models.py
# ============================================================
# WHY SQLALCHEMY MODELS?
# Instead of writing raw SQL like:
#   "CREATE TABLE transcripts (id SERIAL, transcript TEXT...)"
# SQLAlchemy lets you write Python classes that automatically
# become database tables. Cleaner, safer, easier to change later.
# ============================================================

from sqlalchemy import (
    Column, Integer, String, Float, 
    DateTime, Text, JSON
)
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime


class Base(DeclarativeBase):
    """
    Base class all models inherit from.
    SQLAlchemy uses this to track all your tables.
    """
    pass


class Transcript(Base):
    """
    One row = one processed audio file.
    
    Think of this like an Excel sheet where:
    - Each column is a field below
    - Each row is one audio transcription job
    """
    __tablename__ = "transcripts"

    # Primary key — auto-increments (1, 2, 3...)
    # Every row needs a unique ID
    id = Column(Integer, primary_key=True, autoincrement=True)

    # When was this transcript created?
    # default=datetime.utcnow means it's set automatically
    created_at = Column(DateTime, default=datetime.utcnow)

    # Audio file details
    filename = Column(String(255), nullable=True)
    duration_seconds = Column(Float, nullable=False)
    total_speakers = Column(Integer, nullable=False)

    # The full transcript text — Text allows unlimited length
    # String(255) would be limited to 255 characters
    full_transcript = Column(Text, nullable=False)

    # Language detected by Sarvam
    language_detected = Column(String(20), nullable=True)

    # Language the user selected
    language_requested = Column(String(20), default="unknown")

    # Processing time in seconds
    processing_time_seconds = Column(Float, nullable=False)

    # Store all segments as JSON — flexible, no separate table needed
    # JSON column stores Python dicts/lists directly
    # Example: [{"speaker": "Shop Owner", "transcript": "...", ...}]
    segments = Column(JSON, nullable=False, default=list)

    def to_dict(self) -> dict:
        """Convert this database row to a Python dict for JSON response."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "filename": self.filename,
            "duration_seconds": self.duration_seconds,
            "total_speakers": self.total_speakers,
            "full_transcript": self.full_transcript,
            "language_detected": self.language_detected,
            "language_requested": self.language_requested,
            "processing_time_seconds": self.processing_time_seconds,
            "segments": self.segments,
        }

def seconds_to_hms(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS format.
    Example: 39.075 → "00:00:39"
             3661.5 → "01:01:01"
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"