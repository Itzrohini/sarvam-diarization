# database/crud.py
# ============================================================
# CRUD = Create, Read, Update, Delete
# These are the 4 basic database operations.
# This file contains all functions that touch the database.
# Keeping them here (not in main.py) keeps code organized.
# ============================================================

from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from database.models import Transcript
from datetime import datetime
from typing import Optional
from core.logger import get_logger

logger = get_logger(__name__)


def save_transcript(
    db: Session,
    filename: str,
    duration_seconds: float,
    total_speakers: int,
    full_transcript: str,
    language_detected: Optional[str],
    language_requested: str,
    processing_time_seconds: float,
    segments: list,
) -> Transcript:
    """
    Save a new transcript to the database.
    
    db.add() — stages the new row (not saved yet)
    db.commit() — actually saves to PostgreSQL
    db.refresh() — reloads from DB to get the auto-generated ID
    """
    transcript = Transcript(
        filename=filename,
        duration_seconds=duration_seconds,
        total_speakers=total_speakers,
        full_transcript=full_transcript,
        language_detected=language_detected,
        language_requested=language_requested,
        processing_time_seconds=processing_time_seconds,
        segments=segments,
    )
    db.add(transcript)
    db.commit()
    db.refresh(transcript)

    logger.info("Saved transcript ID %d to database.", transcript.id)
    return transcript


def get_all_transcripts(
    db: Session,
    skip: int = 0,
    limit: int = 50,
) -> list[Transcript]:
    """
    Get all transcripts, newest first.
    
    skip + limit = pagination
    Example: skip=0, limit=10 → first 10 results
             skip=10, limit=10 → next 10 results
    """
    return (
        db.query(Transcript)
        .order_by(desc(Transcript.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_transcript_by_id(db: Session, transcript_id: int) -> Optional[Transcript]:
    """Get one specific transcript by its ID."""
    return db.query(Transcript).filter(Transcript.id == transcript_id).first()


def search_transcripts(
    db: Session,
    language: Optional[str] = None,
    speaker_count: Optional[int] = None,
    keyword: Optional[str] = None,
    limit: int = 50,
) -> list[Transcript]:
    """
    Search transcripts with filters.
    
    Examples:
      language="kn-IN" → only Kannada transcripts
      speaker_count=2   → only 2-speaker conversations
      keyword="sugar"   → transcripts containing "sugar"
    """
    query = db.query(Transcript)

    if language:
        query = query.filter(Transcript.language_detected == language)

    if speaker_count:
        query = query.filter(Transcript.total_speakers == speaker_count)

    if keyword:
        # ilike = case-insensitive LIKE search
        # % means "anything before or after"
        query = query.filter(
            Transcript.full_transcript.ilike(f"%{keyword}%")
        )

    return query.order_by(desc(Transcript.created_at)).limit(limit).all()


def delete_transcript(db: Session, transcript_id: int) -> bool:
    """Delete a transcript by ID. Returns True if deleted, False if not found."""
    transcript = get_transcript_by_id(db, transcript_id)
    if not transcript:
        return False
    db.delete(transcript)
    db.commit()
    logger.info("Deleted transcript ID %d.", transcript_id)
    return True